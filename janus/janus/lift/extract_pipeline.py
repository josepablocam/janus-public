#!/usr/bin/env python3
from argparse import ArgumentParser
import inspect
import os
import pickle
import textwrap

import networkx as nx
import sklearn
import sklearn.pipeline

from plpy.rewrite.expr_lifter import lift_source
from plpy.analyze.dynamic_tracer import DynamicDataTracer
from plpy.analyze.dynamic_trace_events import ExecLine
from plpy.analyze.graph_builder import DynamicTraceToGraph

from janus.pipeline import pipeline_to_tree as pt


def is_file_path(s):
    return len(s.split("\n")) == 1 and os.path.exists(s)


def try_wrapped(func, msg):
    try:
        return (func(), 0)
    except Exception as err:
        print(msg)
        print(err)
        return (None, 1)


def is_relevant_lib(_input):
    libs = ("sklearn", "xgboost")
    try:
        if isinstance(_input, str):
            return _input.startswith(libs)
        else:
            return _input.__module__.startswith(libs)
    except:
        return False


def clean_nones_in_graph(g):
    # I annoyingly made some things None
    # rather than empty lists...fix that design error here
    for node_id in g.nodes:
        n = g.nodes[node_id]
        if n["complete_defs"] is None:
            n["complete_defs"] = []
        if n["defs"] is None:
            n["defs"] = []
        if n["uses"] is None:
            n["uses"] = []
    return g


def sanitize_pickle_graph(g):
    for node_id in g.nodes:
        n = g.nodes[node_id]
        for d in n["complete_defs"]:
            d.extra = {}
        for d in n["defs"]:
            d.extra = {}
        for u in n["uses"]:
            u.extra = {}
    return g


def is_predict_call_node(node):
    if node["calls"] is None:
        return False
    for c in node["calls"]:
        d = c.details
        if not is_relevant_lib(d["module"]):
            continue
        if d["qualname"].endswith(("predict", "predict_proba")):
            return True
    return False


def is_fit_call_node(node):
    if node["calls"] is None:
        return False
    for c in node["calls"]:
        d = c.details
        if not is_relevant_lib(d["module"]):
            continue
        if d["qualname"].endswith(("fit", "fit_transform")):
            return True
    return False


def find_calls_sat_predicate(graph, pred):
    return [node_id for node_id in graph.nodes if pred(graph.nodes[node_id])]


def find_pipeline_seeds(graph):
    seed_ids = find_calls_sat_predicate(
        graph,
        lambda node: is_predict_call_node(node) or is_fit_call_node(node),
    )
    return set(seed_ids)


def remove_subgraphs(graphs):
    node_sets = []
    for g in graphs:
        node_sets.append(set(g.nodes))

    clean_graphs = []
    for i, g in enumerate(graphs):
        add = True
        for j, other in enumerate(graphs):
            if i == j:
                continue
            s1 = node_sets[i]
            s2 = node_sets[j]
            if s1.issubset(s2):
                add = False
                break
        if add:
            clean_graphs.append(g)
    return clean_graphs


def get_graph_slices(graph, seed_node_ids):
    reversed_graph = graph.reverse(copy=False)
    slices = []
    for node_id in seed_node_ids:
        slice_nodes = nx.dfs_tree(reversed_graph, node_id)
        _slice = graph.subgraph(slice_nodes)
        slices.append(_slice)
    return slices


def add_obj_to_variable(v, _globals, _locals):
    if v.name in _locals:
        obj = _locals[v.name]
    elif v.name in _globals:
        obj = _globals[v.name]
    else:
        return

    v.extra["obj"] = obj


def add_line_by_line_exec_info(nodes):
    tmp_globals = {}
    tmp_locals = {}
    for n in nodes:
        try:
            exec (n["src"], tmp_globals, tmp_locals)
            for d in n["complete_defs"]:
                add_obj_to_variable(d, tmp_globals, tmp_locals)
            for u in n["uses"]:
                add_obj_to_variable(u, _globals, _locals)
        except:
            # see if can try anything else..
            continue
    return nodes


def execute_graph_line_by_line(graph):
    ordered_ids = nx.algorithms.dag.topological_sort(graph)
    ordered_nodes = [graph.nodes[ix] for ix in ordered_ids]
    # now only execline nodes
    exec_nodes = [n for n in ordered_nodes if isinstance(n["event"], ExecLine)]
    annotated = add_line_by_line_exec_info(exec_nodes)
    return graph


def get_all_lib_objs_from_root(obj, acc=None):
    if acc is None:
        acc = []

    if not is_relevant_lib(obj):
        return acc

    acc.append(obj)

    try:
        for v in obj.get_params().values():
            get_all_lib_objs_from_root(v, acc)
        return acc
    except:
        return acc


def extract_pipeline_components(graph, path_lengths):
    accounted = set()
    components_and_ids = []

    ordered_ids = nx.algorithms.dag.topological_sort(graph)
    ordered_ids = list(ordered_ids)
    sink_node_id = ordered_ids[-1]
    sink_node = graph.nodes[sink_node_id]
    assert is_predict_call_node(sink_node) or is_fit_call_node(sink_node)

    for node_id in reversed(ordered_ids):
        # focus on definitions ... we'll pick up the full constructor then
        added = False
        n = graph.nodes[node_id]
        if n["complete_defs"] is not None:
            for var in n["complete_defs"]:
                if var in accounted:
                    continue
                if not "obj" in var.extra:
                    continue
                obj = var.extra["obj"]
                if id(obj) in accounted:
                    continue
                if not is_relevant_lib(obj):
                    continue
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    # can't add functions/methods
                    # need the object itself
                    continue
                if inspect.isclass(obj):
                    # it's a type --> can come from import statement lines
                    continue
                components_and_ids.append((obj, node_id))
                accounted.update(
                    [id(o) for o in get_all_lib_objs_from_root(obj)])
                added = True
        # if it's a direct use, means we're using it
        # in the constructor -- so we've accounted for it
        if added:
            for var in n["uses"]:
                accounted.add(var)

    # component order determined by length of path along dependency edges
    # to sink node
    l = path_lengths[sink_node_id]
    # shorter distances at the end
    for _, node_id in components_and_ids:
        assert node_id in l, "should be in graph"

    components_and_ids = sorted(
        components_and_ids, key=lambda x: l[x[1]], reverse=True)

    components = [c for c, _ in components_and_ids]
    return components


def build_pipeline(components):
    poss_clf = components[-1]
    if not sklearn.base.is_classifier(poss_clf):
        return None

    if len(components) == 1 and isinstance(poss_clf,
                                           sklearn.pipeline.Pipeline):
        # don't wrap again, trivial
        return poss_clf

    steps = [("step_{}".format(ix), c) for ix, c in enumerate(components)]
    return sklearn.pipeline.Pipeline(steps)


def add_wrapper_code(src):
    wrapped_src = """
    #wrapper code added by janus
    import matplotlib
    matplotlib.use('Agg')
    """
    return textwrap.dedent(wrapped_src) + "\n" + src


class PipelineLifter(object):
    def __init__(self, _input):
        self.input = _input
        if is_file_path(_input):
            assert os.path.splitext(_input)[1] == ".py"
            with open(_input, "r") as fin:
                self.src = fin.read()
        else:
            self.src = _input

        self.src = textwrap.dedent(self.src)
        # add some some helper code
        self.src = add_wrapper_code(self.src)
        self.failed = False
        self.graph = None
        self.set_dynamic_graph()
        if not self.failed:
            self.pipelines = self.extract_pipelines()
        else:
            self.pipelines = []

    def set_dynamic_graph(self):
        lifted_src, status = try_wrapped(lambda: lift_source(self.src),
                                         "Failed to lift")
        self.lifted_src = lifted_src
        if status:
            self.failed = True
            self.graph = None
            return

        tracer = DynamicDataTracer(loop_bound=1)
        _, status = try_wrapped(lambda: tracer.run(self.lifted_src),
                                "Failed to trace")
        if status:
            self.failed = True
            self.graph = None
            return

        builder = DynamicTraceToGraph(ignore_unknown=True, memory_refinement=0)
        g, status = try_wrapped(lambda: builder.run(tracer),
                                "Failed to build graph")
        if status:
            self.failed = True
            self.graph = None
            return

        if os.path.exists("_instrumented.py"):
            os.remove("_instrumented.py")

        self.graph = clean_nones_in_graph(g)

    def extract_pipelines(self):
        assert self.graph is not None
        seed_node_ids = find_pipeline_seeds(self.graph)
        slices = get_graph_slices(self.graph, seed_node_ids)
        slices = remove_subgraphs(slices)

        path_lengths = nx.all_pairs_shortest_path_length(
            self.graph.reverse(copy=True))
        path_lengths = dict(path_lengths)
        pipelines = []
        hashes = set([])
        for _slice in slices:
            annotated_slice = execute_graph_line_by_line(_slice)
            components = extract_pipeline_components(annotated_slice,
                                                     path_lengths)

            pipeline = build_pipeline(components)
            h = pt.md5(pipeline)

            if pipeline is not None and h not in hashes:
                pipelines.append(pipeline)
                hashes.add(h)
        return pipelines

    def dump(self, output_path):
        # need to sanitize the graph
        self.graph = sanitize_pickle_graph(self.graph)
        with open(output_path, "wb") as fout:
            pickle.dump(self, fout)


def get_args():
    parser = ArgumentParser(description="Lift script into pipelines")
    parser.add_argument("--script", type=str, help="Input script")
    parser.add_argument(
        "--output", type=str, help="Path to save pickled results")
    return parser.parse_args()


def main():
    args = get_args()
    result = PipelineLifter(args.script)
    if result.failed:
        print("Failed processing " + args.script)
    else:
        print("Number pipelines extracted: {}".format(len(result.pipelines)))
        result.dump(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
