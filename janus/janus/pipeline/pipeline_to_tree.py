import copy
import hashlib
import importlib

import graphviz
import numpy as np
import sklearn
import sklearn.pipeline
import tpot.builtins
import zss

PARAM_NODE_PREFIX = "param_"
PARAM_NODE_TYPE = "param"
COMP_NODE_TYPE = "component"
ROOT_NODE_TYPE = "root"
TARGET_APIS = (
    "sklearn",
    "xgboost",
    "tpot",
)


class PipelineNode(object):
    def __init__(self, label, node_type=None, payload=None):
        self.label = label
        self.children = list()
        self.node_type = node_type
        self.payload = payload
        self.parent = None
        self.left = None
        self.right = None
        self.annotated = False

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    @staticmethod
    def get_payload(node):
        return node.payload

    def addkid(self, node, before=False):
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def annotate(self):
        if self.annotated:
            return
        left = None
        for c in self.children:
            if left is not None:
                left.right = c
            c.parent = self
            c.left = left
            c.annotate()
            left = c
        if len(self.children) > 0:
            self.children[-1].right = None
        self.annotated = True
        return self

    def siblings(self):
        assert self.annotated
        if self.parent is None:
            return []
        else:
            including_self = list(self.parent.children)
            including_self.remove(self)
            return including_self

    def replace_child(self, i, new_c):
        old_c = self.children[i]
        new_c.parent = self
        new_c.left = old_c.left
        new_c.right = old_c.right
        self.children[i] = new_c
        return self

    def delete_child_by_ix(self, i):
        old_c = self.children[i]
        old_c_left = old_c.left
        old_c_right = old_c.right
        if old_c_left is not None:
            old_c_left.right = old_c_right
        if old_c_right is not None:
            old_c_right.left = old_c_left
        del self.children[i]
        return self

    def delete_child_by_node(self, node):
        ix = self.children.index(node)
        self.delete_child_by_ix(ix)

    def delete_child(self, obj):
        if isinstance(obj, PipelineNode):
            self.delete_child_by_node(obj)
        elif isinstance(obj, int):
            self.delete_child_by_ix(obj)
        else:
            raise TypeError("obj must be of type PipelineNode or int")

    def insert_child(self, i, c):
        assert i >= 0 and i <= len(self.children)
        # new left will be at current i - 1
        left = None if i == 0 else self.children[i - 1]
        # new right will be at current i
        right = None if i == len(self.children) else self.children[i]

        c.left = left
        c.right = right
        c.parent = self

        if left is not None:
            left.right = c
        if right is not None:
            right.left = c

        self.children.insert(i, c)

    def set_children(self, new_children):
        # remove any dummy None nodes
        new_children = [c for c in new_children if c is not None]
        left = None
        for c in new_children:
            c.parent = self
            c.left = left
            if left is not None:
                left.right = c
            left = c
        if len(new_children) > 0:
            new_children[-1].right = None
        self.children = new_children
        return self


def shallow_copy(node):
    copy_node = copy.copy(node)
    copy_node.children = list(copy_node.children)
    return copy_node


def deep_copy(node):
    return copy.deepcopy(node)


def is_param_node(node):
    return node is not None and node.node_type == PARAM_NODE_TYPE


def is_component_node(node):
    return node is not None and node.node_type == COMP_NODE_TYPE


def is_root_node(node):
    return node is not None and node.node_type == ROOT_NODE_TYPE


def is_composition_node(node):
    return is_component_node(node) and node.label in set([
        "sklearn.pipeline.Pipeline",
        "sklearn.pipeline.FeatureUnion",
    ])


def mk_root_node():
    return PipelineNode("root", node_type=ROOT_NODE_TYPE)


def mk_param_node(*args, **kwargs):
    return PipelineNode(*args, node_type=PARAM_NODE_TYPE, **kwargs)


def mk_comp_node(*args, **kwargs):
    return PipelineNode(*args, node_type=COMP_NODE_TYPE, **kwargs)


def get_obj_label(obj):
    module_str = obj.__class__.__module__
    class_str = obj.__class__.__name__
    return module_str + "." + class_str


def is_object_in_target_api(obj):
    return get_obj_label(obj).startswith(TARGET_APIS)


def is_composition_op(obj):
    composition_types = (
        sklearn.pipeline.FeatureUnion,
        sklearn.pipeline.Pipeline,
    )

    if isinstance(obj, type):
        return issubclass(obj, composition_types)
    else:
        return isinstance(obj, composition_types)


def get_steps(obj):
    if isinstance(obj, sklearn.pipeline.FeatureUnion):
        return obj.transformer_list
    elif isinstance(obj, sklearn.pipeline.Pipeline):
        return obj.steps
    else:
        raise TypeError("No steps for {}".format(type(obj)))


def convert_obj_to_node(obj):
    obj_label = get_obj_label(obj)
    obj_node = mk_comp_node(obj_label)

    if is_composition_op(obj):
        children = get_steps(obj)
    else:
        children = list(obj.get_params(deep=False).items())
        # sort by param name to guarantee order
        children = sorted(children, key=lambda x: x[0])

    for child_name, child_obj in children:
        if is_object_in_target_api(child_obj):
            # if object is interesting
            child_node = convert_obj_to_node(child_obj)
            # and we keep track of this
            # store the name of the parameter, so we can rebuild
            child_node.payload = (child_name, None)
        else:
            # hyperparameter of interest
            param_label = "{}{}:{}".format(
                PARAM_NODE_PREFIX,
                child_name,
                child_obj,
            )
            payload = (child_name, child_obj)
            child_node = mk_param_node(param_label, payload=payload)
        obj_node.addkid(child_node)
    return obj_node


def is_pipeline_repr(pipeline):
    # not using isinstance(), since conflicts when reorganizing code
    # FIX: change this to perform isinstance() check after we finalize code
    # org
    try:
        return type(pipeline).__name__ == "PipelineNode"
    except AttributeError:
        return False


def to_tree(pipeline):
    if is_pipeline_repr(pipeline):
        return pipeline
    root = mk_root_node()
    pipeline_node = convert_obj_to_node(pipeline)
    root.addkid(pipeline_node)
    return root


def to_json(node):
    assert is_pipeline_repr(node), "Must be node representation"
    if is_root_node(node):
        assert len(node.children) == 1, "Root can only have 1 child"
        return to_json(node.children[0])
    elif is_component_node(node):
        return {node.label: [to_json(c) for c in node.children]}
    elif is_param_node(node):
        return node.payload
    else:
        raise TypeError("Unknown node type: {}".format(node.node_type))


def to_frozen_json(obj):
    if isinstance(obj, dict):
        return frozenset([(k, to_frozen_json(v)) for k, v in obj.items()])
    elif isinstance(obj, (tuple, list)):
        return tuple([to_frozen_json(o) for o in obj])
    else:
        return obj


def to_hashable_json(tree):
    raw_json = to_json(tree)
    frozen_json = to_frozen_json(raw_json)
    return frozen_json


def to_nested_labels(node):
    assert is_pipeline_repr(node), "Must be node representation"
    children = [to_nested_labels(c) for c in node.children]
    if len(children) == 0:
        return node.label
    else:
        return (node.label, children)


def binary_dist(d1, d2):
    return 1.0 if d1 != d2 else 0.0


def tree_edit_distance(pipeline1, pipeline2, return_operations=False):
    if not is_pipeline_repr(pipeline1):
        pipeline1 = to_tree(pipeline1)

    if not is_pipeline_repr(pipeline2):
        pipeline2 = to_tree(pipeline2)

    if not pipeline1.annotated and return_operations:
        pipeline1.annotate()

    if not pipeline2.annotated and return_operations:
        pipeline2.annotate()

    # TODO: note that we should eventually modify this to use something other
    # than binary distance
    return zss.simple_distance(
        pipeline1,
        pipeline2,
        get_children=PipelineNode.get_children,
        get_label=PipelineNode.get_label,
        label_dist=binary_dist,
        return_operations=return_operations,
    )


def extract_operators(pipeline):
    stack = [to_tree(pipeline)]
    ops = []
    while len(stack) > 0:
        curr = stack.pop()
        if is_root_node(curr):
            stack.extend(curr.children)
        elif curr.label.startswith(TARGET_APIS):
            if not curr.label.startswith("sklearn.pipeline.Pipeline"):
                ops.append(curr.label)
            stack.extend(curr.children)
        else:
            continue
    return ops


def try_parse(value_str, cast_op):
    try:
        return cast_op(value_str)
    except ValueError:
        return None


def get_param_value(node):
    return node.payload[1]


def get_param_name(node):
    return node.payload[0]


def get_param_node_feature(node, use_param_value, parent):
    feature_name = parent.label + ":" + get_param_name(node)
    feature_value = get_param_value(node)
    if use_param_value and isinstance(
            feature_value, (float, int)) and not np.isnan(feature_value):
        feature = {feature_name: feature_value}
    else:
        feature_name = feature_name + ":" + str(feature_value)
        feature = {feature_name: 1.0}
    return feature


def extract_feature_node(node, param_values=False, parent=None):
    if is_param_node(node):
        return get_param_node_feature(
            node,
            use_param_value=param_values,
            parent=parent,
        )
    else:
        return {node.label: 1.0}


def traverse_tree_for_features(node, param_values=False, parent=None):
    feat = extract_feature_node(node, param_values=param_values, parent=parent)
    acc = [feat]

    for child in node.children:
        child_features = traverse_tree_for_features(child, parent=node)
        acc.extend(child_features)
    return acc


def aggregate_features(feature_list):
    feature_dict = {}
    for feature in feature_list:
        name, value = next(iter(feature.items()))
        if name not in feature_dict:
            feature_dict[name] = value
        else:
            feature_dict[name] += value
    return feature_dict


def extract_features(pipeline, param_values=False):
    tree = to_tree(pipeline)
    feature_list = traverse_tree_for_features(
        tree, param_values=param_values, parent=None)
    feature_dict = aggregate_features(feature_list)
    return feature_dict


def get_constructor(obj, label=None):
    if label is None:
        label = obj.label
    path_steps = label.split(".")
    module_steps, basename = path_steps[:-1], path_steps[-1]
    module_path = ".".join(module_steps)
    return getattr(importlib.import_module(module_path), basename)


def compile_tree(obj, debug=False):
    if debug:
        import pdb
        pdb.set_trace()
    if is_param_node(obj):
        return obj.payload
    elif is_component_node(obj):
        constructor = get_constructor(obj)
        params = [compile_tree(c) for c in obj.children]
        if len(params) == 0:
            return constructor()
        elif issubclass(constructor, sklearn.pipeline.Pipeline):
            return sklearn.pipeline.make_pipeline(*params)
        elif issubclass(constructor, sklearn.pipeline.FeatureUnion):
            return sklearn.pipeline.make_union(*params)
        else:
            args = []
            kwargs = []
            for ix in range(len(params)):
                p = params[ix]
                if isinstance(p, tuple) and isinstance(p[0], str):
                    kwargs.extend(params[ix:])
                    break
                else:
                    args.append(p)
            return constructor(*args, **dict(kwargs))
    else:
        raise Exception("Unknown node type: {}".format(obj))


def to_pipeline(obj):
    if not is_pipeline_repr(obj):
        return obj
    if is_root_node(obj):
        assert len(obj.children) == 1, "Root has exactly 1 child"
        return to_pipeline(obj.children[0])
    else:
        assert is_component_node(obj), "Can't compile non components"
        return compile_tree(obj)


def md5(obj):
    # janky util
    t = to_tree(obj)
    j = to_json(t)
    s = str(j).encode("utf-8")
    return hashlib.md5(s).hexdigest()


def dot_stub_label(node):
    if is_root_node(node):
        return "root"
    elif is_param_node(node):
        return "p"
    elif is_component_node(node):
        if "Pipeline" in node.label:
            return "Pipeline"
        else:
            return "C"


class Dotifier(object):
    def __init__(self, obj, max_children=None, label_fun=None):
        self.node_ct = 0
        self.node_map = {}

        if max_children is not None:
            assert max_children > 0
        self.max_children = max_children

        self.label_fun = label_fun

        self.dot = graphviz.Digraph()
        t = to_tree(obj)
        self.to_dot(t)

    def add_node(self, n):
        if n not in self.node_map:
            node_id = "node_{}".format(self.node_ct)
            self.node_ct += 1
            self.node_map[n] = node_id
            if self.label_fun is not None:
                label = self.label_fun(n)
            else:
                label = n.label
            self.dot.node(node_id, label)

        return self.node_map[n]

    def add_edge(self, n1, n2):
        id1 = self.add_node(n1)
        id2 = self.add_node(n2)
        self.dot.edge(id1, id2)

    def to_dot(self, n):
        if is_root_node(n):
            assert len(n.children) == 1
            n = n.children[0]

        self.add_node(n)

        children = n.children
        if self.max_children is not None:
            children = children[:self.max_children]

        for c in children:
            self.add_node(c)
            self.add_edge(n, c)
            self.to_dot(c)

    def dump(self, path, view=False):
        self.dot.render(path, view=view)


def to_dot(obj, path=None, view=False, max_children=None, label_fun=None):
    dotter = Dotifier(obj, max_children=max_children, label_fun=label_fun)
    if path is not None:
        dotter.dump(path, view=view)
    return dotter.dot.source


class Texter(object):
    def __init__(self, obj, indent_str="  ", max_children=None):
        self.text = ""
        self.indent_str = indent_str
        self.ident_ct = 0

        if max_children is not None:
            assert max_children > 0
        self.max_children = max_children

        t = to_tree(obj)
        self.to_text(t)

    def add_node(self, n):
        if not is_param_node(n):
            label = n.label.split(".")[-1]
        else:
            label = "{}={}".format(*n.payload)
        self.text += ((self.ident_ct * self.indent_str) + label + "\n")

    def to_text(self, n):
        if is_root_node(n):
            assert len(n.children) == 1
            n = n.children[0]

        self.add_node(n)

        children = n.children
        if self.max_children is not None:
            children = children[:self.max_children]

        for c in nchildren:
            self.ident_ct += 1
            self.to_text(c)
            self.ident_ct -= 1


def to_text(obj, indent_str="  ", max_children=None):
    return Texter(obj, indent_str=indent_str, max_children=max_children).text
