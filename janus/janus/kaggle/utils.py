import ast


class CollectImports(ast.NodeVisitor):
    def __init__(self):
        self.imports = set([])

    def visit_Import(self, node):
        self.imports.update([a.name for a in node.names])

    def visit_ImportFrom(self, node):
        self.imports.add(node.module)

    def run(self, src):
        if isinstance(src, str):
            src = ast.parse(src)
        self.visit(src)
        return self.imports
