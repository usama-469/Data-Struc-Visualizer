import ast
import os
import sys
import networkx as nx
import plotly.graph_objs as go
from typing import Dict, Tuple, List

TYPE_COLOR = {
    "list": "rgb(0,150,255)",
    "dict": "rgb(255,140,0)",
    "set": "rgb(50,200,50)",
    "tuple": "rgb(200,0,200)",
    "class": "rgb(255,0,0)",
    "function": "rgb(0,0,255)",
    "int": "rgb(200,200,200)",
    "str": "rgb(255,105,180)",
    "float": "rgb(180,180,255)",
    "bool": "rgb(150,150,150)",
    "module": "rgb(0,0,0)",
    "unknown": "rgb(120,120,120)",
}

def guess_type(node) -> str:
    if isinstance(node, ast.List): return "list"
    if isinstance(node, ast.Dict): return "dict"
    if isinstance(node, ast.Set): return "set"
    if isinstance(node, ast.Tuple): return "tuple"
    if isinstance(node, ast.Constant):
        t = type(node.value).__name__
        return t if t in TYPE_COLOR else "unknown"
    if isinstance(node, ast.Call):
        # Simple heuristics for constructors like list(), dict(), set()
        if isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            return name if name in TYPE_COLOR else "unknown"
    return "unknown"

def build_graph(py_path: str) -> nx.Graph:
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=py_path)

    G = nx.Graph()
    module_name = os.path.basename(py_path)
    G.add_node(module_name, kind="module", label=module_name)

    # Track variables and their types
    var_types: Dict[str, str] = {}

    # Add classes and functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            G.add_node(node.name, kind="class", label=node.name)
            G.add_edge(module_name, node.name, relation="contains")
            # Link methods
            for body in node.body:
                if isinstance(body, ast.FunctionDef):
                    fn_label = f"{node.name}.{body.name}()"
                    G.add_node(fn_label, kind="function", label=fn_label)
                    G.add_edge(node.name, fn_label, relation="method")
        elif isinstance(node, ast.FunctionDef):
            fn_label = f"{node.name}()"
            G.add_node(fn_label, kind="function", label=fn_label)
            G.add_edge(module_name, fn_label, relation="contains")
        elif isinstance(node, ast.Assign):
            # Left-hand side targets can be multiple
            val_type = guess_type(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    var_types[tgt.id] = val_type
                    G.add_node(tgt.id, kind=val_type, label=tgt.id)
                    G.add_edge(module_name, tgt.id, relation="var")
        elif isinstance(node, ast.AnnAssign):
            # Annotated assignments
            if isinstance(node.target, ast.Name):
                ann = ast.unparse(node.annotation) if hasattr(ast, "unparse") else "unknown"
                val_type = guess_type(node.value) if node.value else ann
                kind = val_type if val_type in TYPE_COLOR else (ann if ann in TYPE_COLOR else "unknown")
                var_types[node.target.id] = kind
                G.add_node(node.target.id, kind=kind, label=node.target.id)
                G.add_edge(module_name, node.target.id, relation="var")
        elif isinstance(node, ast.Call):
            # Link variables passed to calls
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = f"{node.func.id}()"
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{ast.unparse(node.func)}()" if hasattr(ast, "unparse") else f"{node.func.attr}()"
            if func_name:
                if not G.has_node(func_name):
                    G.add_node(func_name, kind="function", label=func_name)
                G.add_edge(module_name, func_name, relation="calls")
                for arg in node.args:
                    if isinstance(arg, ast.Name) and G.has_node(arg.id):
                        G.add_edge(arg.id, func_name, relation="arg")

    # Link keys/items within dicts, lists, sets if literals appear
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    parent = tgt.id
                    for k, v in zip(node.value.keys, node.value.values):
                        k_label = ast.unparse(k) if hasattr(ast, "unparse") else "key"
                        v_type = guess_type(v)
                        child_name = f"{parent}.{k_label}"
                        G.add_node(child_name, kind=v_type, label=child_name)
                        G.add_edge(parent, child_name, relation="dict-item")
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Set, ast.Tuple)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    parent = tgt.id
                    for idx, elt in enumerate(node.value.elts):
                        v_type = guess_type(elt)
                        child_name = f"{parent}[{idx}]"
                        G.add_node(child_name, kind=v_type, label=child_name)
                        G.add_edge(parent, child_name, relation="seq-item")

    return G

def layout_3d(G: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
    # Use spring layout in 3D by embedding 2D to 3D
    pos2d = nx.spring_layout(G, dim=2, k=0.6, seed=42)
    # Lift into 3D by adding a z jitter
    positions = {}
    import random
    for n, (x, y) in pos2d.items():
        z = (random.random() - 0.5) * 0.8
        positions[n] = (x, y, z)
    return positions

def graph_to_plotly_3d(G: nx.Graph):
    pos = layout_3d(G)

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # build hover texts for edges (repeat for the two endpoints, None for the separator)
    edge_hover = []
    for u, v in G.edges():
        rel = G.edges[u, v].get("relation", "")
        label = f"{u} → {v}"
        if rel:
            label = f"{label} ({rel})"
        edge_hover += [label, label, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgb(0,100,200)', width=3),
        hoverinfo='text',
        hovertext=edge_hover,
        name='relations'
    )

    # Nodes grouped by kind for coloring
    kinds = {}
    for n, data in G.nodes(data=True):
        kind = data.get("kind", "unknown")
        kinds.setdefault(kind, []).append(n)

    node_traces = []
    for kind, nodes in kinds.items():
        xs, ys, zs, texts = [], [], [], []
        for n in nodes:
            x, y, z = pos[n]
            xs.append(x); ys.append(y); zs.append(z); texts.append(G.nodes[n].get("label", n))
        node_traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text',
            text=texts,
            textposition='top center',
            marker=dict(size=6, color=TYPE_COLOR.get(kind, TYPE_COLOR["unknown"]), opacity=0.9),
            name=kind
        ))

    layout = go.Layout(
        title='Python Program Data Structures (3D)',
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig = go.Figure(data=[edge_trace] + node_traces, layout=layout)
    return fig

def visualize_file(py_path: str, out_html: str = None):
    G = build_graph(py_path)
    fig = graph_to_plotly_3d(G)
    if out_html:
        fig.write_html(out_html, auto_open=True)
        print(f"Saved 3D visualization to: {out_html}")
    else:
        fig.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_structures_3d.py <path_to_python_file> [output.html]")
        sys.exit(1)
    py_path = sys.argv[1]
    out_html = sys.argv[2] if len(sys.argv) >= 3 else None
    visualize_file(py_path, out_html)

if __name__ == "__main__":
    main()