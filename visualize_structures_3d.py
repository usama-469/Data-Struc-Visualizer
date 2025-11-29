import ast
import os
import sys
import glob
import networkx as nx
import plotly.graph_objs as go
from typing import Dict, Tuple, List
from pathlib import Path

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
    "import": "rgb(100,200,100)",
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

def find_python_files(path: str) -> List[str]:
    """Find all Python files in path (file or directory)"""
    path_obj = Path(path)
    if path_obj.is_file() and path_obj.suffix == '.py':
        return [str(path_obj)]
    elif path_obj.is_dir():
        return [str(p) for p in path_obj.rglob('*.py')]
    return []

def build_graph_from_file(py_path: str, G: nx.Graph) -> None:
    """Build graph from a single Python file"""
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=py_path)
    except Exception as e:
        print(f"Warning: Could not parse {py_path}: {e}")
        return

    module_name = os.path.basename(py_path)
    if not G.has_node(module_name):
        G.add_node(module_name, kind="module", label=module_name, file_path=py_path)

    # Track variables and their types
    var_types: Dict[str, str] = {}
    
    # Track imported modules and their members
    imported_names = set()

    # Track imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_name = alias.name
                imported_names.add(alias.asname if alias.asname else alias.name)
                if not G.has_node(import_name):
                    G.add_node(import_name, kind="import", label=import_name)
                G.add_edge(module_name, import_name, relation="imports")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_name = node.module
                # Track individual imported names
                for alias in node.names:
                    imported_names.add(alias.asname if alias.asname else alias.name)
                if not G.has_node(import_name):
                    G.add_node(import_name, kind="import", label=import_name)
                G.add_edge(module_name, import_name, relation="imports")

    # Add classes and functions
    source_lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_label = f"{module_name}::{node.name}"
            docstring = ast.get_docstring(node) or ""
            # Get base classes for inheritance
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
            
            # Extract code snippet
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 5
            code_snippet = '\n'.join(source_lines[start_line:min(end_line, start_line + 20)])
            
            G.add_node(class_label, kind="class", label=node.name, 
                      docstring=docstring[:100], bases=bases, module=module_name,
                      code_snippet=code_snippet, lineno=node.lineno)
            G.add_edge(module_name, class_label, relation="contains")
            
            # Add inheritance edges
            for base_name in bases:
                # Try to find base class in graph
                base_label = f"{module_name}::{base_name}"
                if G.has_node(base_label):
                    G.add_edge(base_label, class_label, relation="inherits")
                else:
                    # Check other modules
                    for node_id in G.nodes():
                        if G.nodes[node_id].get('label') == base_name and G.nodes[node_id].get('kind') == 'class':
                            G.add_edge(node_id, class_label, relation="inherits")
                            break
            
            # Link methods
            for body in node.body:
                if isinstance(body, ast.FunctionDef):
                    fn_label = f"{class_label}.{body.name}()"
                    fn_docstring = ast.get_docstring(body) or ""
                    # Get function signature
                    params = [arg.arg for arg in body.args.args]
                    
                    # Extract code snippet
                    start_line = body.lineno - 1
                    end_line = body.end_lineno if hasattr(body, 'end_lineno') else start_line + 5
                    code_snippet = '\n'.join(source_lines[start_line:min(end_line, start_line + 20)])
                    
                    G.add_node(fn_label, kind="function", label=f"{body.name}()",
                              docstring=fn_docstring[:100], params=params, module=module_name,
                              code_snippet=code_snippet, lineno=body.lineno)
                    G.add_edge(class_label, fn_label, relation="method")
    
    # Second pass: Add top-level functions only (not methods inside classes)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_label = f"{module_name}::{node.name}()"
            # Skip if already added (shouldn't happen, but check to be safe)
            if G.has_node(func_label):
                print(f"Skipping duplicate function: {func_label}")
                continue
            docstring = ast.get_docstring(node) or ""
            params = [arg.arg for arg in node.args.args]
            
            # Extract code snippet for user-defined functions
            start_line = node.lineno - 1
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 5
            code_snippet = '\n'.join(source_lines[start_line:min(end_line, start_line + 20)])
            
            G.add_node(func_label, kind="function", label=f"{node.name}()", 
                      docstring=docstring[:100], params=params, module=module_name,
                      code_snippet=code_snippet, lineno=node.lineno)
            G.add_edge(module_name, func_label, relation="contains")
    
    # Third pass: Process variables and function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Left-hand side targets can be multiple
            val_type = guess_type(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    var_types[tgt.id] = val_type
                    var_label = f"{module_name}::{tgt.id}"
                    G.add_node(var_label, kind=val_type, label=tgt.id, module=module_name)
                    G.add_edge(module_name, var_label, relation="var")
        elif isinstance(node, ast.AnnAssign):
            # Annotated assignments
            if isinstance(node.target, ast.Name):
                ann = ast.unparse(node.annotation) if hasattr(ast, "unparse") else "unknown"
                val_type = guess_type(node.value) if node.value else ann
                kind = val_type if val_type in TYPE_COLOR else (ann if ann in TYPE_COLOR else "unknown")
                var_types[node.target.id] = kind
                var_label = f"{module_name}::{node.target.id}"
                G.add_node(var_label, kind=kind, label=node.target.id, module=module_name)
                G.add_edge(module_name, var_label, relation="var")
        elif isinstance(node, ast.Call):
            # Link variables passed to calls
            func_name = None
            is_builtin = False
            call_args = []
            
            # Extract argument names/types
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    call_args.append(arg.id)
                elif isinstance(arg, ast.Constant):
                    call_args.append(f"{type(arg.value).__name__}")
                elif hasattr(ast, 'unparse'):
                    call_args.append(ast.unparse(arg)[:30])
            
            if isinstance(node.func, ast.Name):
                func_name = f"{node.func.id}()"
                # Check if it's a built-in or imported function
                is_builtin = node.func.id in imported_names or node.func.id in dir(__builtins__)
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{ast.unparse(node.func)}()" if hasattr(ast, "unparse") else f"{node.func.attr}()"
                is_builtin = True  # Assume attribute calls are external
            if func_name:
                # Only create nodes for built-in functions, not user-defined ones
                if G.has_node(func_name):
                    # Node already exists - just update usage info if it's a built-in
                    if call_args and G.nodes[func_name].get('kind') == 'builtin-function':
                        if 'params' not in G.nodes[func_name]:
                            G.nodes[func_name]['params'] = call_args
                        usage = f"{func_name.replace('()', '')}({', '.join(call_args)})"
                        G.nodes[func_name]['usage_example'] = usage
                elif is_builtin:
                    # Only add node if it's a built-in/imported function
                    usage = f"{func_name.replace('()', '')}({', '.join(call_args)})" if call_args else func_name
                    G.add_node(func_name, kind="builtin-function", label=func_name, 
                              params=call_args if call_args else [], 
                              usage_example=usage)
                    G.add_edge(module_name, func_name, relation="calls")
                # Skip creating nodes for user-defined function calls - they're already in the graph from the second pass
                for arg in node.args:
                    if isinstance(arg, ast.Name) and G.has_node(arg.id):
                        G.add_edge(arg.id, func_name, relation="arg")

    # Link keys/items within dicts, lists, sets if literals appear
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    parent = tgt.id
                    parent_label = f"{module_name}::{parent}"
                    for k, v in zip(node.value.keys, node.value.values):
                        k_label = ast.unparse(k) if hasattr(ast, "unparse") else "key"
                        v_type = guess_type(v)
                        child_name = f"{parent_label}.{k_label}"
                        G.add_node(child_name, kind=v_type, label=f"{parent}.{k_label}", module=module_name)
                        if G.has_node(parent_label):
                            G.add_edge(parent_label, child_name, relation="dict-item")
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Set, ast.Tuple)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    parent = tgt.id
                    parent_label = f"{module_name}::{parent}"
                    for idx, elt in enumerate(node.value.elts):
                        v_type = guess_type(elt)
                        child_name = f"{parent_label}[{idx}]"
                        G.add_node(child_name, kind=v_type, label=f"{parent}[{idx}]", module=module_name)
                        if G.has_node(parent_label):
                            G.add_edge(parent_label, child_name, relation="seq-item")

def build_graph(path: str) -> nx.Graph:
    """Build graph from file or directory of Python files"""
    G = nx.Graph()
    py_files = find_python_files(path)
    
    if not py_files:
        print(f"No Python files found in: {path}")
        return G
    
    print(f"Analyzing {len(py_files)} Python file(s)...")
    for py_file in py_files:
        build_graph_from_file(py_file, G)
    
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

    # default edge trace
    default_color = 'rgb(0,100,200)'
    default_opacity = 1.0

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=default_color, width=3),
        opacity=default_opacity,
        hoverinfo='text',
        hovertext=edge_hover,
        name='relations'
    )

    # Create individual sliders for R, G, B, and opacity on the left side (30% smaller)
    sliders = [
        {
            'active': 0,
            'yanchor': 'top',
            'y': 0.95,
            'xanchor': 'left',
            'x': 0.01,
            'len': 0.14,
            'currentvalue': {
                'prefix': 'Red: ',
                'visible': True,
                'xanchor': 'left',
                'font': {'size': 10}
            },
            'pad': {'b': 5, 't': 5},
            'font': {'size': 8},
            'steps': [
                {
                    'label': str(r),
                    'method': 'restyle',
                    'args': [{'line.color': f'rgb({r},100,200)'}, [0]]
                }
                for r in range(0, 256, 25)
            ]
        },
        {
            'active': 4,
            'yanchor': 'top',
            'y': 0.75,
            'xanchor': 'left',
            'x': 0.01,
            'len': 0.14,
            'currentvalue': {
                'prefix': 'Green: ',
                'visible': True,
                'xanchor': 'left',
                'font': {'size': 10}
            },
            'pad': {'b': 5, 't': 5},
            'font': {'size': 8},
            'steps': [
                {
                    'label': str(g),
                    'method': 'restyle',
                    'args': [{'line.color': f'rgb(0,{g},200)'}, [0]]
                }
                for g in range(0, 256, 25)
            ]
        },
        {
            'active': 8,
            'yanchor': 'top',
            'y': 0.55,
            'xanchor': 'left',
            'x': 0.01,
            'len': 0.14,
            'currentvalue': {
                'prefix': 'Blue: ',
                'visible': True,
                'xanchor': 'left',
                'font': {'size': 10}
            },
            'pad': {'b': 5, 't': 5},
            'font': {'size': 8},
            'steps': [
                {
                    'label': str(b),
                    'method': 'restyle',
                    'args': [{'line.color': f'rgb(0,100,{b})'}, [0]]
                }
                for b in range(0, 256, 25)
            ]
        },
        {
            'active': 10,
            'yanchor': 'top',
            'y': 0.35,
            'xanchor': 'left',
            'x': 0.01,
            'len': 0.14,
            'currentvalue': {
                'prefix': 'Opacity: ',
                'visible': True,
                'xanchor': 'left',
                'font': {'size': 10}
            },
            'pad': {'b': 5, 't': 5},
            'font': {'size': 8},
            'steps': [
                {
                    'label': f'{o:.1f}',
                    'method': 'restyle',
                    'args': [{'opacity': o}, [0]]
                }
                for o in [i/10 for i in range(0, 11)]
            ]
        }
    ]
    
    # Nodes grouped by kind for coloring
    kinds = {}
    for n, data in G.nodes(data=True):
        kind = data.get("kind", "unknown")
        kinds.setdefault(kind, []).append(n)

    node_traces = []
    for kind, nodes in kinds.items():
        xs, ys, zs, texts, hovertexts = [], [], [], [], []
        customdata_list = []
        
        for n in nodes:
            x, y, z = pos[n]
            node_data = G.nodes[n]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            texts.append(node_data.get("label", n))
            
            # Build hover text with metadata
            hover_parts = [f"<b>{node_data.get('label', n)}</b>"]
            hover_parts.append(f"Type: {kind}")
            if node_data.get('module'):
                hover_parts.append(f"Module: {node_data.get('module')}")
            if node_data.get('params'):
                hover_parts.append(f"Params: {', '.join(node_data.get('params', []))}")
            if node_data.get('docstring'):
                hover_parts.append(f"Doc: {node_data.get('docstring')}")
            if node_data.get('bases'):
                hover_parts.append(f"Inherits: {', '.join(node_data.get('bases', []))}")
            hovertexts.append("<br>".join(hover_parts))
            
            # Add custom data for info panel - as a list of values
            customdata_list.append([
                node_data.get('docstring', ''),
                node_data.get('params', []),
                node_data.get('module', ''),
                node_data.get('bases', []),
                node_data.get('label', n),
                node_data.get('code_snippet', ''),
                node_data.get('lineno', ''),
                node_data.get('usage_example', '')
            ])
        
        node_traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text',
            text=texts,
            textposition='top center',
            hovertext=hovertexts,
            hoverinfo='text',
            marker=dict(size=6, color=TYPE_COLOR.get(kind, TYPE_COLOR["unknown"]), opacity=0.9),
            customdata=customdata_list,
            name=kind
        ))
        
        # Debug: print first customdata entry for this trace
        if customdata_list:
            print(f"Trace '{kind}' - First customdata: {customdata_list[0]}")

    layout = go.Layout(
        title='Python Program Data Structures (3D)',
        showlegend=True,
        legend=dict(
            x=1.0,
            y=0.0,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    fig = go.Figure(data=[edge_trace] + node_traces, layout=layout)
    return fig

def visualize_file(py_path: str, out_html: str = None):
    G = build_graph(py_path)
    fig = graph_to_plotly_3d(G)
    
    # Add custom HTML/CSS/JS for collapsible slider panel and filtering
    custom_html = """
    <style>
        body {
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        body.dark-mode {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .js-plotly-plot .plotly {
            transition: background-color 0.3s ease;
        }
        #theme-toggle {
            position: fixed;
            bottom: 10px;
            left: 10px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(to bottom, #f7d358 50%, #4a5568 50%);
            border: 3px solid #ccc;
            cursor: pointer;
            z-index: 10002;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            transition: transform 0.3s ease, border-color 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        #theme-toggle:hover {
            transform: scale(1.1);
        }
        body.dark-mode #theme-toggle {
            border-color: #555;
            transform: rotate(180deg);
        }
        body.dark-mode #theme-toggle:hover {
            transform: rotate(180deg) scale(1.1);
        }
        #slider-panel {
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #ccc;
            border-left: none;
            border-radius: 0 10px 10px 0;
            padding: 15px 20px;
            transition: left 0.3s ease, background-color 0.3s ease, border-color 0.3s ease;
            z-index: 10000;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            width: 200px;
            max-height: 90vh;
            overflow-y: auto;
            overflow-x: visible;
        }
        body.dark-mode #slider-panel {
            background-color: rgba(30, 30, 30, 0.95);
            border-color: #555;
            color: #e0e0e0;
        }
        #slider-panel.collapsed {
            left: -240px;
        }
        #toggle-btn {
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 30px;
            height: 60px;
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #ccc;
            border-radius: 0 10px 10px 0;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            color: #666;
            transition: left 0.3s ease, background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
            user-select: none;
            z-index: 10001;
        }
        body.dark-mode #toggle-btn {
            background-color: rgba(30, 30, 30, 0.95);
            border-color: #555;
            color: #aaa;
        }
        #toggle-btn.panel-open {
            left: 240px;
        }
        #toggle-btn:hover {
            background-color: rgba(230, 230, 230, 0.95);
        }
        .control-section {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .control-section:last-child {
            border-bottom: none;
        }
        .section-title {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #555;
            transition: color 0.3s ease;
        }
        body.dark-mode .section-title {
            color: #bbb;
        }
        .slider-group {
            margin-bottom: 12px;
        }
        .slider-label {
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 3px;
            color: #333;
        }
        .slider-control {
            width: 100%;
        }
        .slider-value {
            font-size: 10px;
            color: #666;
            text-align: center;
            margin-top: 2px;
        }
        .filter-checkbox {
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            font-size: 11px;
        }
        .filter-checkbox input {
            margin-right: 6px;
        }
        .search-box {
            width: 100%;
            padding: 6px;
            font-size: 11px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-bottom: 8px;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        body.dark-mode .search-box {
            background-color: #2a2a2a;
            color: #e0e0e0;
            border-color: #555;
        }
        #info-panel {
            position: fixed;
            right: 10px;
            top: 70px;
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 9999;
            display: none;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        body.dark-mode #info-panel {
            background-color: rgba(30, 30, 30, 0.95);
            border-color: #555;
            color: #e0e0e0;
        }
        #info-panel h4 {
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #333;
            transition: color 0.3s ease;
        }
        body.dark-mode #info-panel h4 {
            color: #e0e0e0;
        }
        #info-panel p {
            margin: 4px 0;
            font-size: 11px;
            color: #666;
            transition: color 0.3s ease;
        }
        body.dark-mode #info-panel p {
            color: #aaa;
        }
        #info-panel pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            margin: 8px 0;
            font-size: 10px;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        body.dark-mode #info-panel pre {
            background-color: #252525;
            border-color: #444;
        }
        #file-hierarchy-panel {
            position: fixed;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            max-width: 350px;
            max-height: 70vh;
            overflow-y: auto;
            z-index: 9998;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            transition: background-color 0.3s ease, border-color 0.3s ease, transform 0.3s ease;
        }
        body.dark-mode #file-hierarchy-panel {
            background-color: rgba(30, 30, 30, 0.95);
            border-color: #555;
            color: #e0e0e0;
        }
        #file-hierarchy-panel h4 {
            margin: 0 0 10px 0;
            font-size: 13px;
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            transition: color 0.3s ease, border-color 0.3s ease;
        }
        body.dark-mode #file-hierarchy-panel h4 {
            color: #e0e0e0;
            border-color: #555;
        }
        .file-item {
            margin: 6px 0;
            padding: 4px 0;
        }
        .file-checkbox {
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 11px;
            padding: 4px 6px;
            border-radius: 4px;
            transition: background-color 0.2s ease;
        }
        .file-checkbox:hover {
            background-color: rgba(0, 100, 200, 0.1);
        }
        body.dark-mode .file-checkbox:hover {
            background-color: rgba(100, 150, 255, 0.15);
        }
        .file-checkbox input {
            margin-right: 8px;
            cursor: pointer;
        }
        .file-checkbox label {
            cursor: pointer;
            flex: 1;
            word-break: break-all;
        }
        .file-stats {
            font-size: 9px;
            color: #888;
            margin-left: 24px;
            font-style: italic;
        }
        body.dark-mode .file-stats {
            color: #999;
        }
        .hierarchy-controls {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }
        .hierarchy-btn {
            flex: 1;
            padding: 4px 8px;
            font-size: 10px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s ease, border-color 0.3s ease;
        }
        .hierarchy-btn:hover {
            background-color: #e0e0e0;
        }
        body.dark-mode .hierarchy-btn {
            background-color: #3a3a3a;
            border-color: #555;
            color: #e0e0e0;
        }
        body.dark-mode .hierarchy-btn:hover {
            background-color: #4a4a4a;
        }
        #focus-reset-btn {
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            padding: 8px 16px;
            background-color: rgba(255, 100, 100, 0.9);
            color: white;
            border: 2px solid #cc0000;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            z-index: 10003;
            display: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: background-color 0.2s ease, transform 0.2s ease;
        }
        #focus-reset-btn:hover {
            background-color: rgba(255, 50, 50, 0.95);
            transform: translateX(-50%) scale(1.05);
        }
        body.dark-mode #focus-reset-btn {
            background-color: rgba(255, 80, 80, 0.9);
            border-color: #ff4444;
        }
        .focus-mode-hint {
            position: fixed;
            top: 50px;
            left: 50%;
            transform: translateX(-50%);
            padding: 6px 12px;
            background-color: rgba(0, 100, 200, 0.9);
            color: white;
            border-radius: 4px;
            font-size: 11px;
            z-index: 10002;
            display: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        body.dark-mode .focus-mode-hint {
            background-color: rgba(50, 150, 255, 0.9);
        }
        #info-panel code {
            font-family: 'Courier New', monospace;
            font-size: 10px;
        }
        .close-info {
            float: right;
            cursor: pointer;
            font-size: 16px;
            color: #999;
        }
        .close-info:hover {
            color: #333;
        }
    </style>
    
    <div id="theme-toggle" onclick="toggleTheme()" title="Toggle Dark Mode">🌓</div>
    
    <button id="focus-reset-btn" onclick="resetFocusMode()">🔄 Reset Focus Mode</button>
    <div class="focus-mode-hint">Double-click any node to focus on its connections</div>
    
    <div id="toggle-btn" onclick="togglePanel()">◀</div>
    
    <div id="slider-panel">
        <h3 style="margin: 0 0 15px 0; font-size: 14px; color: #333; text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 8px;">Visualization Controls</h3>
        
        <div class="control-section">
            <div class="section-title">Filter by Type</div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-modules" checked onchange="filterNodes()">
                <label>Modules</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-classes" checked onchange="filterNodes()">
                <label>Classes</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-functions" checked onchange="filterNodes()">
                <label>User Functions</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-builtins" checked onchange="filterNodes()">
                <label>Built-in Functions</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-variables" checked onchange="filterNodes()">
                <label>Variables</label>
            </div>
            <div class="filter-checkbox">
                <input type="checkbox" id="show-imports" checked onchange="filterNodes()">
                <label>Imports</label>
            </div>
        </div>
        
        <div class="control-section">
            <div class="section-title">Search</div>
            <input type="text" class="search-box" id="search-box" placeholder="Search nodes..." list="node-suggestions" oninput="searchNodes()">
            <datalist id="node-suggestions"></datalist>
        </div>
        
        <div class="control-section">
            <div class="section-title">Edge Styling</div>
            <div class="slider-group">
                <div class="slider-label">Red</div>
                <input type="range" class="slider-control" id="red-slider" min="0" max="255" value="0" oninput="updateColor()">
                <div class="slider-value" id="red-value">0</div>
            </div>
            
            <div class="slider-group">
                <div class="slider-label">Green</div>
                <input type="range" class="slider-control" id="green-slider" min="0" max="255" value="100" oninput="updateColor()">
                <div class="slider-value" id="green-value">100</div>
            </div>
            
            <div class="slider-group">
                <div class="slider-label">Blue</div>
                <input type="range" class="slider-control" id="blue-slider" min="0" max="255" value="200" oninput="updateColor()">
                <div class="slider-value" id="blue-value">200</div>
            </div>
            
            <div class="slider-group">
                <div class="slider-label">Opacity</div>
                <input type="range" class="slider-control" id="opacity-slider" min="0" max="100" value="100" oninput="updateOpacity()">
                <div class="slider-value" id="opacity-value">1.0</div>
            </div>
        </div>
    </div>
    
    <div id="info-panel">
        <span class="close-info" onclick="closeInfo()">✕</span>
        <div id="info-content"></div>
    </div>
    
    <div id="file-hierarchy-panel">
        <h4>📁 File Hierarchy</h4>
        <div class="hierarchy-controls">
            <button class="hierarchy-btn" onclick="selectAllFiles()">Select All</button>
            <button class="hierarchy-btn" onclick="deselectAllFiles()">Deselect All</button>
        </div>
        <div id="file-list"></div>
    </div>
    
    <script>
        let graphData = null;
        let fileNodeMap = {}; // Maps file names to their associated node indices
        let allFiles = [];
        let focusModeActive = false;
        let focusedNodeData = null;
        
        // Get a safe copy of the layout for 3D plots
        function getSafeLayout() {
            return {
                title: 'Python Program Data Structures (3D)',
                showlegend: true,
                legend: {
                    x: 1.0,
                    y: 0.0,
                    xanchor: 'right',
                    yanchor: 'bottom',
                    bgcolor: document.body.classList.contains('dark-mode') ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                    bordercolor: document.body.classList.contains('dark-mode') ? 'rgba(85, 85, 85, 0.3)' : 'rgba(0, 0, 0, 0.3)',
                    borderwidth: 1,
                    font: {
                        color: document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#000000'
                    }
                },
                scene: {
                    xaxis: {showbackground: false, showticklabels: false, visible: false},
                    yaxis: {showbackground: false, showticklabels: false, visible: false},
                    zaxis: {showbackground: false, showticklabels: false, visible: false}
                },
                margin: {l: 0, r: 0, t: 40, b: 0},
                paper_bgcolor: document.body.classList.contains('dark-mode') ? '#1a1a1a' : 'white',
                plot_bgcolor: document.body.classList.contains('dark-mode') ? '#1a1a1a' : 'white',
                font: {
                    color: document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#000000'
                }
            };
        }
        
        // Build file hierarchy and node mapping
        function buildFileHierarchy() {
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !myDiv.data) return;
            
            fileNodeMap = {};
            const fileStats = {};
            
            // Scan all traces to find module nodes and their children
            myDiv.data.forEach((trace, idx) => {
                if (idx === 0) return; // Skip edge trace
                
                if (trace.customdata) {
                    trace.customdata.forEach((data, pointIdx) => {
                        if (data && data[2]) { // data[2] is the module name
                            const moduleName = data[2];
                            if (!fileNodeMap[moduleName]) {
                                fileNodeMap[moduleName] = [];
                            }
                            fileNodeMap[moduleName].push({ traceIdx: idx, pointIdx: pointIdx });
                            
                            // Track stats
                            if (!fileStats[moduleName]) {
                                fileStats[moduleName] = { nodes: 0, type: trace.name };
                            }
                            fileStats[moduleName].nodes++;
                        }
                    });
                }
                
                // Also check for module nodes themselves
                if (trace.name === 'module' && trace.text) {
                    trace.text.forEach((text, pointIdx) => {
                        if (!fileNodeMap[text]) {
                            fileNodeMap[text] = [];
                        }
                        fileNodeMap[text].push({ traceIdx: idx, pointIdx: pointIdx });
                        if (!fileStats[text]) {
                            fileStats[text] = { nodes: 1, type: 'module' };
                        }
                    });
                }
            });
            
            allFiles = Object.keys(fileNodeMap).sort();
            
            // Populate the file list UI
            const fileList = document.getElementById('file-list');
            if (!fileList) return;
            
            fileList.innerHTML = '';
            allFiles.forEach(fileName => {
                const stats = fileStats[fileName] || { nodes: 0 };
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                const checkbox = document.createElement('div');
                checkbox.className = 'file-checkbox';
                checkbox.innerHTML = `
                    <input type="checkbox" id="file-${fileName}" checked onchange="toggleFile('${fileName.replace(/'/g, "\\'")}')"/>
                    <label for="file-${fileName}">${fileName}</label>
                `;
                
                const statDiv = document.createElement('div');
                statDiv.className = 'file-stats';
                statDiv.textContent = `${stats.nodes} node(s)`;
                
                fileItem.appendChild(checkbox);
                fileItem.appendChild(statDiv);
                fileList.appendChild(fileItem);
            });
        }
        
        // Toggle file and regenerate graph with only selected files
        function toggleFile(fileName) {
            regenerateGraph();
        }
        
        // Regenerate the graph based on selected files
        function regenerateGraph() {
            if (!graphData) return;
            
            // Get list of enabled files
            const enabledFiles = allFiles.filter(fileName => {
                const checkbox = document.getElementById(`file-${fileName}`);
                return checkbox && checkbox.checked;
            });
            
            if (enabledFiles.length === 0) {
                // If no files selected, show empty graph
                const emptyData = [{
                    x: [], y: [], z: [],
                    mode: 'lines',
                    type: 'scatter3d',
                    line: {color: 'rgb(0,100,200)', width: 3},
                    name: 'relations'
                }];
                Plotly.newPlot('myDiv', emptyData, getSafeLayout());
                setTimeout(setupClickHandler, 100);
                return;
            }
            
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !graphData) return;
            
            // Filter the original data to include only nodes from enabled files
            const filteredData = [];
            
            graphData.forEach((trace, idx) => {
                if (idx === 0) {
                    // Edge trace - we'll rebuild this based on filtered nodes
                    filteredData.push({
                        x: [], y: [], z: [],
                        mode: 'lines',
                        line: trace.line,
                        opacity: trace.opacity,
                        hoverinfo: trace.hoverinfo,
                        hovertext: [],
                        name: trace.name,
                        type: 'scatter3d'
                    });
                } else {
                    // Node traces - filter points by module
                    const newTrace = {
                        x: [], y: [], z: [],
                        mode: trace.mode,
                        text: [],
                        textposition: trace.textposition,
                        hovertext: [],
                        hoverinfo: trace.hoverinfo,
                        marker: JSON.parse(JSON.stringify(trace.marker)),
                        customdata: [],
                        name: trace.name,
                        type: 'scatter3d'
                    };
                    
                    // Filter points that belong to enabled files
                    for (let i = 0; i < trace.x.length; i++) {
                        const customData = trace.customdata ? trace.customdata[i] : null;
                        const moduleName = customData ? customData[2] : null;
                        
                        // Include if: module is in enabled files, or it's a module node in enabled files
                        let shouldInclude = false;
                        
                        if (trace.name === 'module' && trace.text && trace.text[i]) {
                            shouldInclude = enabledFiles.includes(trace.text[i]);
                        } else if (moduleName) {
                            shouldInclude = enabledFiles.includes(moduleName);
                        } else if (trace.name === 'import' || trace.name === 'builtin-function') {
                            // Include imports and builtins if ANY file is selected
                            shouldInclude = true;
                        }
                        
                        if (shouldInclude) {
                            newTrace.x.push(trace.x[i]);
                            newTrace.y.push(trace.y[i]);
                            newTrace.z.push(trace.z[i]);
                            newTrace.text.push(trace.text ? trace.text[i] : '');
                            newTrace.hovertext.push(trace.hovertext ? trace.hovertext[i] : '');
                            if (trace.customdata) {
                                newTrace.customdata.push(trace.customdata[i]);
                            }
                        }
                    }
                    
                    if (newTrace.x.length > 0) {
                        filteredData.push(newTrace);
                    }
                }
            });
            
            // Rebuild edges based on filtered nodes
            // Build a comprehensive map of visible nodes with all possible identifiers
            const visibleNodes = new Map(); // Maps node identifier to {x, y, z, label}
            const nodeIdentifiers = new Set(); // Set of all possible node identifiers
            
            filteredData.forEach((trace, idx) => {
                if (idx > 0) {
                    for (let i = 0; i < trace.x.length; i++) {
                        const customData = trace.customdata ? trace.customdata[i] : null;
                        const displayText = trace.text ? trace.text[i] : '';
                        const module = customData ? customData[2] : '';
                        const fullLabel = customData ? customData[4] : displayText;
                        
                        const nodeInfo = {
                            x: trace.x[i],
                            y: trace.y[i],
                            z: trace.z[i],
                            label: displayText
                        };
                        
                        // Add all possible identifiers for this node
                        // 1. Display text (e.g., "myFunction()")
                        if (displayText) {
                            visibleNodes.set(displayText, nodeInfo);
                            nodeIdentifiers.add(displayText);
                        }
                        
                        // 2. Full label (e.g., "module.py::myFunction()")
                        if (fullLabel) {
                            visibleNodes.set(fullLabel, nodeInfo);
                            nodeIdentifiers.add(fullLabel);
                        }
                        
                        // 3. Module::text format
                        if (module && displayText && trace.name !== 'module') {
                            const withModule = `${module}::${displayText}`;
                            visibleNodes.set(withModule, nodeInfo);
                            nodeIdentifiers.add(withModule);
                        }
                        
                        // 4. For modules, use just the module name
                        if (trace.name === 'module' && module) {
                            visibleNodes.set(module, nodeInfo);
                            nodeIdentifiers.add(module);
                        }
                    }
                }
            });
            
            console.log('Visible node identifiers:', Array.from(nodeIdentifiers));
            
            // Filter original edges
            if (graphData[0] && graphData[0].hovertext) {
                const edgeTrace = filteredData[0];
                const originalEdgeTrace = graphData[0];
                
                for (let i = 0; i < originalEdgeTrace.hovertext.length; i += 3) {
                    if (i >= originalEdgeTrace.x.length) break;
                    
                    const hoverText = originalEdgeTrace.hovertext[i];
                    if (!hoverText) continue;
                    
                    // Parse edge: "nodeA → nodeB (relation)" or "nodeA → nodeB"
                    const match = hoverText.match(/^(.+?)\s*→\s*(.+?)(?:\s*\(|$)/);
                    if (match) {
                        let [, nodeA, nodeB] = match;
                        nodeA = nodeA.trim();
                        nodeB = nodeB.trim().replace(/\s*\(.+$/, ''); // Remove relation part if present
                        
                        // Check if both nodes exist in visible nodes (check all possible formats)
                        const nodeAExists = nodeIdentifiers.has(nodeA);
                        const nodeBExists = nodeIdentifiers.has(nodeB);
                        
                        if (nodeAExists && nodeBExists) {
                            edgeTrace.x.push(originalEdgeTrace.x[i]);
                            edgeTrace.y.push(originalEdgeTrace.y[i]);
                            edgeTrace.z.push(originalEdgeTrace.z[i]);
                            edgeTrace.x.push(originalEdgeTrace.x[i + 1]);
                            edgeTrace.y.push(originalEdgeTrace.y[i + 1]);
                            edgeTrace.z.push(originalEdgeTrace.z[i + 1]);
                            edgeTrace.x.push(null);
                            edgeTrace.y.push(null);
                            edgeTrace.z.push(null);
                            edgeTrace.hovertext.push(hoverText);
                            edgeTrace.hovertext.push(hoverText);
                            edgeTrace.hovertext.push(null);
                        }
                    }
                }
            }
            
            // Update the plot with filtered data
            if (myDiv && myDiv.layout) {
                Plotly.newPlot('myDiv', filteredData, getSafeLayout());
                setTimeout(setupClickHandler, 100);
            }
        }
        
        // Select all files
        function selectAllFiles() {
            allFiles.forEach(fileName => {
                const checkbox = document.getElementById(`file-${fileName}`);
                if (checkbox) {
                    checkbox.checked = true;
                }
            });
            regenerateGraph();
        }
        
        // Deselect all files
        function deselectAllFiles() {
            allFiles.forEach(fileName => {
                const checkbox = document.getElementById(`file-${fileName}`);
                if (checkbox) {
                    checkbox.checked = false;
                }
            });
            regenerateGraph();
        }
        
        // Find all nodes connected to a given node
        function findConnectedNodes(clickedNodeId, clickedNodeType) {
            if (!graphData || !graphData[0]) return new Set();
            
            const connectedNodes = new Set();
            connectedNodes.add(clickedNodeId); // Include the clicked node itself
            
            const edgeTrace = graphData[0];
            if (!edgeTrace.hovertext) return connectedNodes;
            
            // Parse all edges and find direct connections only
            for (let i = 0; i < edgeTrace.hovertext.length; i += 3) {
                const hoverText = edgeTrace.hovertext[i];
                if (!hoverText) continue;
                
                const match = hoverText.match(/^(.+?)\s*→\s*(.+?)(?:\s*\(|$)/);
                if (match) {
                    let [, nodeA, nodeB] = match;
                    nodeA = nodeA.trim();
                    nodeB = nodeB.trim().replace(/\s*\(.+$/, '');
                    
                    // Check if the clicked node is one of the endpoints
                    const isNodeA = (nodeA === clickedNodeId || nodeA.includes(clickedNodeId) || 
                                    (clickedNodeId.includes('::') && nodeA.includes(clickedNodeId.split('::').pop())));
                    const isNodeB = (nodeB === clickedNodeId || nodeB.includes(clickedNodeId) || 
                                    (clickedNodeId.includes('::') && nodeB.includes(clickedNodeId.split('::').pop())));
                    
                    // Only add direct connections, and skip module nodes unless clicked node is a module
                    if (isNodeA) {
                        // Don't add module nodes unless the clicked node is a module
                        if (clickedNodeType === 'module' || !nodeB.endsWith('.py')) {
                            connectedNodes.add(nodeB);
                        }
                    }
                    if (isNodeB) {
                        // Don't add module nodes unless the clicked node is a module
                        if (clickedNodeType === 'module' || !nodeA.endsWith('.py')) {
                            connectedNodes.add(nodeA);
                        }
                    }
                }
            }
            
            return connectedNodes;
        }
        
        // Activate focus mode for a specific node
        function activateFocusMode(clickedText, clickedCustomData, clickedNodeType) {
            if (!graphData) return;
            
            focusModeActive = true;
            focusedNodeData = { text: clickedText, customData: clickedCustomData, type: clickedNodeType };
            
            // Get the clicked node's full identifier
            const module = clickedCustomData ? clickedCustomData[2] : '';
            const fullLabel = clickedCustomData ? clickedCustomData[4] : clickedText;
            
            // Build possible identifiers for the clicked node
            const clickedNodeIds = [clickedText];
            if (fullLabel) clickedNodeIds.push(fullLabel);
            if (module && clickedText) clickedNodeIds.push(`${module}::${clickedText}`);
            
            // Find all connected nodes (using all possible identifiers)
            let connectedNodes = new Set();
            clickedNodeIds.forEach(id => {
                const nodes = findConnectedNodes(id, clickedNodeType);
                nodes.forEach(n => connectedNodes.add(n));
            });
            
            console.log('Focus mode - Node type:', clickedNodeType);
            console.log('Focus mode - Connected nodes:', Array.from(connectedNodes));
            
            // Instead of filtering, rebuild the graph with only connected nodes
            // Get current plot's data and extract only the connected nodes
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !myDiv.data) return;
            
            // Build new data arrays with only connected nodes
            const newData = [];
            const edgeData = {
                x: [], y: [], z: [],
                mode: 'lines',
                line: {color: 'rgb(0,100,200)', width: 3},
                opacity: 1.0,
                hoverinfo: 'text',
                hovertext: [],
                name: 'relations',
                type: 'scatter3d'
            };
            newData.push(edgeData);
            
            // Track which node identifiers we've seen
            const processedNodes = new Set();
            const nodeTraceMap = new Map(); // Maps trace names to new trace data
            
            // Process each trace and extract connected nodes
            myDiv.data.forEach((trace, idx) => {
                if (idx === 0) return; // Skip edge trace
                
                for (let i = 0; i < trace.x.length; i++) {
                    const text = trace.text ? trace.text[i] : '';
                    const customData = trace.customdata ? trace.customdata[i] : null;
                    const nodeModule = customData ? customData[2] : '';
                    const nodeFullLabel = customData ? customData[4] : text;
                    
                    // Check if this node is in the connected set
                    let isConnected = false;
                    if (connectedNodes.has(text)) isConnected = true;
                    if (nodeFullLabel && connectedNodes.has(nodeFullLabel)) isConnected = true;
                    if (nodeModule && text && connectedNodes.has(`${nodeModule}::${text}`)) isConnected = true;
                    
                    // Also check if any connected node matches this one
                    connectedNodes.forEach(connectedId => {
                        if (text && (connectedId.includes(text) || text.includes(connectedId.split('::').pop()))) {
                            isConnected = true;
                        }
                    });
                    
                    if (isConnected) {
                        // Create or get trace for this node type
                        if (!nodeTraceMap.has(trace.name)) {
                            nodeTraceMap.set(trace.name, {
                                x: [], y: [], z: [],
                                mode: trace.mode,
                                text: [],
                                textposition: trace.textposition,
                                hovertext: [],
                                hoverinfo: trace.hoverinfo,
                                marker: JSON.parse(JSON.stringify(trace.marker)),
                                customdata: [],
                                name: trace.name,
                                type: 'scatter3d'
                            });
                        }
                        
                        const newTrace = nodeTraceMap.get(trace.name);
                        newTrace.x.push(trace.x[i]);
                        newTrace.y.push(trace.y[i]);
                        newTrace.z.push(trace.z[i]);
                        newTrace.text.push(text);
                        newTrace.hovertext.push(trace.hovertext ? trace.hovertext[i] : '');
                        if (trace.customdata) {
                            newTrace.customdata.push(trace.customdata[i]);
                        }
                        
                        processedNodes.add(text);
                        if (nodeFullLabel) processedNodes.add(nodeFullLabel);
                        if (nodeModule && text) processedNodes.add(`${nodeModule}::${text}`);
                    }
                }
            });
            
            // Add all node traces to newData
            nodeTraceMap.forEach(trace => {
                if (trace.x.length > 0) {
                    newData.push(trace);
                }
            });
            
            // Add edges between connected nodes
            const originalEdgeTrace = graphData[0];
            if (originalEdgeTrace && originalEdgeTrace.hovertext) {
                for (let i = 0; i < originalEdgeTrace.hovertext.length; i += 3) {
                    if (i >= originalEdgeTrace.x.length) break;
                    
                    const hoverText = originalEdgeTrace.hovertext[i];
                    if (!hoverText) continue;
                    
                    const match = hoverText.match(/^(.+?)\s*→\s*(.+?)(?:\s*\(|$)/);
                    if (match) {
                        let [, nodeA, nodeB] = match;
                        nodeA = nodeA.trim();
                        nodeB = nodeB.trim().replace(/\s*\(.+$/, '');
                        
                        if (connectedNodes.has(nodeA) && connectedNodes.has(nodeB)) {
                            edgeData.x.push(originalEdgeTrace.x[i]);
                            edgeData.y.push(originalEdgeTrace.y[i]);
                            edgeData.z.push(originalEdgeTrace.z[i]);
                            edgeData.x.push(originalEdgeTrace.x[i + 1]);
                            edgeData.y.push(originalEdgeTrace.y[i + 1]);
                            edgeData.z.push(originalEdgeTrace.z[i + 1]);
                            edgeData.x.push(null);
                            edgeData.y.push(null);
                            edgeData.z.push(null);
                            edgeData.hovertext.push(hoverText);
                            edgeData.hovertext.push(hoverText);
                            edgeData.hovertext.push(null);
                        }
                    }
                }
            }
            
            // Update the plot with new data
            Plotly.newPlot('myDiv', newData, getSafeLayout());
            
            // Show reset button
            document.getElementById('focus-reset-btn').style.display = 'block';
        }
        
        // Reset focus mode and show all nodes
        function resetFocusMode() {
            focusModeActive = false;
            focusedNodeData = null;
            
            // Hide reset button first
            document.getElementById('focus-reset-btn').style.display = 'none';
            
            // Restore graph by reapplying current filters
            // Check which filters are active
            const showModules = document.getElementById('show-modules').checked;
            const showClasses = document.getElementById('show-classes').checked;
            const showFunctions = document.getElementById('show-functions').checked;
            const showBuiltins = document.getElementById('show-builtins').checked;
            const showVariables = document.getElementById('show-variables').checked;
            const showImports = document.getElementById('show-imports').checked;
            
            // Check if any type filters are disabled
            const hasTypeFilters = !showModules || !showClasses || !showFunctions || 
                                   !showBuiltins || !showVariables || !showImports;
            
            // Check if any files are disabled
            const hasFileFilters = allFiles.some(fileName => {
                const checkbox = document.getElementById(`file-${fileName}`);
                return checkbox && !checkbox.checked;
            });
            
            // Restore original graph
            const myDiv = document.getElementById('myDiv');
            if (myDiv && graphData) {
                Plotly.newPlot('myDiv', graphData, getSafeLayout());
                setTimeout(() => {
                    setupClickHandler();
                    // Reapply filters if any were active
                    if (hasFileFilters) {
                        regenerateGraph();
                    } else if (hasTypeFilters) {
                        filterNodes();
                    }
                }, 100);
            }
        }
        
        // Dark mode toggle function
        function toggleTheme() {
            const body = document.body;
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');
            
            // Save preference
            localStorage.setItem('darkMode', isDark);
            
            // Update Plotly layout
            const myDiv = document.getElementById('myDiv');
            if (myDiv) {
                const update = {
                    'paper_bgcolor': isDark ? '#1a1a1a' : 'white',
                    'plot_bgcolor': isDark ? '#1a1a1a' : 'white',
                    'font.color': isDark ? '#e0e0e0' : '#000000',
                    'legend.bgcolor': isDark ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                    'legend.bordercolor': isDark ? 'rgba(85, 85, 85, 0.3)' : 'rgba(0, 0, 0, 0.3)',
                    'legend.font.color': isDark ? '#e0e0e0' : '#000000'
                };
                Plotly.relayout('myDiv', update);
            }
        }
        
        // Load theme preference on page load
        function loadThemePreference() {
            const isDark = localStorage.getItem('darkMode') === 'true';
            if (isDark) {
                document.body.classList.add('dark-mode');
                // Apply dark mode to Plotly after a short delay to ensure plot is ready
                setTimeout(() => {
                    const myDiv = document.getElementById('myDiv');
                    if (myDiv) {
                        const update = {
                            'paper_bgcolor': '#1a1a1a',
                            'plot_bgcolor': '#1a1a1a',
                            'font.color': '#e0e0e0',
                            'legend.bgcolor': 'rgba(30, 30, 30, 0.9)',
                            'legend.bordercolor': 'rgba(85, 85, 85, 0.3)',
                            'legend.font.color': '#e0e0e0'
                        };
                        Plotly.relayout('myDiv', update);
                    }
                }, 100);
            }
        }
        
        // Store original graph data and populate search suggestions on load
        window.addEventListener('DOMContentLoaded', function() {
            const myDiv = document.getElementById('myDiv');
            if (myDiv && myDiv.data) {
                graphData = JSON.parse(JSON.stringify(myDiv.data));
                populateSearchSuggestions();
                // Build file hierarchy after data is loaded
                setTimeout(buildFileHierarchy, 200);
            }
            // Load saved theme preference
            loadThemePreference();
        });
        
        function populateSearchSuggestions() {
            const datalist = document.getElementById('node-suggestions');
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !myDiv.data) return;
            
            const suggestions = new Set();
            myDiv.data.forEach((trace, idx) => {
                if (idx === 0) return; // Skip edge trace
                if (trace.text) {
                    trace.text.forEach(text => {
                        if (text) suggestions.add(text);
                    });
                }
            });
            
            datalist.innerHTML = '';
            suggestions.forEach(suggestion => {
                const option = document.createElement('option');
                option.value = suggestion;
                datalist.appendChild(option);
            });
        }
        
        function togglePanel() {
            const panel = document.getElementById('slider-panel');
            const btn = document.getElementById('toggle-btn');
            panel.classList.toggle('collapsed');
            
            if (panel.classList.contains('collapsed')) {
                btn.innerHTML = '▶';
                btn.classList.remove('panel-open');
            } else {
                btn.innerHTML = '◀';
                btn.classList.add('panel-open');
            }
        }
        
        // Initialize button position on load
        window.addEventListener('DOMContentLoaded', function() {
            const btn = document.getElementById('toggle-btn');
            if (btn) {
                btn.classList.add('panel-open');
            }
        });
        
        function filterNodes() {
            if (!graphData) return;
            
            // Get filter states
            const showModules = document.getElementById('show-modules').checked;
            const showClasses = document.getElementById('show-classes').checked;
            const showFunctions = document.getElementById('show-functions').checked;
            const showBuiltins = document.getElementById('show-builtins').checked;
            const showVariables = document.getElementById('show-variables').checked;
            const showImports = document.getElementById('show-imports').checked;
            
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !graphData) return;
            
            // Determine which types should be visible
            function isTypeVisible(traceName) {
                const name = traceName.toLowerCase();
                if (name === 'module') return showModules;
                if (name === 'class') return showClasses;
                if (name === 'function') return showFunctions;
                if (name === 'builtin-function') return showBuiltins;
                if (name === 'import') return showImports;
                if (['list', 'dict', 'set', 'tuple', 'int', 'str', 'float', 'bool', 'unknown'].includes(name)) {
                    return showVariables;
                }
                return true;
            }
            
            // Filter the original data to include only nodes of visible types
            const filteredData = [];
            
            graphData.forEach((trace, idx) => {
                if (idx === 0) {
                    // Edge trace - we'll rebuild this based on filtered nodes
                    filteredData.push({
                        x: [], y: [], z: [],
                        mode: 'lines',
                        line: trace.line,
                        opacity: trace.opacity,
                        hoverinfo: trace.hoverinfo,
                        hovertext: [],
                        name: trace.name,
                        type: 'scatter3d'
                    });
                } else {
                    // Check if this trace type should be visible
                    if (!isTypeVisible(trace.name)) {
                        return; // Skip this trace entirely
                    }
                    
                    // Include all nodes from this trace
                    const newTrace = {
                        x: [...trace.x],
                        y: [...trace.y],
                        z: [...trace.z],
                        mode: trace.mode,
                        text: trace.text ? [...trace.text] : [],
                        textposition: trace.textposition,
                        hovertext: trace.hovertext ? [...trace.hovertext] : [],
                        hoverinfo: trace.hoverinfo,
                        marker: JSON.parse(JSON.stringify(trace.marker)),
                        customdata: trace.customdata ? JSON.parse(JSON.stringify(trace.customdata)) : [],
                        name: trace.name,
                        type: 'scatter3d'
                    };
                    
                    filteredData.push(newTrace);
                }
            });
            
            // Rebuild edges based on filtered nodes
            // Build a comprehensive map of visible nodes with all possible identifiers
            const nodeIdentifiers = new Set(); // Set of all possible node identifiers
            
            filteredData.forEach((trace, idx) => {
                if (idx > 0) {
                    for (let i = 0; i < trace.x.length; i++) {
                        const customData = trace.customdata ? trace.customdata[i] : null;
                        const displayText = trace.text ? trace.text[i] : '';
                        const module = customData ? customData[2] : '';
                        const fullLabel = customData ? customData[4] : displayText;
                        
                        // Add all possible identifiers for this node
                        if (displayText) nodeIdentifiers.add(displayText);
                        if (fullLabel) nodeIdentifiers.add(fullLabel);
                        if (module && displayText && trace.name !== 'module') {
                            nodeIdentifiers.add(`${module}::${displayText}`);
                        }
                        if (trace.name === 'module' && module) {
                            nodeIdentifiers.add(module);
                        }
                    }
                }
            });
            
            // Filter original edges
            if (graphData[0] && graphData[0].hovertext) {
                const edgeTrace = filteredData[0];
                const originalEdgeTrace = graphData[0];
                
                for (let i = 0; i < originalEdgeTrace.hovertext.length; i += 3) {
                    if (i >= originalEdgeTrace.x.length) break;
                    
                    const hoverText = originalEdgeTrace.hovertext[i];
                    if (!hoverText) continue;
                    
                    // Parse edge: "nodeA → nodeB (relation)" or "nodeA → nodeB"
                    const match = hoverText.match(/^(.+?)\s*→\s*(.+?)(?:\s*\(|$)/);
                    if (match) {
                        let [, nodeA, nodeB] = match;
                        nodeA = nodeA.trim();
                        nodeB = nodeB.trim().replace(/\s*\(.+$/, ''); // Remove relation part if present
                        
                        // Check if both nodes exist in visible nodes
                        const nodeAExists = nodeIdentifiers.has(nodeA);
                        const nodeBExists = nodeIdentifiers.has(nodeB);
                        
                        if (nodeAExists && nodeBExists) {
                            edgeTrace.x.push(originalEdgeTrace.x[i]);
                            edgeTrace.y.push(originalEdgeTrace.y[i]);
                            edgeTrace.z.push(originalEdgeTrace.z[i]);
                            edgeTrace.x.push(originalEdgeTrace.x[i + 1]);
                            edgeTrace.y.push(originalEdgeTrace.y[i + 1]);
                            edgeTrace.z.push(originalEdgeTrace.z[i + 1]);
                            edgeTrace.x.push(null);
                            edgeTrace.y.push(null);
                            edgeTrace.z.push(null);
                            edgeTrace.hovertext.push(hoverText);
                            edgeTrace.hovertext.push(hoverText);
                            edgeTrace.hovertext.push(null);
                        }
                    }
                }
            }
            
            // Update the plot with filtered data
            if (myDiv && myDiv.layout) {
                Plotly.newPlot('myDiv', filteredData, getSafeLayout());
                setTimeout(setupClickHandler, 100);
            }
        }
        
        function searchNodes() {
            const searchTerm = document.getElementById('search-box').value.toLowerCase();
            const myDiv = document.getElementById('myDiv');
            if (!myDiv || !myDiv.data) return;
            
            if (!searchTerm) {
                // Reset all markers to normal size
                myDiv.data.forEach((trace, idx) => {
                    if (idx > 0) { // Skip edge trace
                        const defaultSizes = new Array(trace.x.length).fill(6);
                        Plotly.restyle('myDiv', {'marker.size': [defaultSizes]}, [idx]);
                    }
                });
                return;
            }
            
            // Highlight matching nodes
            myDiv.data.forEach((trace, idx) => {
                if (idx === 0) return; // Skip edge trace
                
                const sizes = [];
                for (let i = 0; i < trace.text.length; i++) {
                    const text = trace.text[i];
                    if (text && text.toLowerCase().includes(searchTerm)) {
                        sizes.push(15);
                    } else {
                        sizes.push(6);
                    }
                }
                
                Plotly.restyle('myDiv', {'marker.size': [sizes]}, [idx]);
            });
        }
        
        function updateColor() {
            const r = document.getElementById('red-slider').value;
            const g = document.getElementById('green-slider').value;
            const b = document.getElementById('blue-slider').value;
            
            document.getElementById('red-value').textContent = r;
            document.getElementById('green-value').textContent = g;
            document.getElementById('blue-value').textContent = b;
            
            const color = `rgb(${r},${g},${b})`;
            Plotly.restyle('myDiv', {'line.color': color}, [0]);
        }
        
        function updateOpacity() {
            const opacity = document.getElementById('opacity-slider').value / 100;
            document.getElementById('opacity-value').textContent = opacity.toFixed(1);
            
            Plotly.restyle('myDiv', {opacity: opacity}, [0]);
        }
        
        function closeInfo() {
            document.getElementById('info-panel').style.display = 'none';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Add click handler for showing node information
        function setupClickHandler() {
            const myDiv = document.getElementById('myDiv');
            let lastClickTime = 0;
            let lastClickedPoint = null;
            
            if (myDiv && myDiv.on) {
                myDiv.on('plotly_click', function(data) {
                    const point = data.points[0];
                    const currentTime = new Date().getTime();
                    const timeDiff = currentTime - lastClickTime;
                    
                    console.log('Clicked point full object:', JSON.stringify(Object.keys(point)));
                    console.log('Point index:', point.pointIndex);
                    console.log('Point number:', point.pointNumber);
                    console.log('Curve number:', point.curveNumber);
                    console.log('Trace name:', point.data.name);
                    console.log('Trace has customdata?', !!point.data.customdata);
                    
                    if (point.data.name === 'relations') return; // Ignore edge clicks
                    
                    // Check for double-click (within 300ms)
                    if (timeDiff < 300 && lastClickedPoint && 
                        lastClickedPoint.text === point.text && 
                        lastClickedPoint.curveNumber === point.curveNumber) {
                        // Double-click detected - activate focus mode
                        const idx = point.pointNumber !== undefined ? point.pointNumber : point.pointIndex;
                        const customData = point.data.customdata ? point.data.customdata[idx] : null;
                        const nodeType = point.data.name; // Get the node type from trace name
                        activateFocusMode(point.text, customData, nodeType);
                        lastClickTime = 0; // Reset to prevent triple-click issues
                        lastClickedPoint = null;
                        return;
                    }
                    
                    lastClickTime = currentTime;
                    lastClickedPoint = point;
                    
                    const infoPanel = document.getElementById('info-panel');
                    const infoContent = document.getElementById('info-content');
                    
                    if (!infoPanel || !infoContent) {
                        console.error('Panel elements not found');
                        return;
                    }
                    
                    let html = `<h4>${point.text || 'Node'}</h4>`;
                    html += `<p><strong>Type:</strong> ${point.data.name}</p>`;
                    
                    // Try to get additional info from trace
                    if (point.data.customdata) {
                        // Use pointNumber instead of pointIndex
                        const idx = point.pointNumber !== undefined ? point.pointNumber : point.pointIndex;
                        const customData = point.data.customdata[idx];
                        console.log('Using index:', idx);
                        console.log('Custom data:', customData);
                        if (customData) {
                            // customdata is an array: [docstring, params, module, bases, label, code_snippet, lineno, usage_example]
                            const docstring = customData[0];
                            const params = customData[1];
                            const module = customData[2];
                            const bases = customData[3];
                            const label = customData[4];
                            const code_snippet = customData[5];
                            const lineno = customData[6];
                            const usage_example = customData[7];
                            
                            // Basic info
                            if (module) {
                                html += `<p><strong>Module:</strong> ${module}</p>`;
                            }
                            if (lineno) {
                                html += `<p><strong>Line:</strong> ${lineno}</p>`;
                            }
                            if (bases && bases.length > 0) {
                                html += `<p><strong>Inherits:</strong> ${bases.join(', ')}</p>`;
                            }
                            if (docstring) {
                                html += `<p><strong>Doc:</strong> ${docstring}</p>`;
                            }
                            
                            // Parameters section
                            if (params && params.length > 0) {
                                html += `<p><strong>Parameters:</strong> ${params.join(', ')}</p>`;
                            }
                            
                            // Code or usage example
                            if (point.data.name === 'builtin-function') {
                                if (usage_example) {
                                    html += `<p><strong>Usage Example:</strong></p>`;
                                    html += `<pre><code>${escapeHtml(usage_example)}</code></pre>`;
                                }
                            } else if (code_snippet) {
                                html += `<p><strong>Code:</strong></p>`;
                                html += `<pre><code>${escapeHtml(code_snippet)}</code></pre>`;
                            } else {
                                console.log('No code snippet found');
                            }
                        } else {
                            console.log('customData is null/undefined');
                        }
                    } else {
                        console.log('No customdata in trace');
                    }
                    
                    infoContent.innerHTML = html;
                    infoPanel.style.display = 'block';
                });
                console.log('Click handler attached successfully');
            } else {
                console.error('Plotly div not ready, retrying...');
                setTimeout(setupClickHandler, 100);
            }
        }
        
        // Wait for both DOM and Plotly to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(setupClickHandler, 500);
            });
        } else {
            setTimeout(setupClickHandler, 500);
        }
    </script>
    """
    
    if out_html:
        fig.write_html(out_html, auto_open=True, include_plotlyjs='cdn', div_id='myDiv')
        # Inject custom HTML
        with open(out_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        html_content = html_content.replace('</body>', custom_html + '</body>')
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved 3D visualization to: {out_html}")
    else:
        fig.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_structures_3d.py <path_to_file_or_directory> [output.html]")
        print("\nExamples:")
        print("  python visualize_structures_3d.py script.py")
        print("  python visualize_structures_3d.py src/ output.html")
        print("  python visualize_structures_3d.py . visualization.html")
        sys.exit(1)
    
    py_path = sys.argv[1]
    out_html = sys.argv[2] if len(sys.argv) >= 3 else None
    
    # Check if path exists
    if not os.path.exists(py_path):
        print(f"Error: Path '{py_path}' does not exist")
        sys.exit(1)
    
    visualize_file(py_path, out_html)

if __name__ == "__main__":
    main()