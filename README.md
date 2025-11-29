# Python Program Data Structure Visualizer (3D)

An interactive 3D visualization tool that analyzes Python source code and displays its data structures, classes, functions, and variables as an interactive graph.

## Features

- **AST-based Analysis**: Parses Python files using the Abstract Syntax Tree (AST) to extract program structure
- **3D Interactive Graph**: Renders an interactive 3D graph using Plotly with zoom, rotate, and pan controls
- **Type Detection**: Automatically identifies and color-codes different data types (lists, dicts, sets, tuples, classes, functions, etc.)
- **Relationship Mapping**: Shows connections between modules, classes, functions, variables, and data structures
- **Edge Customization**: Interactive slider to change edge colors and opacity in the visualization
- **Hover Information**: Displays relationship details when hovering over edges

## Installation

Install required dependencies:

```powershell
pip install plotly networkx
```

## Usage

### Basic Usage

```powershell
python visualize_structures_3d.py <path_to_python_file> [output.html]
```

### Examples

**Generate and auto-open visualization:**
```powershell
python visualize_structures_3d.py "your_script.py" "output.html"
```

**Display in browser without saving:**
```powershell
python visualize_structures_3d.py "your_script.py"
```

**Real example:**
```powershell
python visualize_structures_3d.py "W:\University\Photogrammetric CV\Assignment2\assignment 2 submission\assignment2_fixed.py" "viz_assignment2.html"
```

## Color Coding

The visualization uses distinct colors for different element types:

| Element Type | Color |
|-------------|-------|
| **list** | Blue (0,150,255) |
| **dict** | Orange (255,140,0) |
| **set** | Green (50,200,50) |
| **tuple** | Magenta (200,0,200) |
| **class** | Red (255,0,0) |
| **function** | Dark Blue (0,0,255) |
| **int** | Gray (200,200,200) |
| **str** | Pink (255,105,180) |
| **float** | Light Blue (180,180,255) |
| **bool** | Gray (150,150,150) |
| **module** | Black (0,0,0) |
| **unknown** | Gray (120,120,120) |

## Customization

### Modify Edge Appearance

Edit line 165-169 in `visualize_structures_3d.py`:

```python
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='rgb(0,100,200)', width=3),  # Change color/width here
    opacity=1.0,  # Adjust transparency (0.0-1.0)
    ...
)
```

### Add Custom Types

Add to the `TYPE_COLOR` dictionary at the top of the file:

```python
TYPE_COLOR = {
    "list": "rgb(0,150,255)",
    "your_custom_type": "rgb(R,G,B)",  # Add custom types here
    ...
}
```

## What Gets Visualized

- **Modules**: The Python file itself
- **Classes**: Class definitions and their methods
- **Functions**: Function definitions and calls
- **Variables**: Variable assignments with type inference
- **Data Structures**: Lists, dictionaries, sets, tuples and their elements
- **Relationships**: 
  - `contains`: Module contains classes/functions/variables
  - `method`: Class contains methods
  - `calls`: Function calls
  - `arg`: Arguments passed to functions
  - `dict-item`: Dictionary key-value pairs
  - `seq-item`: List/set/tuple elements

## Interactive Controls

Once the HTML file opens in your browser:

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel or pinch
- **Pan**: Right-click and drag
- **Toggle Legend**: Click legend items to show/hide element types
- **Edge Styling**: Use the slider at the bottom to change edge colors and opacity
- **Hover**: Mouse over nodes and edges to see details

## Limitations

- Only detects literal data structures (defined directly in code)
- Type inference is basic and may not capture all complex types
- Does not analyze runtime behavior, only static code structure
- Works best with well-structured Python code

## Technical Details

- **Parser**: Python's built-in `ast` module
- **Graph Library**: NetworkX for graph construction
- **Layout Algorithm**: Spring layout (force-directed) with 3D extension
- **Visualization**: Plotly for interactive 3D rendering
- **Output Format**: Self-contained HTML file with embedded JavaScript

## License

Free to use for educational and personal projects.

## Author

Created as a visualization tool for analyzing Python program structure in 3D space.
