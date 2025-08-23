# Graphius
Graphius is a python library for creating and visualizing graphs. It has algorithms for finding shortest paths, minimum spanning trees, and more. You can visualize the graphs and algorithms step by step. You can use graphius with Jupyter notebooks or as a standalone application.

For examples see the [examples](https://github.com/alperenunlu/graphius/blob/master/examples/Examples.ipynb) notebook.

## Installation
```bash
pip install graphius
```
>If you want to use the visualization features you need to install `graphviz`.
See [graphviz.org/download](https://graphviz.org/download/) for installation instructions.

## Usage
```python
from graphius import Graph

adjacency_list = {
    's': [('t', 10), ('y', 5)],
    't': [('x', 1), ('y', 2)],
    'x': [('z', 4)],
    'y': [('t', 3), ('x', 9), ('z', 2)],
    'z': [('s', 7), ('x', 6)]
}

g = Graph(adjacency_list, weighted=True, directed=True)
g.dijkstra('s', visualized=True, gif_path='dijkstra.gif')
```
`dijsktra.gif:`

![dijkstra](https://github.com/alperenunlu/graphius/blob/master/examples/dijkstra.gif?raw=true)


This project is licensed under the MIT License - see the [LICENSE](https://github.com/alperenunlu/graphius/blob/master/LICENSE) file for details