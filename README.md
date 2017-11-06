<img src="/examples/prim69499.gif" width="500" alt="Prim's algorithm running on a graph with 14 nodes." title="Prim's algorithm running on a graph with 14 nodes.">


# graphanim
Python graph algorithm animation library. Add a few lines to an existing algorithm to view it as a GIF.

## Installation

This library is not yet hosted on PyPI, so you will need to clone the repository into your project directory.

```bash
$ git clone https://github.com/avikj/graphanim.git
```

## Usage

Import required `graphanim` functions into your Python program.

```python
from graphanim.animation import GraphAnimation
from graphanim.utils import find_optimal_coords
```

Now, define your graph - currently graphanim only supports adjacency lists. Each element of the adjacency list should be a tuple containing the node which this edge is connected to and the weight of the edge.

After defining a graph, use a utility function to automatically find a good layout for the nodes in the visualization. You can also set some parameters for the layout-computing algorithm.

Then you can initialize the animation.

```python
graph = [
  [(1, 10), (2, 5)], 
  [(0, 10), (3, 7)],
  [(0, 5)],
  [(1, 7)]
]

node_locations = find_optimal_coords(4, graph, verbose=False, tolerance=3e-5)

anim = GraphAnimation(4, graph, locations[:,0], locations[:,1], labels='ABCD', initial_color=(0, 0, 0))
```

Once you have created an animation object with the graph, you can add lines within your algorithm implementation to reflect changes in the graph.

For example, to make node `0` red in the current frame, and make the edge between nodes `1` and `2` blue, you can run.

```python
anim.set_node_color(0, (255, 0, 0))
anim.set_edge_color(1, 2, (0, 0, 255))
```

After making visual changes to reflect a step in the algorithm, call `anim.next_frame()` save the visual state and move on to the next frame in the animation.

Once the algorithm is complete, you can render it as a GIF.

```python 
anim.save_gif('dijkstra.gif', node_radius=20, size=(512, 512), fps=1.5)
```

Full examples for Dijkstra's shortest path algorithm, Prim's minimum spanning tree algorithm, and Kruskal's minimum spanning tree algorithm can be found in the examples folder.
