from __future__ import absolute_import
import numpy as np
from phanim.animation import GraphAnimation
from phanim.utils import find_optimal_coords, random_graph
import os
from heapq import heappop, heappush
n = 10
e = 16
adj_list = random_graph(n, e)
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print 'Optimizing graph layout.'
x, y = find_optimal_coords(n, adj_list, verbose=False, tolerance=3e-5, edge_spacing_factor=0.5, node_spacing_factor=1)

# Helper functions for disjoint set union
def find(i, parent):
  if parent[i] != i:
    parent[i] = find(parent[i], parent)
  return parent[i]

def merge(i, j, parent, rank):
  i_rep = find(i, parent)
  j_rep = find(j, parent)
  if i_rep == j_rep:
    return
  if rank[i_rep] < rank[j_rep]:
    parent[i_rep] = j_rep
  elif rank[i_rep] > rank[j_rep]:
    parent[j_rep] = i_rep
  else:
    parent[i_rep] = j_rep
    rank[j_rep] += 1

# Set up animation
highlighted_1 = (255, 0, 0)
highlighted_2 = (0, 0, 255)
unhighlighted = (127, 127, 127)
anim = GraphAnimation(n, adj_list, x, y, labels=labels, initial_color=unhighlighted)
anim.next_frame()

# Run Kruskal's algorithm
print 'Running Kruskal\'s minimum spanning tree algorithm.'
edges = []
mst_edges = []
for i in range(n):
  for j, cost in adj_list[i]:
    if (cost, j, i) not in edges:
      edges.append((cost, i, j))
edges.sort()

parent = [i for i in range(n)]
rank = [0]*n
for cost, i, j in edges:
  anim.set_edge_color(i, j, highlighted_2)
  anim.next_frame()
  if find(i, parent) == find(j, parent): # don't join nodes which are already connected
    anim.set_edge_color(i, j, unhighlighted)
  else:
    mst_edges.append((i, j))
    merge(i, j, parent, rank)
    anim.set_edge_color(i, j, highlighted_1)
  anim.next_frame()
anim.next_frame()
for cost, i, j in edges:
  if (i, j) not in mst_edges and (j, i) not in mst_edges:
    anim.set_edge_color(i, j, (200, 200, 200))
anim.next_frame()
anim.next_frame()
print 'Writing GIF.'
anim.save_gif('kruskal.gif', node_radius=20, size=(1000, 1000), fps=1.5)
print 'Wrote animation to kruskal.gif.'
