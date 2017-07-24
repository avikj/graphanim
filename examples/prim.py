from __future__ import absolute_import
import numpy as np
from graphim.animation import GraphAnimation
from graphim.utils import find_optimal_coords, random_graph
import os
from heapq import heappop, heappush
n = 8
adj_list = random_graph(n, 13)
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print 'Optimizing graph layout.'
x, y = find_optimal_coords(n, adj_list, verbose=False)

# Set up animation
highlighted_1 = (255, 0, 0)
higlighted_2 = (0, 0, 255)
unhighlighted = (127, 127, 127)
anim = GraphAnimation(n, adj_list, x, y, labels=labels, initial_color=unhighlighted)
anim.next_frame()
# Run Prim's minimum spanning tree algorithm from node 0
print 'Running Prim\'s minimum spanning tree algorithm from node %s'%labels[0]
cost = [float('inf')]*n # cost[i] will hold min cost to reach node i from node a
in_q = [True]*n
parent = [-1]*n
q = [(0, 0)] # set cost of 0 to 0
cost[0] = 0
while q:
  cost_u, u = heappop(q)
  anim.set_node_color(u, color=highlighted_1)
  anim.set_edge_color(u, parent[u], color=highlighted_1)
  anim.next_frame()
  if not in_q[u]:
    continue
  in_q[u] = False
  for v, edge_cost in adj_list[u]:
    if in_q[v] and edge_cost < cost[v]:
      cost[v] = edge_cost
      anim.set_node_color(v, color=higlighted_2)
      anim.set_edge_color(u, v, color=higlighted_2)
      anim.set_edge_color(parent[v], v, color=unhighlighted) # unhighlight previous shortest path to v
      parent[v] = u
      heappush(q, (cost[v], v))
  anim.next_frame()
  anim.set_node_color(u, color=(127, 0, 0))
for i in range(n):
  for j, cost in adj_list[i]:
    if i != parent[j] and j != parent[i]:
      anim.set_edge_color(i, j, (200, 200, 200))
anim.next_frame()
anim.next_frame()
print 'Writing GIF.'
anim.save_gif('prim.gif', node_radius=20, size=(1000, 1000), fps=1.2)
print 'Wrote animation to prim.gif.'
