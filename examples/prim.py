from __future__ import absolute_import
import numpy as np
from graphanim.animation import GraphAnimation
from graphanim.utils import find_optimal_coords, random_graph, random_spanning_tree
import os
from heapq import heappop, heappush
n = 10
adj_list = random_graph(n, 13)
print adj_list
'''adj_list = [[] for i in range(n)]
for i in range(n):
  for j in range(2):
    adj_list[i].append(((i+j+1)%n, 1))
    adj_list[(i+j+1)%n].append((i, 1))'''
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
print 'Optimizing graph layout.'
locations, edge_mids = find_optimal_coords(n, adj_list, verbose=True, tolerance=1e-4, max_iter=1000, mutation_rate=0.4, curved_edges=True, spring_mode='edges_only', node_spacing_factor=0.5)

# Set up animation
highlighted_1 = (255, 0, 0)
highlighted_2 = (0, 0, 255)
unhighlighted = (127, 127, 127)
anim = GraphAnimation(n, adj_list, locations[:,0], locations[:,1], labels=labels, edge_midpoints=edge_mids, initial_color=unhighlighted)
anim.next_frame()
# Run Prim's minimum spanning tree algorithm from node 0
print 'Running Prim\'s minimum spanning tree algorithm from node %s.'%labels[0]
cost = [float('inf')]*n # cost[i] will hold min cost to reach node i from node a
in_q = [True]*n
parent = [-1]*n
q = [(0, 0)] # set cost of 0 to 0
cost[0] = 0
while q:
  cost_u, u = heappop(q)
  anim.set_node_color(u, color=highlighted_1)
  if parent[u] != -1:
    anim.set_edge_color(u, parent[u], color=highlighted_1)
  anim.next_frame()
  if not in_q[u]:
    continue
  in_q[u] = False
  for v, edge_cost in adj_list[u]:
    if in_q[v] and edge_cost < cost[v]:
      print 'asdf'
      cost[v] = edge_cost
      anim.set_node_color(v, color=highlighted_2)
      anim.set_edge_color(u, v, color=highlighted_2)
      if parent[v] != -1: # unhighlight previous shortest path to v
        anim.set_edge_color(parent[v], v, color=unhighlighted) 
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
rand = np.random.randint(100000)
anim.save_gif('prim%d.gif'%rand, node_radius=17, size=(1200, 1200), fps=1.2)
anim.save_json('prim%d.json'%rand)
print 'Wrote animation to prim%d.gif.'%rand
