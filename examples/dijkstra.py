from __future__ import absolute_import
import numpy as np
from graphim.animation import GraphAnimation
from graphim.viz_utils import find_optimal_coords
import os
from heapq import heappop, heappush
with open(os.path.abspath('graph.in')) as graph_file:
  l = list(graph_file)
  n = int(l[0].strip())
  labels = []
  for i in range(1, 1+n):
    labels.append(l[i].strip())
  e = int(l[1+n].strip())
  adj_list = [[] for i in range(n)]
  for i in range(e):
    a, b, c = [int(s)-1 for s in l[i+2+n].strip().split(' ')]
    c += 1
    if c == -1:
      c = np.random.randint(5, 15)
    adj_list[a].append((b, c))
    adj_list[b].append((a, c))
print adj_list
x, y = find_optimal_coords(n, adj_list, verbose=True)

# Set up animation
anim = GraphAnimation(n, adj_list, x, y, labels=labels)
anim.next_frame()
# Run Dijkstra's shortest path algorithm from node 0
cost = [float('inf')]*n # cost[i] will hold min cost to reach node i from node a
in_q = [True]*n
parent = [-1]*n
q = [(0, 0)] # set cost of 0 to 0
cost[0] = 0
while q:
  cost_u, u = heappop(q)
  anim.set_node_color(u, color='red')
  anim.set_edge_color(u, parent[u], color='red')
  anim.next_frame()
  if not in_q[u]:
    continue
  in_q[u] = False
  for v, edge_cost in adj_list[u]:
    if in_q[v] and cost_u+edge_cost < cost[v]:
      cost[v] = cost_u+edge_cost
      anim.set_node_color(v, color='blue')
      anim.set_edge_color(u, v, color='blue')
      anim.set_edge_color(parent[v], v, color='black') # unhighlight previous shortest path to v
      parent[v] = u
      heappush(q, (cost[v], v))
  anim.next_frame()
  anim.set_node_color(u, color=(127, 0, 0))
anim.save_gif('dijkstra.gif', node_radius=20, size=(1000, 1000))
print 'Wrote animation to dijkstra.gif.'
