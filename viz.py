""" viz.py: utilities for visualizing graph data structures """
import numpy as np
import math
import svgwrite
from copy import deepcopy

def main():
  adjList = [[(1, 10), (2, 10), (3, 10), (4, 10)], 
    [(0, 10), (2, 10), (3, 10), (4, 10)], 
    [(0, 10), (1, 10), (3, 10), (4, 10)], 
    [(0, 10), (1, 10), (2, 10), (4, 10)], 
    [(0, 10), (1, 10), (2, 10), (3, 10), (5, 6)], [(4, 6)]]
  '''adjList = [[] for i in range(7)]
  with open('graph.in') as graph_file:
    l = list(graph_file)
    print l
    e = int(l[0].strip())
    for i in range(e):
      a, b = [int(s)-1 for s in l[i+1].strip().split(' ')]
      adjList[a].append((b, 1))
      adjList[b].append((a, 1))'''
  print adjList
  x, y = coords(len(adjList), adjList)
  print 'x =', x, '\ny = ', y
  save_svg('graph1.svg', len(adjList), x, y, adjList)

'''
Find (x, y) to minimize cost function J(x, y, E), where x is list of x coords, y is 
list of y coords, and E is list of edges in graph. Uses gradient descent algorithm
for optimization.
Objective function J:
J(x, y, E)=\sum_{(i, j, c) \in E} \left ( \sqrt{(x_i-x_j)^2+(y_i-y_j)^2}-c\right )^2
+ sum for each node {sum for each pair of edges from that node {cosine similarity of edge vectors}}
'''
def coords(n, adjacency, representation='list'):
  adjacency = deepcopy(adjacency)
  if representation == 'list':
    mean_edge_cost = get_mean_edge_cost(adjacency)
    x = mean_edge_cost*np.random.randn(n)
    y = mean_edge_cost*np.random.randn(n)
    learning_rate = .01# *mean_edge_cost
    improvement = 1
    i = 0
    while improvement > 0.000001:
      gradient_x, gradient_y = _coords_loss_gradient(x, y, n, adjacency)
      prev = _coords_loss(x, y, n, adjacency)
      x -= learning_rate*gradient_x
      y -= learning_rate*gradient_y
      improvement = prev-_coords_loss(x, y, n, adjacency)
      i += 1
      if i < 15 or i % 50 == 0: 
        x -= np.min(x)
        y -= np.min(y)
        save_svg('temp%d.svg'%i, len(adjacency), x, y, adjacency)
      print 'iter: %d, cost: %.4f'%(i, _coords_loss(x, y, n, adjacency))
    return x, y
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)
def _coords_loss(x, y, n, adjacency, k=1):
  result = 0
  mean_edge_cost = get_mean_edge_cost(adjacency)
  for i in range(n):
    for j, c in adjacency[i]:
      result += (math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)-c)**2
  '''
  # add cosine of angle between edges in cost
  for i in range(n): # TODO: differentiate this and update gradient code
    for j_0, c_0 in adjacency[i]:
      for j_1, c_1 in adjacency[i]:
        result += k*cosine_similarity(x, y, i, j_0, i, j_1)*mean_edge_cost
  '''
  for i in range(n):
    for j in range(n):
      if i != j and j not in adjacency[i] and i not in adjacency[j]:
        result += k*math.log(1+math.exp(0.5*mean_edge_cost-math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2))) # 
  return result
def _coords_loss_gradient(x, y, n, adjacency):
  partial_x = np.zeros(n)
  partial_y = np.zeros(n)
  for i in range(n): # assume unweighted for now
    # find partial_x[i]
    delta = 0.01
    loss_0 = _coords_loss(x, y, n, adjacency)
    x[i] += delta
    loss_f = _coords_loss(x, y, n, adjacency)
    x[i] -= delta
    partial_x[i] = (loss_f-loss_0)/delta
    loss_0 = _coords_loss(x, y, n, adjacency)
    y[i] += delta
    loss_f = _coords_loss(x, y, n, adjacency)
    y[i] -= delta
    partial_y[i] = (loss_f-loss_0)/delta
    '''for j, c in adjacency[i]:
      actual_dist = math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
      partial_x[i] += 2*(1-c/actual_dist)*(x[i]-x[j])
      partial_y[i] += 2*(1-c/actual_dist)*(y[i]-y[j])'''
  return partial_x, partial_y
# i_0, j_0 are endpoints of first vector, i_1, j_1 are endpoints of second
def cosine_similarity(x, y, i_0, j_0, i_1, j_1):
  vec_0 = [x[i_0]-x[j_0], y[i_0]-y[j_0]]
  vec_1 = [x[i_1]-x[j_1], y[i_1]-y[j_1]]
  return np.dot(vec_0, vec_1)/(np.linalg.norm(vec_0)*np.linalg.norm(vec_1))
def get_mean_edge_cost(adjacency):
  return np.mean([cost for edges in adjacency for dest, cost in edges])
def save_svg(filename, n, x, y, adjacency, representation='list', labels=None):
  labels = labels or 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  scale = 200/get_mean_edge_cost(adjacency)
  shift = 20
  node_radius = 15
  x = x*scale+shift
  y = y*scale+shift
  if representation == 'list':
    dwg = svgwrite.Drawing(filename)
    for i in range(n):
      for j, c in adjacency[i]:
        dwg.add(dwg.line((x[i], y[i]), (x[j], y[j]), stroke='black'))
        dwg.add(dwg.text(str(c), insert=((x[i]+x[j])/2, (y[i]+y[j])/2), fill='black'))
    for i in range(n):
      node = dwg.add(dwg.g(id=str(i)))
      node.add(dwg.circle((x[i], y[i]), node_radius, fill='white', stroke='black'))
      label = node.add(dwg.text(labels[i], insert=(x[i], y[i]), fill='black'))
      label['text-anchor'] = 'middle'
      label['dominant-baseline'] = 'middle'
      dwg.add(node)
    dwg.save()
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)
if __name__ == '__main__':
  main()