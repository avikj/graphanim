""" viz_utils.py: utilities for visualizing graph data structures """
import numpy as np
import math
import svgwrite
from copy import deepcopy
import time 
import random

'''
Find (x, y) to minimize cost function J(x, y, E), where x is list of x coords, y is 
list of y coords, and E is list of edges in graph. Uses gradient descent algorithm
for optimization.
'''
def find_optimal_coords(n, adjacency, representation='list', vertex_spacing_factor=1, edge_spacing_factor=0, mutation_rate=0.4, verbose=False, max_iter=0):
  adjacency = deepcopy(adjacency)
  start = time.clock()
  if verbose:
    print 'adjacency:', adjacency
  if representation == 'list':
    mean_edge_cost = get_mean_edge_cost(adjacency)
    for i in range(n):
      for j in range(len(adjacency[i])):
        dest, cost = adjacency[i][j]
        adjacency[i][j] = (dest, cost/mean_edge_cost)
    x = np.random.randn(n)
    y = np.random.randn(n)
    learning_rate = .0003*n**2# *mean_edge_cost
    if verbose:
      print 'learning rate:', learning_rate
    improvement = 1
    iteration = 0
    prev_loss = float('inf')
    mutated = False
    mutation_count = 0
    while (improvement > 0.0001 or mutated) and (max_iter == 0 or iteration < max_iter):
      # mutated = False
      gradient_x, gradient_y = _coords_loss_gradient(x, y, n, adjacency)
      x -= learning_rate*gradient_x
      y -= learning_rate*gradient_y
      current_loss = _coords_loss(x, y, n, adjacency, vertex_spacing_factor=vertex_spacing_factor, edge_spacing_factor=edge_spacing_factor)
      improvement = prev_loss-current_loss
      prev_loss = current_loss
      iteration += 1
      # try random mutations to avoid local minima
      if np.random.rand() < mutation_rate:
        current_loss, mutated = mutate(x, y, n, adjacency, vertex_spacing_factor, edge_spacing_factor, current_loss)
        if verbose and mutated:
          print 'accepted mutation; cost:', current_loss
      if iteration < 15 or iteration % 5 == 0: 
        # save_svg('temp.svg', len(adjacency), x, y, adjacency)
        if verbose:
          print 'iter: %d, time_elapsed: %s, cost: %.8f'%(iteration, time.clock()-start,_coords_loss(x, y, n, adjacency, vertex_spacing_factor=vertex_spacing_factor, edge_spacing_factor=edge_spacing_factor))
    #for i in range(30):
     # current_loss, mutated = mutate(x, y, n, adjacency, vertex_spacing_factor, edge_spacing_factor, current_loss)
    #print count_intersections(x, y, n, adjacency), 'intersections'
    return x, y
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)

# TODO random location swap
def mutate(x, y, n, adjacency, vertex_spacing_factor, edge_spacing_factor, current_loss):
  current_intersections = count_intersections(x, y, n, adjacency)
  low_edge_vertices = [i for i in range(n) if len(adjacency[i])<=2]
  if len(low_edge_vertices) == 0:
    return current_loss, False
  mutation_index = np.random.randint(len(low_edge_vertices))
  if len(adjacency[low_edge_vertices[mutation_index]]) == 1: # random rotate around neighbor
    neighbor_i, _ = adjacency[low_edge_vertices[mutation_index]][0]
    radius = math.sqrt((x[low_edge_vertices[mutation_index]]-x[neighbor_i])**2+(y[low_edge_vertices[mutation_index]]-y[neighbor_i])**2)
    angle = np.random.rand()*2*math.pi
    x_shift = radius*math.cos(angle)+x[neighbor_i]-x[low_edge_vertices[mutation_index]]
    y_shift = radius*math.sin(angle)+y[neighbor_i]-y[low_edge_vertices[mutation_index]]
    # print 'Rotating node %d about neighbor'% low_edge_vertices[mutation_index]
  elif len(adjacency[low_edge_vertices[mutation_index]]) == 2:
    (neighbor_i, _), (neighbor_j, _) = adjacency[low_edge_vertices[mutation_index]]
    # print low_edge_vertices[mutation_index], neighbor_i, neighbor_j
    # print neighbor_i, neighbor_j
    a = (y[neighbor_i]-y[neighbor_j])/(x[neighbor_i]-x[neighbor_j])
    c = y[neighbor_i]-a*x[neighbor_i]
    d = (x[low_edge_vertices[mutation_index]] + (y[low_edge_vertices[mutation_index]] - c)*a)/(1 + a**2)
    x_shift = 2*d - 2*x[low_edge_vertices[mutation_index]]
    y_shift = 2*d*a + 2*c - 2*y[low_edge_vertices[mutation_index]]
  else:
    x_shift = 0
    y_shift = 0
    # print 'Flipping 2-n node across edge'
  x[low_edge_vertices[mutation_index]] += x_shift
  y[low_edge_vertices[mutation_index]] += y_shift
  mutated_loss = _coords_loss(x, y, n, adjacency, vertex_spacing_factor=vertex_spacing_factor, edge_spacing_factor=edge_spacing_factor)
  mutated_intersections = count_intersections(x, y, n, adjacency)
  if (mutated_loss < current_loss) and mutated_intersections == current_intersections or (mutated_intersections < current_intersections):
    # print 'Mutation accepted. Moved from (%f, %f) to (%f, %f)'%(x[low_edge_vertices[mutation_index]] - x_shift, y[low_edge_vertices[mutation_index]] - y_shift, x[low_edge_vertices[mutation_index]], y[low_edge_vertices[mutation_index]])
    current_loss = mutated_loss
    return current_loss, True
  else:
    # print 'intersections before:',current_intersections,', intersections after:', count_intersections(x, y, n, adjacency)
    # print 'Mutation rejected. Did not move from (%f, %f) to (%f, %f)'%(x[low_edge_vertices[mutation_index]] - x_shift, y[low_edge_vertices[mutation_index]] - y_shift, x[low_edge_vertices[mutation_index]], y[low_edge_vertices[mutation_index]])
    # save_svg('mutated%f.svg'%current_loss, n, x, y, adjacency)
    x[low_edge_vertices[mutation_index]] -= x_shift
    y[low_edge_vertices[mutation_index]] -= y_shift
    # save_svg('unmutated%f.svg'%current_loss, n, x, y, adjacency)
    return current_loss, False

def count_intersections(x, y, n, adjacency):
  def orientation(i, j, k):
    return  (y[k]-y[i])*(x[j]-x[i]) > (y[j]-y[i])*(x[k]-x[i])
  def intersect(i,j,k,l):
    return orientation(i, k, l) != orientation(j, k, l) and orientation(i, j, k) != orientation(i, j, l)
  edges = []
  for i in range(n):
    for j, c in adjacency[i]:
      if i < j:
        edges.append((i, j))
  count = 0
  for a in edges:
    for b in edges:
      if len(set([a[0], a[1], b[0], b[1]])) == 4 and intersect(a[0], a[1], b[0], b[1]):
        count += 1
  return count/2

def _coords_loss(x, y, n, adjacency, vertex_spacing_factor=1, edge_spacing_factor=0):
  result = 0
  mean_edge_cost = get_mean_edge_cost(adjacency)
  for i in range(n):
    for j, c in adjacency[i]:
      result += (math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)-c)**2
  # vertex-vertex spacing
  for i in range(n):
    for j in range(n):
      if i != j and j not in adjacency[i] and i not in adjacency[j]:
        result += mean_edge_cost*vertex_spacing_factor*math.log(1+math.exp(0.5-math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)/mean_edge_cost)) # penalize pairs of nodes closer than half mean edge length
  if edge_spacing_factor != 0:
    # vertex-edge and edge-edge spacing
    edge_midpoints_and_vertices = []
    for i in range(n):
      for j, c in adjacency[i]:
        edge_midpoints_and_vertices.append(((x[i]+x[j])/2,(y[i]+y[j])/2))
    for i in range(n):
      edge_midpoints_and_vertices.append((x[i], y[i]))
    for i in range(len(edge_midpoints_and_vertices)):
      for j in range(len(edge_midpoints_and_vertices)):
        if i != j:
          result += mean_edge_cost*edge_spacing_factor*math.log(1+math.exp(0.2-math.sqrt((edge_midpoints_and_vertices[i][0]-edge_midpoints_and_vertices[j][0])**2+(edge_midpoints_and_vertices[i][1]-edge_midpoints_and_vertices[j][1])**2)/mean_edge_cost))
  return result
def _coords_loss_gradient(x, y, n, adjacency):
  partial_x = np.zeros(n)
  partial_y = np.zeros(n)
  for i in range(n): # approximate derivative to experiment with different cost functions
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
    '''for j, c in adjacency[i]: # found derivative by hand and implemented
      actual_dist = math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
      partial_x[i] += 2*(1-c/actual_dist)*(x[i]-x[j])
      partial_y[i] += 2*(1-c/actual_dist)*(y[i]-y[j])'''
  return partial_x, partial_y
def get_mean_edge_cost(adjacency):
  return np.mean([cost for edges in adjacency for dest, cost in edges])
def save_svg(filename, n, x, y, adjacency, representation='list', labels=None):
  x = np.copy(x)
  y = np.copy(y)
  x -= min(x)
  y -= min(y)
  labels = labels or 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  scale = 600/max(np.max(x), np.max(y))
  shift = 20
  node_radius = 15
  x = x*scale+shift
  y = y*scale+shift
  if representation == 'list':
    dwg = svgwrite.Drawing(filename)
    for i in range(n):
      for j, c in adjacency[i]:
        dwg.add(dwg.line((x[i], y[i]), (x[j], y[j]), stroke='black'))
        dwg.add(dwg.text(str(c), insert=((x[i]+x[j])/2, (y[i]+y[j])/2), fill='black', style="font-family:Arial"))
    for i in range(n):
      node = dwg.add(dwg.g(id=str(i)))
      node.add(dwg.circle((x[i], y[i]), node_radius, fill='black', stroke='black'))
      label = node.add(dwg.text(labels[i], insert=(x[i], y[i]), fill='white', style="font-family:Arial"))
      label['text-anchor'] = 'middle'
      label['dominant-baseline'] = 'middle'
      dwg.add(node)
    dwg.save()
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)

def random_graph(n, e, representation='list', mean_edge_cost=10):
  # first, create a random spanning tree from a fully connected graph
  adj_list = [[] for i in range(n)]
  nodes = set(range(n))
  unconnected = set(range(n))
  connected = set()
  current = random.sample(nodes, 1)[0]
  unconnected.remove(current)
  connected.add(current)
  edge_count = 0
  while unconnected:
    neighbor = random.sample(nodes, 1)[0]
    if neighbor not in connected:
      edge_cost = np.random.binomial(mean_edge_cost*2-2, 0.5)+1
      adj_list[current].append((neighbor, edge_cost))
      adj_list[neighbor].append((current, edge_cost))
      unconnected.remove(neighbor)
      connected.add(neighbor)
      edge_count += 1
    current = neighbor
  while edge_count < e:
    a = np.random.randint(n)
    b = np.random.randint(n)
    if a != b and len([d for d, c in adj_list[a] if d == b]) == 0:
      edge_cost = np.random.binomial(mean_edge_cost*2-2, 0.5)+1
      adj_list[a].append((b, edge_cost))
      adj_list[b].append((a, edge_cost))
      edge_count += 1
  if representation == 'list':
    return adj_list
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)