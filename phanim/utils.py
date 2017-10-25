""" viz_utils.py: utilities for visualizing graph data structures """
import numpy as np
import math
import svgwrite
from copy import deepcopy
import time 
import random
from animation import GraphAnimation
'''
Find (x, y) to minimize cost function J(x, y, E), where x is list of x coords, y is 
list of y coords, and E is list of edges in graph. Uses gradient descent algorithm
for optimization.
'''
def find_optimal_coords(n, adjacency, representation='list', tolerance=1e-5, mutation_rate=0.4, verbose=False, max_iter=500, curved_edges=True, spring_mode='graph_theoretic_dist', node_spacing_factor=10):
  actual_adjacency = adjacency
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
    locations = np.random.rand(n, 2)
    learning_rate = .001
    min_learning_rate = 0.000001*n
    if verbose:
      print 'learning rate:', learning_rate
    improvement = 1
    iteration = 0
    prev_loss = float('inf')
    mutated = False
    mutation_count = 0
    min_dist_matrix = get_min_dist_matrix(n, adjacency)
    while (learning_rate >= min_learning_rate) and (max_iter == 0 or iteration < max_iter):
      if iteration % 5 == 0:
        anim = GraphAnimation(n, actual_adjacency, locations[:,0], locations[:,1])
        anim.save_png('temps/%s.png'%(str(iteration).zfill(10)), node_radius=17, size=(800, 800))
      # mutated = False
      gradient = _coords_loss_gradient(locations, n, adjacency, min_dist_matrix, spring_mode=spring_mode, node_spacing_factor=10)
      locations -= learning_rate*gradient 
      current_loss = _coords_loss(locations, n, adjacency, min_dist_matrix, spring_mode=spring_mode, node_spacing_factor=10)
      improvement = prev_loss-current_loss
      prev_loss = current_loss
      iteration += 1
      # try random mutations to avoid local minima
      if np.random.rand() < mutation_rate:
        current_loss, mutated = mutate(locations, n, adjacency, current_loss, min_dist_matrix, spring_mode=spring_mode, node_spacing_factor=10)
        if verbose and mutated:
          print 'accepted mutation; cost:', current_loss
      if iteration < 15 or iteration % 5 == 0: 
        save_svg('temp.svg', len(adjacency), locations[:,0], locations[:,1], adjacency)
        if verbose:
          print 'iter: %d, time_elapsed: %s, cost: %.8f'%(iteration, time.clock()-start,_coords_loss(locations, n, adjacency, min_dist_matrix, spring_mode=spring_mode, node_spacing_factor=10))
      if improvement < tolerance and not mutated:
        learning_rate /= 2
        if verbose:
          print 'Did not improve by more than %f; decreased learning rate to %f.' % (tolerance, learning_rate)
     #for i in range(30):
     # current_loss, mutated = mutate(x, y, n, adjacency, node_spacing_factor, edge_spacing_factor, current_loss)
    #print count_intersections(x, y, n, adjacency), 'intersections'
    if not curved_edges:
      return locations
    else:
      edge_midpoints = np.full((n, n, 2), np.inf)
      for i in range(n):
        for j, c in adjacency[i]:
          edge_midpoints[i][j] = (locations[i]+locations[j])/2
      edge_count = 0
      for i in range(n):
        for j, c in adjacency[i]:
          edge_count += 1
      # print edge_midpoints

      total_movement = 0
      learning_rate = 0.001
      improvement = None
      last_energy = float('inf')
      while improvement is None or improvement > tolerance/4:
        iterations += 1
        if iteration % 5 == 0:
        anim = GraphAnimation(n, actual_adjacency, locations[:,0], locations[:,1], edge_midpoints = edge_midpoints)
        anim.save_png('temps/%s.png'%(str(iteration).zfill(10)), node_radius=17, size=(800, 800))
        force = np.zeros((n, n, 2))
        energy = 0
        for i in range(n):
          for j, c in adjacency[i]: # for each edge midpoint
            k = (n)*2
            force[i][j] += k*(locations[i]-edge_midpoints[i][j])
            force[i][j] += k*(locations[j]-edge_midpoints[i][j])
            energy += 0.5*k*(np.linalg.norm(locations[i]-edge_midpoints[i][j]))**2
            energy +=  0.5*k*(np.linalg.norm(locations[j]-edge_midpoints[i][j]))**2
            for k in range(n):
              force[i][j] += (edge_midpoints[i][j]-locations[k])/np.linalg.norm(edge_midpoints[i][j]-locations[k])**2
              energy -= math.log(np.linalg.norm(edge_midpoints[i][j]-locations[k]))
            for i2 in range(n):
              for j2, c2 in adjacency[i2]:
                if (i, j) != (i2, j2) and (j, i) != (i2, j2):
                  force[i][j] += (edge_midpoints[i][j]-edge_midpoints[i2][j2])/np.linalg.norm(edge_midpoints[i][j]-edge_midpoints[i2][j2])**2
                  energy -= math.log(np.linalg.norm(edge_midpoints[i][j]-edge_midpoints[i2][j2]))
        
        for i in range(n): # update midpoint location
          for j, c in adjacency[i]: 
            edge_midpoints[i][j] += force[i][j]*learning_rate
            # project point onto bisector to prevent accumulated error
            segment = locations[j]-locations[i]
            actual_midpoint = (locations[i]+locations[j])/2
            bisector = np.array([-segment[1], segment[0]])
            edge_midpoints[i][j] = bisector*np.dot((edge_midpoints[i][j]-actual_midpoint), bisector)/np.linalg.norm(bisector)**2+actual_midpoint
            total_movement += np.linalg.norm(force[i][j]*learning_rate)
        print energy
        improvement = last_energy-energy
        last_energy = energy
        # print total_movement
      # print edge_midpoints
      return locations, edge_midpoints
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)


# TODO random location swap
def mutate(locations, n, adjacency, current_loss, min_dist_matrix, spring_mode='graph_theoretic_dist', node_spacing_factor=10):
  x = locations[:,0]
  y = locations[:,1]
  current_intersections = count_intersections(locations, n, adjacency)
  low_edge_nodes = [i for i in range(n) if len(adjacency[i])<=2]
  if len(low_edge_nodes) == 0:
    return current_loss, False
  mutation_index = np.random.randint(len(low_edge_nodes))
  if len(adjacency[low_edge_nodes[mutation_index]]) == 1: # random rotate around neighbor
    neighbor_i, _ = adjacency[low_edge_nodes[mutation_index]][0]
    radius = math.sqrt((x[low_edge_nodes[mutation_index]]-x[neighbor_i])**2+(y[low_edge_nodes[mutation_index]]-y[neighbor_i])**2)
    angle = np.random.rand()*2*math.pi
    x_shift = radius*math.cos(angle)+x[neighbor_i]-x[low_edge_nodes[mutation_index]]
    y_shift = radius*math.sin(angle)+y[neighbor_i]-y[low_edge_nodes[mutation_index]]
    # print 'Rotating node %d about neighbor'% low_edge_nodes[mutation_index]
  elif len(adjacency[low_edge_nodes[mutation_index]]) == 2:
    (neighbor_i, _), (neighbor_j, _) = adjacency[low_edge_nodes[mutation_index]]
    # print low_edge_nodes[mutation_index], neighbor_i, neighbor_j
    # print neighbor_i, neighbor_j
    a = (y[neighbor_i]-y[neighbor_j])/(x[neighbor_i]-x[neighbor_j])
    c = y[neighbor_i]-a*x[neighbor_i]
    d = (x[low_edge_nodes[mutation_index]] + (y[low_edge_nodes[mutation_index]] - c)*a)/(1 + a**2)
    x_shift = 2*d - 2*x[low_edge_nodes[mutation_index]]
    y_shift = 2*d*a + 2*c - 2*y[low_edge_nodes[mutation_index]]
  else:
    x_shift = 0
    y_shift = 0
    # print 'Flipping 2-n node across edge'
  x[low_edge_nodes[mutation_index]] += x_shift
  y[low_edge_nodes[mutation_index]] += y_shift
  mutated_loss = _coords_loss(locations, n, adjacency, min_dist_matrix, spring_mode=spring_mode, node_spacing_factor=node_spacing_factor)
  mutated_intersections = count_intersections(locations, n, adjacency)
  if (mutated_loss < current_loss) and mutated_intersections == current_intersections or (mutated_intersections < current_intersections):
    # print 'Mutation accepted. Moved from (%f, %f) to (%f, %f)'%(x[low_edge_nodes[mutation_index]] - x_shift, y[low_edge_nodes[mutation_index]] - y_shift, x[low_edge_nodes[mutation_index]], y[low_edge_nodes[mutation_index]])
    current_loss = mutated_loss
    return current_loss, True
  else:
    # print 'intersections before:',current_intersections,', intersections after:', count_intersections(x, y, n, adjacency)
    # print 'Mutation rejected. Did not move from (%f, %f) to (%f, %f)'%(x[low_edge_nodes[mutation_index]] - x_shift, y[low_edge_nodes[mutation_index]] - y_shift, x[low_edge_nodes[mutation_index]], y[low_edge_nodes[mutation_index]])
    # save_svg('mutated%f.svg'%current_loss, n, x, y, adjacency)
    x[low_edge_nodes[mutation_index]] -= x_shift
    y[low_edge_nodes[mutation_index]] -= y_shift
    # save_svg('unmutated%f.svg'%current_loss, n, x, y, adjacency)
    return current_loss, False

def count_intersections(locations, n, adjacency):
  def orientation(i, j, k):
    return  (locations[k][1]-locations[i][1])*(locations[j][0]-locations[i][0]) > (locations[j][1]-locations[i][1])*(locations[k][0]-locations[i][0])
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

def _coords_loss(locations, n, adjacency, dist, node_spacing_factor=10, spring_mode='graph_theoretic_dist'):
  if spring_mode == 'graph_theoretic_dist':
    result = 0
    for i in range(n):
      for j in range(n):
        if i != j:
          result += (np.linalg.norm(locations[i]-locations[j])**2-dist[i][j])**2
  elif spring_mode == 'edges_only':
    result = 0
    for i in range(n):
      for j, c in adjacency[i]:
        result += (np.linalg.norm(locations[i]-locations[j])**2-dist[i][j])**2
    for i in range(n):
      for j in range(n):
        if i != j:
          if np.linalg.norm(locations[i]-locations[j]) == 0:
            print np.linalg.norm(locations[i]-locations[j]), locations[i], locations[j]
          result += node_spacing_factor/np.linalg.norm(locations[i]-locations[j])
  return result
def _coords_loss_gradient(locations, n, adjacency, dist, spring_mode='graph_theoretic_dist', node_spacing_factor=10):

  gradient = np.zeros((n, 2))
  for i in range(n): # approximate derivative to experiment with different cost functions
    delta = 0.01
    loss_0 = _coords_loss(locations, n, adjacency, dist, spring_mode=spring_mode, node_spacing_factor=node_spacing_factor)
    locations[i][0] += delta
    loss_f = _coords_loss(locations, n, adjacency, dist, spring_mode=spring_mode, node_spacing_factor=node_spacing_factor)
    locations[i][0] -= delta
    gradient[i][0] = (loss_f-loss_0)/delta
    loss_0 = _coords_loss(locations, n, adjacency, dist, spring_mode=spring_mode, node_spacing_factor=node_spacing_factor)
    locations[i][1] += delta
    loss_f = _coords_loss(locations, n, adjacency, dist, spring_mode=spring_mode, node_spacing_factor=node_spacing_factor)
    locations[i][1] -= delta
    gradient[i][1] = (loss_f-loss_0)/delta
    '''for j in range(n): # found derivative by hand and implemented
      if i != j:
        drawn_dist = np.linalg.norm(locations[i]-locations[j])
        gradient[i][0] += 2*(1-dist[i][j]/drawn_dist)*(locations[i][0]-locations[j][0])
        gradient[i][1] += 2*(1-dist[i][j]/drawn_dist)*(locations[i][1]-locations[j][1])'''
  return gradient
def _cosine_similarity(x, y, i1, j1, i2, j2):
  vec1 = np.array([x[i1]-x[j1], y[i1]-y[j1]])
  vec2 = np.array([x[i2]-x[j2], y[i2]-y[j2]])
  # print np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
  return max(1, min(-1, np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))
def get_mean_edge_cost(adjacency):
  return np.mean([cost for edges in adjacency for dest, cost in edges])

# Floyd-Warshall algorithm 
def get_min_dist_matrix(n, adj_list):
  dist = np.full((n, n), np.inf)
  for i in range(n):
    dist[i][i] = 0
  for i in range(n):
    for j, cost in adj_list[i]:
      dist[i][j] = cost
  for k in range(n):
    for i in range(n):
      for j in range(n):
        if dist[i][j] == np.inf and dist[i][k] != np.inf and dist[k][j] != np.inf or dist[i][j] > dist[i][k] + dist[k][j]:
          dist[i][j] = dist[i][k]+dist[k][j]
  return dist

def save_svg(filename, n, x, y, adjacency, representation='list', labels=None):
  x = np.array(x)
  y = np.array(y)
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
  adj_list = random_spanning_tree(n, mean_edge_cost=mean_edge_cost)
  edge_count = n-1
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

def random_spanning_tree(n, representation='list', mean_edge_cost=10):
  """Returns a random spanning tree assuming a fully connected graph."""
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
  if representation == 'list':
    return adj_list
  else:
    raise ValueError('Graph representation \'%s\' is not supported.'%representation)
if __name__ == '__main__':
  adj = random_graph(10, 12)
  print adj
  print get_min_dist_matrix(10, adj)