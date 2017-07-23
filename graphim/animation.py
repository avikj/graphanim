from PIL import Image, ImageDraw, ImageFont
import numpy as np
from viz_utils import find_optimal_coords
import copy

class GraphAnimation:
  def __init__(self, n, adjacency, x, y, representation='list', labels=None, weighted=False, initial_color=(0, 0, 0)):
    if representation != 'list':
      raise ValueError('Graph representation \'%s\' is not supported.'%representation)
    if weighted:
      raise ValueError('Unweighted graphs are not supported.')
    self.n = n
    self.adjacency = adjacency
    self.x = x
    self.y = y
    self.past_states = []
    self.frame_count = 0
    self.current_node_colors = [initial_color]*n
    self.current_edge_colors = {}
    self.labels = labels
    for i in range(n):
      for j, cost in adjacency[i]:
        self.current_edge_colors[(i, j)] = initial_color
        self.current_edge_colors[(j, i)] = initial_color
  def set_node_color(self, i, color=1):
    """Highlights a given node."""
    self.current_node_colors[i] = color
  def set_edge_color(self, i, j, color=1):
    """Highlights the edge between two given nodes."""
    self.current_edge_colors[(i, j)] = color
    self.current_edge_colors[(j, i)] = color

  def next_frame(self):
    """Stores current state information. Later function calls will only affect state of later frames."""
    self.past_states.append((self.current_node_colors, self.current_edge_colors))
    self.current_node_colors = copy.deepcopy(self.current_node_colors)
    self.current_edge_colors = copy.deepcopy(self.current_edge_colors)
    self.frame_count += 1

  def save_gif(self, filename, size=(512, 512), fps=1, node_radius=15, edge_width=2, padding=40):
    x = node_radius+padding+(np.copy(self.x) - np.min(self.x))*(size[0]-2*node_radius-2*padding)/(np.max(self.x)-np.min(self.x))
    y = node_radius+padding+(np.copy(self.y) - np.min(self.y))*(size[1]-2*node_radius-2*padding)/(np.max(self.y)-np.min(self.y))
    frames = []
    font = ImageFont.truetype('Arial', size=30)
    for node_colors, edge_colors in self.past_states:
      im = Image.new("RGB", size, "white")
      ctx = ImageDraw.Draw(im)
      # draw edges
      for edge_src in range(self.n):
        for edge_dest, cost in self.adjacency[edge_src]:
          ctx.line([x[edge_src], y[edge_src], x[edge_dest], y[edge_dest]], width=edge_width, fill=edge_colors[(edge_src, edge_dest)])
          # edge label
          text_w, text_h = ctx.textsize(str(cost), font=font)
          ctx.text(((x[edge_src]+x[edge_dest])/2, (y[edge_src]+y[edge_dest])/2), str(cost), fill='black', font=font)
      # draw nodes
      for vertex in range(self.n):
        for i in range(2):
          radius = node_radius-i
          ctx.ellipse([x[vertex]-radius, y[vertex]-radius, x[vertex]+radius, y[vertex]+radius], fill='white', outline=node_colors[vertex])
          # node label
          text_w, text_h = ctx.textsize(self.labels[vertex], font=font)
          ctx.text((x[vertex]-text_w/2, y[vertex]-text_h/2), self.labels[vertex], fill='black', font=font)
      frames.append(im)
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=1000/fps)
if __name__ == '__main__':
  with open('graph.in') as graph_file:
    l = list(graph_file)
    print l
    v = int(l[0].strip())
    labels = []
    for i in range(1, 1+v):
      labels.append(l[i].strip())
    e = int(l[1+v].strip())
    adjList = [[] for i in range(v)]
    for i in range(e):
      a, b, c = [int(s)-1 for s in l[i+2+v].strip().split(' ')]
      c += 1
      if c == -1:
        c = np.random.randint(5, 15)
      adjList[a].append((b, c))
      adjList[b].append((a, c))
  print adjList
  # x, y = find_optimal_coords(len(adjList), adjList, vertex_spacing_factor=1)
  x = [ 367.84228811,503.14438045,174.29192134,512,355.12095785,0] 
  y = [ 496.81964145,0,429.34652448,457.45613558,175.76995579,512]
  highlighted = (255, 0, 0)
  anim = GraphAnimation(v, adjList, x, y, labels=labels)
  anim.next_frame()
  anim.set_node_color(0, highlighted)
  anim.next_frame()
  anim.set_edge_color(0, 1, highlighted)
  anim.next_frame()
  anim.set_edge_color(0, 3, highlighted)
  anim.next_frame()
  anim.save_gif('out.gif', size=(1024, 1024), node_radius=30)