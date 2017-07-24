from PIL import Image, ImageDraw, ImageFont
import numpy as np
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