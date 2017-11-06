from PIL import Image, ImageDraw, ImageFont
import numpy as np
import copy
import json
import math
class GraphAnimation:
  def __init__(self, n, adjacency, x, y, edge_midpoints=None, labels='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789', initial_color=(0, 0, 0)):
    # TODO: check that the provided graph is unweighted
    self.n = n
    self.adjacency = adjacency
    self.x = x
    self.y = y
    self.edge_midpoints = edge_midpoints
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
    if i >= self.n or j >= self.n or i < 0 or j < 0 or j not in [b for b, c in self.adjacency[i]]:
      raise IndexError("Cannot set color of edge %s."%str((i, j)))
    self.current_edge_colors[(i, j)] = color
    self.current_edge_colors[(j, i)] = color

  def next_frame(self, count=1):
    """Stores current state information. Later function calls will only affect state of later frames."""
    for i in range(count):
      self.past_states.append((self.current_node_colors, self.current_edge_colors))
      self.current_node_colors = copy.deepcopy(self.current_node_colors)
      self.current_edge_colors = copy.deepcopy(self.current_edge_colors)
      self.frame_count += 1

  def save_gif(self, filename, size=(512, 512), fps=1, node_radius=15, edge_width=2, padding=40):
    minx=-1
    miny=-1
    maxx=-1
    maxy=-1
    print 'self.x', self.x, 'self.y', self.y
    for i in range(self.n):
      if minx == -1 or self.x[i] < minx:
        minx = self.x[i]
      if miny == -1 or self.y[i] < miny:
        miny = self.y[i]
      if maxx == -1 or self.x[i] > maxx:
        maxx = self.x[i]
      if maxy == -1 or self.y[i] > maxy:
        maxy = self.y[i]
    if self.edge_midpoints is not None:
      for i in range(self.n):
        for j, c in self.adjacency[i]:
          minx = min(self.edge_midpoints[i][j][0], minx)
          maxx = max(self.edge_midpoints[i][j][0], maxx)
          miny = min(self.edge_midpoints[i][j][1], miny)
          maxy = max(self.edge_midpoints[i][j][1], maxy)
    x = node_radius+padding+(np.copy(self.x) - minx)*(size[0]-2*node_radius-2*padding)/(maxx-minx)
    y = node_radius+padding+(np.copy(self.y) - miny)*(size[1]-2*node_radius-2*padding)/(maxy-miny)
    if self.edge_midpoints is not None:
      edge_midpoints = np.zeros((self.n, self.n, 2))
      for i in range(self.n):
        for j, c in self.adjacency[i]:
          edge_midpoints[i][j][0] = node_radius+padding+(np.copy(self.edge_midpoints[i][j][0]) - minx)*(size[0]-2*node_radius-2*padding)/(maxx-minx)
          edge_midpoints[i][j][1] = node_radius+padding+(np.copy(self.edge_midpoints[i][j][1]) - miny)*(size[1]-2*node_radius-2*padding)/(maxy-miny)
      print edge_midpoints
      print self.adjacency
    frames = []
    font = ImageFont.truetype('Arial', size=30)
    framesss = 0
    for node_colors, edge_colors in self.past_states:
      framesss += 1
      im = Image.new("RGB", size, "white")
      ctx = ImageDraw.Draw(im)
      # draw edges
      for edge_src in range(self.n):
        for edge_dest, cost in self.adjacency[edge_src]:
          if self.edge_midpoints is None:
            ctx.line((x[edge_src], y[edge_src], x[edge_dest], y[edge_dest]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))
          else:
            ox, oy, r, a1, a2 = _arc_through_points(x[edge_src], y[edge_src], edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1], x[edge_dest], y[edge_dest])
            if ox + oy + r != float('nan') and ox + oy + r != float('inf'):
              ctx.arc([ox-r, oy-r, ox+r, oy+r], a1, a2, fill=tuple(edge_colors[(edge_src, edge_dest)]))
              ctx.arc([ox-r-0.5, oy-r-0.5, ox+r+0.5, oy+r+0.5], a1, a2, fill=tuple(edge_colors[(edge_src, edge_dest)]))
              ctx.arc([ox-r+0.5, oy-r+0.5, ox+r+0.5, oy+r+0.5], a1, a2, fill=tuple(edge_colors[(edge_src, edge_dest)]))
              ctx.arc([ox-r-0.5, oy-r-0.5, ox+r-0.5, oy+r-0.5], a1, a2, fill=tuple(edge_colors[(edge_src, edge_dest)]))

            else:
              ctx.line((x[edge_src], y[edge_src], x[edge_dest], y[edge_dest]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))

            # ctx.line((x[edge_src], y[edge_src], edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))
            # ctx.line((edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1], x[edge_dest], y[edge_dest]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))
          # edge label
          text_w, text_h = ctx.textsize(str(cost), font=font)
          if self.edge_midpoints is None:
            ctx.text(((x[edge_src]+x[edge_dest])/2, (y[edge_src]+y[edge_dest])/2), str(cost), fill='black', font=font)
          else:
            ctx.text((edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1]), str(cost), fill='black', font=font)
      # draw nodes
      for node in range(self.n):
        for i in range(2):
          radius = node_radius-i
          ctx.ellipse([x[node]-radius, y[node]-radius, x[node]+radius, y[node]+radius], fill='white', outline=tuple(node_colors[node]))
          # node label
          text_w, text_h = ctx.textsize(self.labels[node], font=font)
          ctx.text((x[node]-text_w/2, y[node]-text_h/2), self.labels[node], fill='black', font=font)
      frames.append(im)
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=1000/fps)

  def save_png(self, filename, size=(512, 512), node_radius=20, edge_width=2, padding=40):
    minx=-1
    miny=-1
    maxx=-1
    maxy=-1
    for i in range(self.n):
      if minx == -1 or self.x[i] < minx:
        minx = self.x[i]
      if miny == -1 or self.y[i] < miny:
        miny = self.y[i]
      if maxx == -1 or self.x[i] > maxx:
        maxx = self.x[i]
      if maxy == -1 or self.y[i] > maxy:
        maxy = self.y[i]
    if self.edge_midpoints is not None:
      for i in range(self.n):
        for j, c in self.adjacency[i]:
          minx = min(self.edge_midpoints[i][j][0], minx)
          maxx = max(self.edge_midpoints[i][j][0], maxx)
          miny = min(self.edge_midpoints[i][j][1], miny)
          maxy = max(self.edge_midpoints[i][j][1], maxy)
    x = node_radius+padding+(np.copy(self.x) - minx)*(size[0]-2*node_radius-2*padding)/(maxx-minx)
    y = node_radius+padding+(np.copy(self.y) - miny)*(size[1]-2*node_radius-2*padding)/(maxy-miny)
    if self.edge_midpoints is not None:
      edge_midpoints = np.zeros((self.n, self.n, 2))
      for i in range(self.n):
        for j, c in self.adjacency[i]:
          edge_midpoints[i][j][0] = node_radius+padding+(np.copy(self.edge_midpoints[i][j][0]) - minx)*(size[0]-2*node_radius-2*padding)/(maxx-minx)
          edge_midpoints[i][j][1] = node_radius+padding+(np.copy(self.edge_midpoints[i][j][1]) - miny)*(size[1]-2*node_radius-2*padding)/(maxy-miny)
      print edge_midpoints
      print self.adjacency
    frames = []
    font = ImageFont.truetype('Arial', size=30)
    framesss = 0
    im = Image.new("RGB", size, "white")
    ctx = ImageDraw.Draw(im)
    # draw edges
    for edge_src in range(self.n):
      for edge_dest, cost in self.adjacency[edge_src]:
        if self.edge_midpoints is None:
          ctx.line((x[edge_src], y[edge_src], x[edge_dest], y[edge_dest]), width=edge_width, fill=(0,0,0))
        else:
          ox, oy, r, a1, a2 = _arc_through_points(x[edge_src], y[edge_src], edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1], x[edge_dest], y[edge_dest])
          if ox + oy + r != float('nan') and ox + oy + r != float('inf'):
            ctx.arc([ox-r, oy-r, ox+r, oy+r], a1, a2, fill=(0,0,0))
            ctx.arc([ox-r-0.5, oy-r-0.5, ox+r+0.5, oy+r+0.5], a1, a2, fill=(0,0,0))
            ctx.arc([ox-r+0.5, oy-r+0.5, ox+r+0.5, oy+r+0.5], a1, a2, fill=(0,0,0))
            ctx.arc([ox-r-0.5, oy-r-0.5, ox+r-0.5, oy+r-0.5], a1, a2, fill=(0,0,0))

          else:
            ctx.line((x[edge_src], y[edge_src], x[edge_dest], y[edge_dest]), width=edge_width, fill=(0,0,0))

          # ctx.line((x[edge_src], y[edge_src], edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))
          # ctx.line((edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1], x[edge_dest], y[edge_dest]), width=edge_width, fill=tuple(edge_colors[(edge_src, edge_dest)]))
        # edge label
        text_w, text_h = ctx.textsize(str(cost), font=font)
        if self.edge_midpoints is None:
          ctx.text(((x[edge_src]+x[edge_dest])/2, (y[edge_src]+y[edge_dest])/2), str(cost), fill='black', font=font)
        else:
          ctx.text((edge_midpoints[edge_src][edge_dest][0], edge_midpoints[edge_src][edge_dest][1]), str(cost), fill='black', font=font)
    # draw nodes
    for node in range(self.n):
      for i in range(2):
        radius = node_radius-i
        ctx.ellipse([x[node]-radius, y[node]-radius, x[node]+radius, y[node]+radius], fill='white', outline=(0,0,0))
        # node label
        text_w, text_h = ctx.textsize(self.labels[node], font=font)
        ctx.text((x[node]-text_w/2, y[node]-text_h/2), self.labels[node], fill='black', font=font)
    
    im.save(filename)
  def save_json(self, filename):
    with open(filename, 'w') as json_file:
      json_file.write(json.dumps({
        'n': self.n,
        'adjacency': self.adjacency,
        'x': list(self.x),
        'y': list(self.y),
        'past_states': [{'node_colors': node_colors, 'edge_colors': _tuple_dict_to_nested_dict(edge_colors)} for node_colors, edge_colors in self.past_states],
        'frame_count': self.frame_count,
        'current_node_colors': self.current_node_colors,
        'current_edge_colors': _tuple_dict_to_nested_dict(self.current_edge_colors),
        'labels': list(self.labels)
      }, indent=2))
  @classmethod
  def load_json(cls, filename):
    with open(filename, 'r') as json_file:
      json_dict = json.loads(json_file.read())
      anim = cls(json_dict['n'], json_dict['adjacency'], json_dict['x'], json_dict['y'], labels=json_dict['labels'])
      anim.past_states = [(state['node_colors'], _nested_dict_to_tuple_dict(state['edge_colors'])) for state in json_dict['past_states']]
      anim.current_edge_colors = _nested_dict_to_tuple_dict(json_dict['current_edge_colors'])
      anim.current_node_colors = json_dict['current_node_colors']
      anim.frame_count = json_dict['frame_count']
      return anim
def _tuple_dict_to_nested_dict(d):
  result = {} # convert to dict because tuples don't work as json keys
  for (i, j), color in d.iteritems():
    if i not in result:
      result[i] = {}
    if j not in result:
      result[j] = {}
    result[i][j] = color
    result[j][i] = color
  return result
def _nested_dict_to_tuple_dict(d):
  result = {}
  for i, subd in d.iteritems():
    for j, color in subd.iteritems():
      result[(int(i), int(j))] = color
      result[(int(j), int(i))] = color
  return result

def _arc_through_points(a, b, c, d, f, g):
  a = np.float32(a)
  b = np.float32(b)
  c = np.float32(c)
  d = np.float32(d)
  f = np.float32(f)
  g = np.float32(g)
  x = (-(a-f)*(a+f)/2/(g-b)+(b+g)/2+(a-c)*(a+c)/2/(d-b)-(b+d)/2)/((a-c)/(d-b)-(a-f)/(g-b))
  y = ((a-f)/(g-b))*(x-(a+f)/2)+(b+g)/2
  r = math.sqrt((a-x)**2+(b-y)**2)
  a1 = _angle([a-x, b-y])
  a_mid = _angle([c-x, d-y])
  a2 = _angle([f-x, g-y])
  cw_diff_1_mid = ((a_mid-a1)+720)%360
  cw_diff_mid_2 = ((a2-a_mid)+720)%360
  if cw_diff_mid_2+cw_diff_1_mid > 360:
    return x, y, r, a2, a1
  return x, y, r, a1, a2

def _angle(vec):
  theta = math.atan2(vec[1], vec[0])
  return (theta*180/math.pi+360)%360

if __name__ == '__main__':
  print _arc_through_points(0.1, 0.0, 0.0, 1.0, 1.0, 0.0)