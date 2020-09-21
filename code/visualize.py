# Testing networkx

"""
####################

# Visualization functions

# Set of functions to plot the networks or their properties
    
# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from models import GatedMLP

input_size = 2
output_size = 2
model = GatedMLP(input_size, output_size, hidden_size = 128, n_layers = 3, type_mod='normal')

final_graph = nx.Graph()
# Location
final_graph.pos = {}
# Size
final_graph.size = {}
# Fill input nodes
prev_nodes = [int(i) for i in np.linspace(0,input_size-1,input_size)]
for p in prev_nodes:
    final_graph.pos[p] = [0, p - input_size/2]
    final_graph.size[p] = 1
cur_node = input_size
current_layer = 1
for m in model.modules():
    
    layer_name = 'l' + str(current_layer)
    print(m.__class__)
    #attrs = vars(m)    
    #print(', '.join("%s: %s" % item for item in attrs.items()))
    #if (m.parameters_.get('weight') is not None)
    if m.__class__ in [nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
        current_layer += 1
    elif m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
        current_layer += 1
    elif m.__class__ in [nn.Linear]:
        print(m._parameters['weight'].shape)
        new_nodes = []
        for n in range(m._parameters['weight'].shape[0]):
            new_nodes.append(cur_node)
            final_graph.pos[cur_node] = [current_layer, n - m._parameters['weight'].shape[0]/2]
            final_graph.size[cur_node] = torch.sqrt(sum(torch.pow(m._parameters['weight'][n], 2))) * 1e5
            for p in range(len(prev_nodes)):
                final_graph.add_edge(prev_nodes[p], cur_node, weight=m._parameters['weight'][n][p])
            cur_node += 1
        prev_nodes = new_nodes
        current_layer += 1
#%%
for n in final_graph:
    print(n)
#%%
g = final_graph
G = final_graph
plt.figure(1, figsize=(24, 24))
plt.clf()
colors = ['b', 'g', 'r']
c = 'b'
node_size = [int(final_graph.size[n] / 10.0) for n in G]
edge_size = [(d['weight'] * 20) for (_, _, d) in G.edges(data=True)]
edge_color = [torch.sign((d['weight'])) for (_, _, d) in G.edges(data=True)]
for c in range(len(edge_color)):
    if (edge_color[c] == -1):
        edge_color[c] = 'r'
    else:
        edge_color[c] = 'b'
nx.draw_networkx_edges(G, G.pos, edge_color=edge_color, width=edge_size, alpha=0.5, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_nodes(G, G.pos, node_size=node_size, node_color='r', alpha=0.5)
nx.draw_networkx_nodes(G, G.pos, node_size=5, node_color='k')
plt.show()

#%%
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=7)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=1, connectionstyle="angle3,angleA=90,angleB=0")
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=1, alpha=0.5, edge_color='b', style='dashed', connectionstyle="angle3,angleA=90,angleB=0")

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()
        

#%%
G = nx.Graph()


G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=0.3)


"""
###########
Example
###########
"""

G = nx.cycle_graph(24)
pos = nx.spring_layout(G, iterations=200)
nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Blues)
plt.show()


"""
###########
Example
###########
"""

G = nx.star_graph(20)
pos = nx.spring_layout(G)
colors = range(20)
nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,
        width=4, edge_cmap=plt.cm.Blues, with_labels=False, connectionstyle="angle3,angleA=90,angleB=0")
plt.show()


"""
###########
Example
###########
"""

G = nx.Graph()

G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=0.3)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=6, connectionstyle="angle3,angleA=90,angleB=0")
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=6, alpha=0.5, edge_color='b', style='dashed', connectionstyle="angle3,angleA=90,angleB=0")

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()


"""
###########
Example
###########
"""

G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
pos = nx.layout.spring_layout(G)

node_sizes = [3 + 10 * i for i in range(len(G))]
M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_colors,
                               edge_cmap=plt.cm.Blues, width=2, connectionstyle='arc3,rad=1.5')
# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
pc.set_array(edge_colors)
plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()


"""
###########
Example
###########
"""

def minard_graph():
    data1 = """\
24.0,54.9,340000,A,1
24.5,55.0,340000,A,1
25.5,54.5,340000,A,1
26.0,54.7,320000,A,1
27.0,54.8,300000,A,1
28.0,54.9,280000,A,1
28.5,55.0,240000,A,1
29.0,55.1,210000,A,1
30.0,55.2,180000,A,1
30.3,55.3,175000,A,1
32.0,54.8,145000,A,1
33.2,54.9,140000,A,1
34.4,55.5,127100,A,1
35.5,55.4,100000,A,1
36.0,55.5,100000,A,1
37.6,55.8,100000,A,1
37.7,55.7,100000,R,1
37.5,55.7,98000,R,1
37.0,55.0,97000,R,1
36.8,55.0,96000,R,1
35.4,55.3,87000,R,1
34.3,55.2,55000,R,1
33.3,54.8,37000,R,1
32.0,54.6,24000,R,1
30.4,54.4,20000,R,1
29.2,54.3,20000,R,1
28.5,54.2,20000,R,1
28.3,54.3,20000,R,1
27.5,54.5,20000,R,1
26.8,54.3,12000,R,1
26.4,54.4,14000,R,1
25.0,54.4,8000,R,1
24.4,54.4,4000,R,1
24.2,54.4,4000,R,1
24.1,54.4,4000,R,1"""
    data2 = """\
24.0,55.1,60000,A,2
24.5,55.2,60000,A,2
25.5,54.7,60000,A,2
26.6,55.7,40000,A,2
27.4,55.6,33000,A,2
28.7,55.5,33000,R,2
29.2,54.2,30000,R,2
28.5,54.1,30000,R,2
28.3,54.2,28000,R,2"""
    data3 = """\
24.0,55.2,22000,A,3
24.5,55.3,22000,A,3
24.6,55.8,6000,A,3
24.6,55.8,6000,R,3
24.2,54.4,6000,R,3
24.1,54.4,6000,R,3"""
    cities = """\
24.0,55.0,Kowno
25.3,54.7,Wilna
26.4,54.4,Smorgoni
26.8,54.3,Moiodexno
27.7,55.2,Gloubokoe
27.6,53.9,Minsk
28.5,54.3,Studienska
28.7,55.5,Polotzk
29.2,54.4,Bobr
30.2,55.3,Witebsk
30.4,54.5,Orscha
30.4,53.9,Mohilow
32.0,54.8,Smolensk
33.2,54.9,Dorogobouge
34.3,55.2,Wixma
34.4,55.5,Chjat
36.0,55.5,Mojaisk
37.6,55.8,Moscou
36.6,55.3,Tarantino
36.5,55.0,Malo-Jarosewii"""

    c = {}
    for line in cities.split('\n'):
        x, y, name = line.split(',')
        c[name] = (float(x), float(y))

    g = []

    for data in [data1, data2, data3]:
        G = nx.Graph()
        i = 0
        G.pos = {}  # location
        G.pop = {}  # size
        last = None
        for line in data.split('\n'):
            x, y, p, r, n = line.split(',')
            G.pos[i] = (float(x), float(y))
            G.pop[i] = int(p)
            if last is None:
                last = i
            else:
                G.add_edge(i, last, **{r: int(n)})
                last = i
            i = i + 1
        g.append(G)

    return g, c

(g, city) = minard_graph()

plt.figure(1, figsize=(11, 5))
plt.clf()
colors = ['b', 'g', 'r']
for G in g:
    c = colors.pop(0)
    node_size = [int(G.pop[n] / 300.0) for n in G]
    nx.draw_networkx_edges(G, G.pos, edge_color=c, width=4, alpha=0.5, connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_nodes(G, G.pos, node_size=node_size, node_color=c, alpha=0.5)
    nx.draw_networkx_nodes(G, G.pos, node_size=5, node_color='k')
for c in city:
    x, y = city[c]
    plt.text(x, y + 0.1, c)
plt.show()




"""
###########
###########

Graph_tool tests

###########
###########
"""
import graph_tool as gt
import numpy as np

"""
###########
Example
###########
"""
g = gt.lattice([10, 10])
pos = gt.planar_layout(g)
gt.graph_draw(g, pos=pos, output="lattice-planar.pdf")

"""
###########
Example
###########
"""
g = gt.price_network(300)
pos = gt.fruchterman_reingold_layout(g, n_iter=1000)
gt.graph_draw(g, pos=pos, output="graph-draw-fr.pdf")

"""
###########
Example
###########
"""
g = gt.price_network(300)
pos = gt.arf_layout(g, max_iter=0)
gt.graph_draw(g, pos=pos, output="graph-draw-arf.pdf")

"""
###########
Example
###########
"""
g = gt.price_network(3000)
pos = gt.sfdp_layout(g)
gt.graph_draw(g, pos=pos, output="graph-draw-sfdp.pdf")

"""
###########
Example
###########
"""
g = gt.price_network(1000)
pos = gt.radial_tree_layout(g, g.vertex(0))
gt.graph_draw(g, pos=pos, output="graph-draw-radial.pdf")


"""
###########
Example
###########
"""
g = gt.price_network(1500)
deg = g.degree_property_map("in")
deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
ebet = gt.betweenness(g)[1]
ebet.a /= ebet.a.max() / 10.
eorder = ebet.copy()
eorder.a *= -1
pos = gt.sfdp_layout(g)
control = g.new_edge_property("vector<double>")
for e in g.edges():
    d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
    control[e] = [0.3, d, 0.7, d]
gt.graph_draw(g, pos=pos, vertex_size=deg, vertex_fill_color=deg, vorder=deg,
              edge_color=ebet, eorder=eorder, edge_pen_width=ebet,
              edge_control_points=control, # some curvy edges
              output="graph-draw.pdf")

"""
###########
Example
###########
"""
g = gt.price_network(1500)
deg = g.degree_property_map("in")
deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
ebet = gt.betweenness(g)[1]
gt.graphviz_draw(g, vcolor=deg, vorder=deg, elen=10,
                 ecolor=ebet, eorder=ebet, output="graphviz-draw.pdf")


"""
###########
Example
###########
"""
g = gt.collection.data["netscience"]
g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
g.purge_vertices()
state = gt.minimize_nested_blockmodel_dl(g, deg_corr=True)
t = gt.get_hierarchy_tree(state)[0]
tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
cts = gt.get_hierarchy_control_points(g, t, tpos)
pos = g.own_property(tpos)
b = state.levels[0].b
shape = b.copy()
shape.a %= 14
gt.graph_draw(g, pos=pos, vertex_fill_color=b, vertex_shape=shape, edge_control_points=cts,
              edge_color=[0, 0, 0, 0.3], vertex_anchor=0, output="netscience_nested_mdl.pdf")

"""
###########
Example
###########
"""
g = gt.collection.data["celegansneural"]
state = gt.minimize_nested_blockmodel_dl(g, deg_corr=True)
gt.draw_hierarchy(state, output="celegansneural_nested_mdl.pdf")


