import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import dwave.optimization
import dwave.optimization.generators


model = dwave.optimization.Model()

c0 = model.constant(np.zeros(5))
print(c0)
i0 = model.integer(5, lower_bound=0, upper_bound=4)
print(c0[i0])

c0 = model.constant(np.zeros((5, 10)))
i0 = model.integer(5, lower_bound=0, upper_bound=4)
print(c0[i0, :])
print(c0[:, i0])

c0 = model.constant(np.zeros((5, 10, 6)))
i0 = model.integer(5, lower_bound=0, upper_bound=4)
i1 = model.integer(5, lower_bound=0, upper_bound=4)
print(c0[i0, :])
print(c0[:, i0])
print(c0[:, i0, :])
print(c0[:, :, i0])
print(c0[i0, :, i1])

processing_times = np.array([[10, 5, 7], [20, 10, 15]])
# processing_times = np.random.randint(5, 20, size=(10, 120))
model = dwave.optimization.generators.flow_shop_scheduling(processing_times)
for node in model.iter_symbols():
    print(node)

G = model.to_networkx()
print(G.nodes)

# plt.figure(figsize=(20, 20))

for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    palette = sns.color_palette("husl", len(nodes))
    for i, node in enumerate(nodes):
        color = palette[i] + (1,)
        print(node, color)
        G.nodes[node]["layer"] = layer
        G.nodes[node]["color"] = color
        for nbr in G[node]:
            G.edges[node, nbr, 0]["color"] = color

# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

nx.draw_networkx(
    G,
    pos=pos,
    nodelist=G.nodes,
    edgelist=G.edges,
    with_labels=False,
    node_size=1000,
    # node_color="white",
    node_color=[G.nodes[node].get("color", "white") for node in G.nodes],
    edgecolors="black",
    edge_color=[G.edges[edge].get("color", "black") for edge in G.edges],
)

options = {
    "font_size": 7,
    # "node_size": 3000,
    # "node_color": "white",
    # "edgecolors": "black",
    # "linewidths": 5,
    # "width": 5,
}
nx.draw_networkx_labels(G, pos=pos, labels={node: G.nodes[node].get(
    "label", node) for node in G.nodes}, **options)
# plt.axis("equal")
plt.show()
