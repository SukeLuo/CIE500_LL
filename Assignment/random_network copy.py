# %%
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# In the following script we are going to generate a random graph

## First, we generate 12 random nodes in the horizontal space (0,3) and vertical space (0,3)
random.seed(46)
pos = {i:(random.random() * 3.0, random.random() * 3.0) for i in range(30)}

## Second, we create an edge list by a given probability.
edge_list = []
for node_pair in combinations(list(range(30)), 2):
    exist_prob = random.random()
    if exist_prob > 0.7:
        edge_list.append(node_pair)
    else:
        continue

## Now, we can create the graph based on the positions and edge list.

G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax, node_color='green', node_size = 5)
# plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "/Users/sukeluo/Documents/UB/Courses/25Spring/CIE500/CIE500_LL/week3/examplegraph.jpg",
    dpi=300,
    bbox_inches="tight",
)


## We can use networkx to get the graph's adjacency matrix
def bmatrix(
    a,
):  # reference source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


A = nx.adjacency_matrix(G, nodelist=list(G.nodes())).toarray()

# print(f"The latex version of adjacency matrix is \n {bmatrix(A)}")


# Now let's add a self-loop to the network

G.add_edge(7, 7)


fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax, arrowstyle="<|-", style="dashed")
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "//Users/sukeluo/Documents/UB/Courses/25Spring/CIE500/CIE500_LL/week3/examplegraph_selfloop.jpg",
    dpi=300,
    bbox_inches="tight",
)

# Finally, we can get the edgelist and adjancency matrix from Graph directly.

print(f"The adjancency matrix of G is \n {nx.adjacency_matrix(G).toarray()}")

print(f"The edge list of G is \n {nx.to_edgelist(G)}")

# Compute network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = sum(dict(G.degree()).values()) / num_nodes
diameter = nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else None
shortest_path_length = nx.shortest_path_length(G, source=3, target=28)
avg_shortest_path = nx.average_shortest_path_length(G.to_undirected()) if nx.is_connected(G.to_undirected()) else None
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Plot degree distribution
degrees = [deg for _, deg in G.degree()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=range(1, max(degrees) + 2), align="left", edgecolor="black", alpha=0.7)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Plot centrality distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(degree_centrality.values(), bins=20, edgecolor="black", alpha=0.7)
axes[0].set_title("Degree Centrality Distribution")
axes[0].set_xlabel("Degree Centrality")
axes[0].set_ylabel("Frequency")

axes[1].hist(betweenness_centrality.values(), bins=20, edgecolor="black", alpha=0.7)
axes[1].set_title("Betweenness Centrality Distribution")
axes[1].set_xlabel("Betweenness Centrality")
axes[1].set_ylabel("Frequency")

axes[2].hist(closeness_centrality.values(), bins=20, edgecolor="black", alpha=0.7)
axes[2].set_title("Closeness Centrality Distribution")
axes[2].set_xlabel("Closeness Centrality")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
# plt.show()
print(f"Number of Nodes: {num_nodes}")
print(f"Number of Edges: {num_edges}")
print(f"Average Degree: {avg_degree}")
print(f"Degree Centrality: {degree_centrality}")
print(f"Diameter: {diameter}")
print(f"Shortest Path Length: {shortest_path_length}")
print(f"Average Shortest Path Length: {avg_shortest_path}")
print(f"In-Degree Distribution: {in_degrees}")
print(f"Out-Degree Distribution: {out_degrees}")
print(f"Degree Centrality: {[degree_centrality[n] for n in G.nodes()]}")
print(f"Betweenness Centrality: {[betweenness_centrality[n] for n in G.nodes()]}")
print(f"Closeness Centrality: {[closeness_centrality[n] for n in G.nodes()]}")

'''''
Graph Characteristics:
- The  directed graph contains 30 nodes and 155 edges.
- The average degree of the nodes is 10.3.
- The averagr shortest path length is 1.66.

Node and Path Characteristics:
- The degree distribution suggests the presence of a mix of highly connected nodes and sparsely connected nodes.
- Some nodes act as important intermediaries, as seen in the betweenness centrality distribution.
- Closeness centrality highlights nodes that are relatively well-connected in terms of shortest paths.
'''''

# %%
