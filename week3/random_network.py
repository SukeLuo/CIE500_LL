# %%
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# In the following script we are going to generate a random graph

## First, we generate 12 random nodes in the horizontal space (0,3) and vertical space (0,3)
random.seed(46)
pos = {i:(random.random() * 3.0, random.random() * 3.0) for i in range(12)}

## Second, we create an edge list by a given probability.
edge_list = []
for node_pair in combinations(list(range(12)), 2):
    exist_prob = random.random()
    if exist_prob > 0.5:
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

print(f"The latex version of adjacency matrix is \n {bmatrix(A)}")


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

# %%
