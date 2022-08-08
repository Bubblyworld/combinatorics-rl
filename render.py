import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms as nxa
import pickle
import os
from src.rewards import girth, moore_coefficient, moore_coefficient_for_degree

N = 10

def to_graph(species):
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            if species[count] == 1:
                G.add_edge(i,j)
            count += 1
    return G

# Ensure there is exactly one argument:
if len(os.sys.argv) != 2:
    print("Usage: render.py <best_species_txt_xxx.txt>")
    exit(1)

# Unpickle the species from the given file:
with open(os.sys.argv[1], 'rb') as f:
    best_species = pickle.load(f)

for i in range(5):
    G = to_graph(best_species[i])
    print(f"{i}: diam {nxa.diameter(G)}, girth {girth(G)}, moore {moore_coefficient(G)}")

    plt.subplot(1,5,i+1)
    nx.draw_kamada_kawai(G)
    plt.title("Species " + str(i))
plt.show()
