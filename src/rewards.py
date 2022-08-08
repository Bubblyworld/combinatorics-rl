from collections import deque
from typing import Callable
import networkx.algorithms as nxa
import networkx as nx

# Arbitrary very large constant, for punishing bad behaviour:
INFINITY = 1e9


def diameter_to_size(G: nx.Graph) -> float:
    if not (nx.is_connected(G)):
        return -INFINITY
    return nxa.diameter(G) / G.size()


def moore_coefficient(G: nx.Graph) -> float:
    """
    Computes the "Moore coefficient" of the graph G, which is the ratio of its
    girth to its diameter. A "Moore graph" is any graph for which the Moore
    coefficient exceeds 2, and such graphs are a rarity indeed! There is one
    of degree 57 that is conjectured to exist, but nobody has found it yet,
    and nobody has proven that it doesn't exist.
    """
    if not (nx.is_connected(G)):
        return -INFINITY
    return girth(G) / nxa.diameter(G)


def moore_coefficient_for_degree(degree: float) -> Callable[[nx.Graph], float]:
    """
    Modification of the moore_coefficient function to push the graph towards
    being `degree`-regular. Weighted to be worth roughly 1/2 of what progress
    towards maximising the moore coefficient would be.
    """
    def _inner(G: nx.Graph) -> float:
        if not (nx.is_connected(G)):
            return -INFINITY

        degrees = list(map(lambda x: x[1], G.degree))
        min_dev = abs(min(degrees) - degree)
        max_dev = abs(max(degrees) - degree)
        moore = girth(G) / nxa.diameter(G)

        return moore - 0.5 * (min_dev + max_dev)
    return _inner

def girth(G: nx.Graph) -> int:
    """
    Computes the girth of G in O(n^3) time using breadth-first search. If the
    graph is cycle-free then this returns 0.
    """
    min_cycle_len = INFINITY
    for n in G.nodes:
        visited = {}
        queue = deque([(n, e, 0) for e in G.edges(n)])
        while len(queue) > 0:
            node, edge, depth = queue.popleft()
            if (node, edge) in visited:
                continue
            visited[(node, edge)] = True

            if edge[1] == n and depth > 1:
                if depth + 1 < min_cycle_len:
                    min_cycle_len = depth + 1

            queue.extend([(edge[1], e, depth + 1) for e in G.edges(edge[1]) if e[1] != edge[0]])

    if min_cycle_len == INFINITY:
        return 0
    return min_cycle_len

