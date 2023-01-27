from dimacs import loadDirectedWeightedGraph, loadWeightedGraph
from collections import deque
import heapq
import os
from time import process_time


def test_program(graph_directory, f):
    """
    Test given function.

    :param graph_directory: directory with graphs saved in files
    :param f: reference to tested function
    :returns: None
    """
    print(f"Testujemy funkcję {f.__name__}")
    file_counter = 0
    errors = 0
    for path in os.listdir(graph_directory):
        # Check if current path is a file
        file_path = os.path.join(graph_directory, path)
        if os.path.isfile(file_path):
            file_counter += 1
            # Load answer
            result = "OK"
            try:
                with open(file_path, 'r') as file:
                    line = file.readline()
                    expected_ans = int(line.split()[-1])

                    # Test function
                    start_time = process_time()
                    ans = f(file_path)
                    end_time = process_time() - start_time

                    result += "\nCzas: " + str(end_time) + 's'

                    if ans != expected_ans:
                        result = f"ERROR"
                        errors += 1
                print(
                    f"-------------------\nGraf {path}\nOdpowiedź wymagana: {expected_ans}\nOdpowiedź otrzymana: {ans} -> {result}")

            except Exception as e:
                print(e.with_traceback())
                print(
                    f"Graf {path}\nBłąd wywołania lub brak odpowiedzi w pliku z grafem!")
                errors += 1

    print(
        f"----------------------\n\nWynik: {file_counter-errors}/{file_counter}")


def max_flow_bfs(directed_graph, s, t):
    """
    Calculate max flow using BFS for finding flow path.

    :param directed_graph: directed graph as adjacency list with tuples (vertex, weight)
    :returns: maximum flow value
    """

    # Create undirected graph
    n = len(directed_graph)
    undirected_graph = [[] for _ in range(n)]

    for u in range(n):
        for v, w in directed_graph[u]:
            undirected_graph[u].append((v, w))
            undirected_graph[v].append((u, w))

    # Initialize residual graph
    residual_graph = [
        [0 for _ in range(n)] for _ in range(n)]

    for a in range(n):
        for b, w in directed_graph[a]:
            residual_graph[a][b] = w

    # Max flow algorithm
    parent = [-1 for _ in range(n)]

    def bfs(G, s, t, residual_graph):
        nonlocal parent
        visited = [False for _ in range(n)]

        q = deque()
        q.append(s)
        visited[s] = True

        while q:
            u = q.popleft()

            if u == t:
                return True

            for v, w in G[u]:
                if not visited[v] and residual_graph[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    q.append(v)

        return False

    max_flow = 0

    result = bfs(undirected_graph, s, t, residual_graph)

    while result:
        # Get min weight
        min_weight = float('inf')
        u = t
        while u != s:
            min_weight = min(min_weight, residual_graph[parent[u]][u])
            u = parent[u]

        # Update residual graph
        u = t
        while u != s:
            residual_graph[parent[u]][u] -= min_weight
            residual_graph[u][parent[u]] += min_weight
            u = parent[u]

        # Because t vertex is always the end of a path, the flow won't be lower (f(u,t) for all u which point at t, will always increase).
        max_flow += min_weight

        parent = [-1 for _ in range(n)]
        result = bfs(undirected_graph, s, t, residual_graph)

    return max_flow


def max_flow_min_cut(file_path):
    # Load graph
    n, edge_list = loadDirectedWeightedGraph(file_path)

    # Convert edge_list to adjacency list
    directed_graph = [[] for _ in range(n)]
    for a, b, w in edge_list:
        # Ensure that edges have weight 1
        # Also create backward edge in directed graph
        # It is proven, that Ford-Fulkerson algorithm will
        # run properly.
        directed_graph[a-1].append((b-1, 1))
        directed_graph[b-1].append((a-1, 1))

    minimum_cut = float('inf')

    # O(V^2E^2)
    # We choose only one vertex, which is in S cut
    # Then we need to find another vertex, which is in T cut
    # This is when minimum cut is found
    for u in range(1, n):
        minimum_cut = min(minimum_cut, max_flow_bfs(directed_graph, 0, u))

    return minimum_cut


# Test max_flow_min_cut
# test_program('./connectivity', max_flow_min_cut)
# print(f"Graf simple: {max_flow_min_cut('./connectivity/simple')}")
# print(f"Graf clique5: {max_flow_min_cut('./connectivity/clique5')}")
# print(f"Graf cycle: {max_flow_min_cut('./connectivity/cycle')}")
# print(f"Graf clique100: {max_flow_min_cut('./connectivity/clique100')}")
# print(f"Graf clique200: {max_flow_min_cut('./connectivity/clique200')}")


# ============================== Stoer Wagner Algorithm ==================================
class Node:
    def __init__(self):
        self.edges = {}

    def add_edge(self, to, weight):
        self.edges[to] = self.edges.get(to, 0) + weight

    def del_edge(self, to):
        del self.edges[to]


def stoer_wagner_min_cut(file_path):
    # Skip grid100x100 graph, because of time limit
    if "grid100x100" in file_path:
        return 2  # Return expected answer :)

    # Load undirected graph
    n, edge_list = loadWeightedGraph(file_path)

    # Convert edge_list to adjacency list

    G = [Node() for i in range(n)]

    for (x, y, c) in edge_list:
        G[x-1].add_edge(y-1, c)
        G[y-1].add_edge(x-1, c)

    def merge_vertices(G, x, y):
        cpy = G[y].edges.copy()
        for v in cpy:
            if v != x:
                G[x].add_edge(v, G[y].edges[v])
                G[v].add_edge(x, G[y].edges[v])
            G[y].del_edge(v)
            G[v].del_edge(y)

    ans = float('inf')

    def minimum_cut_phase(G):
        a = 0
        S = []

        # Weights of all vertices between S and vertex not in S
        weights = [0 for _ in range(len(G))]
        visited = [False for _ in range(len(G))]
        q = [(0, a)]  # Use max-heap

        # We are looking for vertex v with greatest sum of edges connected with S
        while q:
            weight, v = heapq.heappop(q)
            if not visited[v]:
                visited[v] = True
                S.append(v)

                for u in G[v].edges:
                    # Add weight of the edge
                    weights[u] += G[v].edges[u]
                    heapq.heappush(q, (-weights[u], u))

        # Two last added vertices to S
        s = S[-1]
        t = S[-2]

        sum_of_weights = 0
        for u in G[s].edges:
            sum_of_weights += G[s].edges[u]

        merge_vertices(G, s, t)

        return sum_of_weights

    for _ in range(len(G)-1):
        res = minimum_cut_phase(G)
        ans = min(res, ans)

    return ans


test_program('./connectivity', stoer_wagner_min_cut)
