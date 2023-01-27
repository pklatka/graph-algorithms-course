from dimacs import loadDirectedWeightedGraph
from collections import deque
import os
from time import process_time


def test_program(graph_directory, f):
    """ Test given function.

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
                    g = load_directed_graph(file_path)
                    ans = f(g)
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


def load_directed_graph(graph_path):
    """
    Convert graph as adjacency list

    :param graph_path: path to file with graph
    :returns: graph as adjacency list
    """
    # Load graph
    loaded_graph = loadDirectedWeightedGraph(graph_path)
    edge_list = loaded_graph[1]
    n = loaded_graph[0]

    # Convert edge_list to adjacency list
    directed_graph = [[] for _ in range(n)]
    for a, b, w in edge_list:
        directed_graph[a-1].append((b-1, w))

    return directed_graph


def max_flow_dfs(directed_graph):
    """
    Calculate max flow using DFS for finding flow path.

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
    s, t = 0, n-1
    parent = [-1 for _ in range(n)]

    def dfs(G, s, t, residual_graph):
        nonlocal parent
        n = len(G)
        visited = [False for _ in range(n)]

        def dfs_visit(G, u):
            nonlocal visited, parent, t, residual_graph

            if u == t:
                return True

            visited[u] = True

            for b, w in G[u]:
                if not visited[b] and residual_graph[u][b] > 0:
                    parent[b] = u
                    if dfs_visit(G, b):
                        return True

            return False

        return dfs_visit(G, s)

    max_flow = 0

    result = dfs(undirected_graph, s, t, residual_graph)

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
        result = dfs(undirected_graph, s, t, residual_graph)

    return max_flow


def max_flow_bfs(directed_graph):
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
    s, t = 0, n-1
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


# Test program
test_program("./flow", max_flow_bfs)
