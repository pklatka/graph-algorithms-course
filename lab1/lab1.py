import os
from dimacs import loadWeightedGraph
import heapq
from collections import deque


def test_program(graph_directory, f):
    print(f"Testujemy funkcję {f.__name__}")
    file_counter = 0
    errors = 0
    for path in os.listdir(graph_directory):
        # Check if current path is a file
        file_path = os.path.join(graph_directory, path)
        if os.path.isfile(file_path):
            file_counter += 1
            # Load answer
            result = "Odpowiedź prawidłowa"
            try:
                with open(file_path, 'r') as file:
                    line = file.readline()
                    expected_ans = int(line.split()[-1])

                    # Test function
                    ans = f(file_path)
                    if ans != expected_ans:
                        result = f"Odpowiedź błędna - otrzymano {ans} zamiast {expected_ans}"
                        errors += 1

            except:
                result = "Błąd wywołania lub brak odpowiedzi"
                errors += 1

            print(f"Graf {path} - {result}")

    print(f"Wynik: {file_counter-errors}/{file_counter}")


def solution_union_graph(graph_path):
    """
    Algorithm based on union/find structure
    1. we sort the edges by weights in descending order
    2. we perform union when given vertices in an edge are in separate sets
    3. when the vertices s, t are in the same set, this means they are connected, and
        since we are going in sorted order the largest weight is as small as possible

    Time complexity: O(ElogV)
    """

    class UFSet:
        def __init__(self, value) -> None:
            self.parent = self
            self.value = value
            self.rank = 0

    def find(x):
        if x.parent != x:
            x.parent = find(x.parent)
        return x.parent

    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return

        if x.rank > y.rank:
            y.parent = x

        else:
            x.parent = y
            if x.rank == y.rank:
                y.rank += 1

    V, E = loadWeightedGraph(graph_path)
    s, t = 1, 2

    E.sort(key=lambda x: x[2], reverse=True)
    sets = [UFSet(i) for i in range(V+1)]

    for a, b, w in E:
        if find(sets[a]) != find(sets[b]):
            # Union
            union(sets[a], sets[b])

        # Check if s and t has the same parent
        if find(sets[s]) == find(sets[t]):
            return w

    return -1


def solution_binary_search_dfs(graph_path):
    """
    Suppose we have an array with all possible edge weights.
    Then we can "create an array" with dfs calls where 0 -> succeeded in passing the graph,
    1 -> failed to pass the graph with the constraint on the minimum edge weight

    Such an array will usually be of the form: [0,0,0,0,1,1,1]

    So perform a binary search looking for that zero which is the rightmost.

    Time complexity: O((V+E)log(E))
    """
    def dfs(G, min_cost, s, t):
        n = len(G)
        visited = [False for _ in range(n)]
        stack = deque()
        stack.append(s)
        while stack:
            u = stack.pop()

            if u == t:
                return True

            visited[u] = True
            for v, cost in G[u]:
                if cost >= min_cost and not visited[v]:
                    stack.append(v)

        return False

    def binary_search_rightmost(arr, G, s, t):
        left = 0
        right = len(arr)-1

        while left <= right:
            mid = left + (right - left) // 2
            res = dfs(G, arr[mid], s, t)

            if not res:
                right = mid - 1
            else:
                left = mid + 1

        if right < len(arr):
            return arr[right]

        return -1

    V, E = loadWeightedGraph(graph_path)
    s, t = 1, 2
    n = V+1

    G = [[] for _ in range(n)]

    h = set()

    # Convert graph
    for a, b, w in E:
        G[a].append((b, w))
        G[b].append((a, w))
        h.add(w)

    h = sorted(list(h))

    return binary_search_rightmost(h, G, s, t)


def solution_dijkstra(graph_path):
    """
    An algorithm based on Dijkstra.
    We use a queue of type max so that we always get the largest possible weight.
    We then relax:
        - we set the new weight to the minimum of the existing minimum-maximum weight and the tested edge
        - we check if the new weight is greater than the existing one -> if so, we replace it

    Time complexity: O(ElogV)
    """
    V, E = loadWeightedGraph(graph_path)
    s, t = 1, 2
    n = V+1

    G = [[] for _ in range(n)]

    # Convert graph
    for a, b, w in E:
        G[a].append((b, w))
        G[b].append((a, w))

    q = []
    d = [0 for _ in range(n)]
    d[s] = float('inf')
    q.append((-d[s], s))

    while q:
        tmp, v = heapq.heappop(q)

        for u, w in G[v]:
            weight = min(w, d[v])

            if weight > d[u]:
                d[u] = weight
                heapq.heappush(q, (-d[u], u))

    return d[t]


# Test algorithms
test_program('./graphs', solution_union_graph)
test_program('./graphs', solution_binary_search_dfs)
test_program('./graphs', solution_dijkstra)
