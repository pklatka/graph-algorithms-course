from dimacs import loadDirectedWeightedGraph, loadWeightedGraph
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
                    ans = int(f(file_path))
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


class Node:
    def __init__(self, idx):
        self.idx = idx
        self.out = set()  # set of neighbours
        self.RN = set()  # set of neighbours that appeared in O before v, where O is LexBFS order
        self.parent = None  # last appearing vertex in RN

    def connect_to(self, v):
        self.out.add(v)

    def __hash__(self) -> int:
        return hash(self.idx)


def initialize_graph(filename):
    (V, L) = loadWeightedGraph(filename)

    G = [Node(i) for i in range(V)]

    for (u, v, _) in L:
        G[u-1].connect_to(G[v-1])
        G[v-1].connect_to(G[u-1])

    return G


def lexBFS(G: Node):
    # Create list of two sets: vertices that aren't neighbours of v, neighbours of v.
    # Every iteration we will take random vertex from neighbours of v set.
    L = [set([G[i] for i in range(len(G))])]
    visited = []

    for i in range(len(G)):
        # Get random vertex, from neighbour set.
        v = L[-1].pop()

        visited.append(v)

        # For each set separate vertices that are neighbours of v and vertices that aren't. Remove old set.
        i = 0
        while i < len(L):
            s = L[i]
            neighbour = s & v.out
            not_neighbour = s - v.out

            if len(neighbour) > 0:
                L.insert(i+1, neighbour)

            # Insert not_neighbour before neighbour
            if len(not_neighbour) > 0:
                L.insert(i+1, not_neighbour)

            L.remove(L[i])
            i += 1

        # Update RN and parent for v
        neighbours_candidates = set(visited)

        # Select only neighbours that appeared before v
        G[v.idx].RN = neighbours_candidates & v.out

        # Find parent of v
        visited_candidates = visited[:]
        while visited_candidates:
            parent = visited_candidates.pop()
            if parent in G[v.idx].RN:
                G[v.idx].parent = parent
                break

    return visited


def checkLexBFS(G, vs):
    n = len(G)
    pi = [None] * n

    vs = list(map(lambda x: x.idx, vs))

    for i, v in enumerate(vs):
        pi[v] = i

    for i in range(n-1):
        for j in range(i+1, n-1):
            Ni = set(map(lambda x: x.idx, G[vs[i]].out))
            Nj = set(map(lambda x: x.idx, G[vs[j]].out))

            verts = [pi[v] for v in Nj - Ni if pi[v] < i]
            if verts:
                viable = [pi[v] for v in Ni - Nj]
                if not viable or min(verts) <= min(viable):
                    return False
    return True


def check_if_graph_is_chordal(filename):
    G = initialize_graph(filename)
    lex_bfs_order = lexBFS(G)

    if not checkLexBFS(G, lex_bfs_order):
        return False

    # Do not check first vertex, because it has no parent
    for v in lex_bfs_order[1:]:
        if not G[v.idx].RN - {G[v.idx].parent} <= G[G[v.idx].parent.idx].RN:
            return False

    return True


test_program('chordal', check_if_graph_is_chordal)
