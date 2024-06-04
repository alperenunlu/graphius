from graphius import Graph


def random_graph(n, m, directed=False, weighted=False):
    import random
    from itertools import product

    names = product([chr(i) for i in range(ord("a"), ord("z") + 1)], repeat=3)
    names = ["".join(name) for name in names][:n]
    adjacency_list = {name: [] for name in names}
    for _ in range(m):
        u, v = random.sample(names, 2)
        if u != v and v not in map(lambda x: x[0], adjacency_list[u]):
            if weighted:
                w = random.randint(1, 100)
            else:
                w = 0
            adjacency_list[u].append((v, w))
            if not directed:
                adjacency_list[v].append((u, w))

    return Graph(adjacency_list, names=names, directed=directed, weighted=weighted)


def test_bfs():
    adjacency_list = {
        "w": ["r", "v", "x", "z"],
        "x": ["w", "y", "z"],
        "y": ["u", "v", "x"],
        "z": ["w", "x"],
        "r": ["s", "t", "w"],
        "s": ["r", "u", "v"],
        "t": ["r", "u"],
        "u": ["t", "s", "y"],
        "v": ["s", "w", "y"],
    }

    g = Graph(adjacency_list)
    visited = g.bfs("s")
    assert visited == ["s", "r", "u", "v", "t", "w", "y", "x", "z"], "BFS failed"


def test_print_path():
    adjacency_list = {
        "w": ["r", "v", "x", "z"],
        "x": ["w", "y", "z"],
        "y": ["u", "v", "x"],
        "z": ["w", "x"],
        "r": ["s", "t", "w"],
        "s": ["r", "u", "v"],
        "t": ["r", "u"],
        "u": ["t", "s", "y"],
        "v": ["s", "w", "y"],
    }

    g = Graph(adjacency_list)
    g.bfs("s")
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    g.print_path("s", "z")
    path = sys.stdout.getvalue()
    sys.stdout = old_stdout
    assert path == "s\nr\nw\nz\n", "Path failed"


def test_dfs():
    adjacency_list = {
        "m": ["q", "r", "x"],
        "n": ["o", "q", "u"],
        "o": ["r", "s", "v"],
        "p": ["o", "s", "z"],
        "q": ["t"],
        "r": ["u", "y"],
        "s": ["r"],
        "t": [],
        "u": ["t"],
        "v": ["w", "x"],
        "w": ["z"],
        "x": [],
        "y": ["v"],
        "z": [],
    }

    g = Graph(adjacency_list, directed=True)
    visited = []
    g.dfs(finish_func=lambda x: visited.append(x))
    assert visited == [
        "t",
        "q",
        "u",
        "z",
        "w",
        "x",
        "v",
        "y",
        "r",
        "m",
        "s",
        "o",
        "n",
        "p",
    ], "DFS failed"


def test_topological_sort():
    adjacency_list = {
        "m": ["q", "r", "x"],
        "n": ["o", "q", "u"],
        "o": ["r", "s", "v"],
        "p": ["o", "s", "z"],
        "q": ["t"],
        "r": ["u", "y"],
        "s": ["r"],
        "t": [],
        "u": ["t"],
        "v": ["w", "x"],
        "w": ["z"],
        "x": [],
        "y": ["v"],
        "z": [],
    }

    g = Graph(adjacency_list, directed=True)
    assert g.topological_sort() == [
        "p",
        "n",
        "o",
        "s",
        "m",
        "r",
        "y",
        "v",
        "x",
        "w",
        "z",
        "u",
        "q",
        "t",
    ], "Topological Sort failed"


def test_mst():
    from random import sample

    g = random_graph(1000, 1500, directed=False, weighted=True)

    for name in sample(g.names, 100):
        assert (
            g.mst_kruskal().sum_weights() == g.mst_prim(name).sum_weights()
        ), "Kruskal and Prim do not return the same minimum weight spanning tree"


def test_bellman_ford():
    adjacency_list = {
        "s": [("t", 6), ("y", 7)],
        "t": [("x", 5), ("y", 8), ("z", -4)],
        "x": [("t", -2)],
        "y": [("x", -3), ("z", 9)],
        "z": [("s", 2), ("x", 7)],
    }

    g = Graph(adjacency_list, weighted=True, directed=True)

    assert g.bellman_ford("s") == True, "Bellman-Ford failed"
    assert [v.d for v in g.vertices.values()] == [0, 2, 4, 7, -2], "Bellman-Ford failed"


def test_dag_shortest_path():
    adjacency_list = {
        "r": [("s", 5), ("t", 3)],
        "s": [("t", 2), ("x", 6)],
        "t": [("x", 7), ("y", 4), ("z", 2)],
        "x": [("y", -1), ("z", 1)],
        "y": [("z", -2)],
        "z": [],
    }

    g = Graph(adjacency_list, weighted=True, directed=True)
    g.dag_shortest_paths("s")
    assert [v.d for v in g.vertices.values()] == [
        float("inf"),
        0,
        2,
        6,
        5,
        3,
    ], "DAG Shortest Paths failed"


def test_dijkstra():
    adjacency_list = {
        "s": [("t", 10), ("y", 5)],
        "t": [("x", 1), ("y", 2)],
        "x": [("z", 4)],
        "y": [("t", 3), ("x", 9), ("z", 2)],
        "z": [("s", 7), ("x", 6)],
    }

    g = Graph(adjacency_list, weighted=True, directed=True)
    g.dijkstra("s")
    assert [v.d for v in g.vertices.values()] == [0, 8, 9, 5, 7], "Dijkstra failed"


def test_floyd_warshall():
    adjacency_list = {
        "1": [("2", 3), ("3", 8), ("5", -4)],
        "2": [("4", 1), ("5", 7)],
        "3": [("2", 4)],
        "4": [("1", 2), ("3", -5)],
        "5": [("4", 6)],
    }

    g = Graph(adjacency_list, weighted=True, directed=True)

    test_values = (
        [
            [0, 1, -3, 2, -4],
            [3, 0, -4, 1, -1],
            [7, 4, 0, 5, 3],
            [2, -1, -5, 0, -2],
            [8, 5, 1, 6, 0],
        ],
        [
            [None, 2, 3, 4, 0],
            [3, None, 3, 1, 0],
            [3, 2, None, 1, 0],
            [3, 2, 3, None, 0],
            [3, 2, 3, 4, None],
        ],
    )

    assert g.floyd_warshall() == test_values, "Floyd-Warshall failed"


test_bfs()
test_print_path()
test_dfs()
test_topological_sort()
test_mst()
test_bellman_ford()
test_dag_shortest_path()
test_dijkstra()
test_floyd_warshall()

print("All tests passed.")
