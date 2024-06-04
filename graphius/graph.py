from collections import deque
import heapq

import graphviz as gv

from .visual import Visual


class Vertex:
    def __init__(self, name):
        self.name = name
        self.color = "white"
        self.p = None
        self.d = None
        self.f = None


class Graph:
    def __init__(self, adjacency_list={}, names=None, directed=False, weighted=False):
        self.names = sorted(names) if names else sorted(adjacency_list.keys())
        self.vertices = {name: Vertex(name) for name in self.names}
        self.directed = directed
        self.weighted = weighted

        self.adjacency_list = adjacency_list

        self.visualize_adjacency_list = []
        self.weights = {}
        for vertex in self.names:
            for edge in self.adjacency_list[vertex]:
                if not isinstance(edge, (tuple, list)):
                    edge = (edge, 0)
                if (
                    (edge[0], vertex) not in self.visualize_adjacency_list
                ) or self.directed:
                    self.visualize_adjacency_list.append((vertex, edge[0]))
                    self.weights[(vertex, edge[0])] = edge[1]

        self.visualize_edge_color = {
            edge: "black" for edge in self.visualize_adjacency_list
        }
        for edge in self.visualize_adjacency_list:
            self.visualize_edge_color[reversed(edge)] = "black"

    def sum_weights(self):
        if self.directed:
            return sum(self.weights.values())
        return sum(self.weights.values()) // 2

    def visualize(self, label=None):
        g = gv.Digraph(format="png") if self.directed else gv.Graph(format="png")
        if label:
            g.attr(label=label)
        for vertex in self.vertices.values():
            # label = f'{vertex.name} | {vertex.d} | {vertex.f} | {vertex.p}'
            label = vertex.name
            if vertex.d == float("inf") or vertex.d is None:
                label += " | INF"
            else:
                label += f" | {vertex.d}"
            if vertex.f is not None:
                if vertex.f == float("inf"):
                    label += " | INF"
                else:
                    label += f" | {vertex.f}"
            if vertex.p:
                label += f" | {vertex.p}"
            else:
                label += " | None"

            fontcolor = "white" if vertex.color == "black" else "black"
            g.node(
                vertex.name,
                fillcolor=vertex.color,
                style="filled",
                fontcolor=fontcolor,
                label=label,
            )
        for edge in self.visualize_adjacency_list:
            g.edge(
                edge[0],
                edge[1],
                label=f"{self.weights[edge]}" if self.weighted else "",
                color=self.visualize_edge_color[edge],
                fontsize="15",
                penwidth="3",
            )
        return g

    def bfs(self, source, visualized=False, gif_path="bfs.gif"):
        visited = []

        if visualized:
            gif = Visual(gif_path)

        for vertex in self.vertices.values():
            vertex.color = "white"
            vertex.d = float("inf")
            vertex.p = None

        self.vertices[source].color = "gray"
        self.vertices[source].d = 0
        self.vertices[source].p = None

        if visualized:
            gif.add_state(self, label="Initialization")

        Q = deque()
        Q.append(source)

        while Q:
            u = Q.popleft()
            for v in (
                map(lambda x: x[0], self.adjacency_list[u])
                if self.weighted
                else self.adjacency_list[u]
            ):
                if self.vertices[v].color == "white":
                    self.vertices[v].color = "gray"
                    self.vertices[v].d = self.vertices[u].d + 1
                    self.vertices[v].p = u
                    Q.append(v)

            self.vertices[u].color = "black"
            visited.append(u)

            if visualized:
                gif.add_state(self, label=f"Exploring {u}")

        if visualized:
            gif.save_gif()
            return visited, gif_path

        return visited

    def print_path(self, source, destination):
        if source == destination:
            print(source)
        elif self.vertices[destination].p is None:
            print("No path from", source, "to", destination, "exists")
        else:
            self.print_path(source, self.vertices[destination].p)
            print(destination)

    def dfs(
        self,
        start_dfs_tree=None,
        discover_func=None,
        finish_func=None,
        order=None,
        visualized=False,
        gif_path="dfs.gif",
    ):
        if visualized:
            gif = Visual(gif_path)

        def dfs_visit(self, u, discover_func, finish_func):
            nonlocal time, visualized
            if visualized:
                nonlocal gif

            time += 1
            self.vertices[u].d = time
            self.vertices[u].color = "gray"
            if visualized:
                gif.add_state(self, label=f"Discovering {u}")

            for v in (
                map(lambda x: x[0], self.adjacency_list[u])
                if self.weighted
                else self.adjacency_list[u]
            ):
                if self.vertices[v].color == "white":
                    if discover_func:
                        discover_func(v)
                    self.vertices[v].p = u
                    dfs_visit(self, v, discover_func, finish_func)
            time += 1
            self.vertices[u].f = time
            self.vertices[u].color = "black"
            if visualized:
                gif.add_state(self, label=f"Finishing {u}")
            if finish_func:
                finish_func(u)

        time = 0
        for vertex in self.vertices.values():
            vertex.color = "white"
            vertex.p = None
            vertex.d = None
            vertex.f = None

        if visualized:
            gif.add_state(self, label="Initialization")

        if order is None:
            order = self.names

        for vertex in order:
            if self.vertices[vertex].color == "white":
                if start_dfs_tree:
                    start_dfs_tree()
                if discover_func:
                    discover_func(vertex)
                dfs_visit(self, vertex, discover_func, finish_func)

        if visualized:
            gif.save_gif()
            return gif_path

    def topological_sort(self):
        ordered_list = deque()
        self.dfs(finish_func=lambda u: ordered_list.appendleft(u))
        return list(ordered_list)

    def transpose(self):
        transposed_adjacency_list = {vertex: [] for vertex in self.names}
        for vertex in self.names:
            for edge in self.adjacency_list[vertex]:
                transposed_adjacency_list[edge[0]].append(
                    (vertex, edge[1]) if len(edge) == 2 else vertex
                )
        return Graph(
            transposed_adjacency_list,
            names=self.names.copy(),
            directed=self.directed,
            weighted=self.weighted,
        )

    def strongly_connected_components(self):
        components = []
        self.transpose().dfs(
            start_dfs_tree=lambda: components.append([]),
            discover_func=lambda u: components[-1].append(u),
            order=self.topological_sort(),
        )
        return components

    def mst_kruskal(self, visualized=False, gif_path="mst_kruskal.gif"):
        if visualized:
            gif = Visual(gif_path)
            gif.add_state(self)

        def find_set(u):
            if u != p[u]:
                p[u] = find_set(p[u])
            return p[u]

        def union(u, v):
            p[find_set(u)] = find_set(v)

        mst = []
        p = {vertex: vertex for vertex in self.names}
        edges = sorted(self.visualize_adjacency_list, key=lambda x: self.weights[x])
        for edge in edges:
            u, v = edge
            if find_set(u) != find_set(v):
                mst.append((edge, self.weights[edge]))
                self.visualize_edge_color[edge] = "red"
                union(u, v)

                if visualized:
                    gif.add_state(self)

        kruskal_adjacency_list = {vertex: [] for vertex in self.names}
        for edge in mst:
            kruskal_adjacency_list[edge[0][0]].append((edge[0][1], edge[1]))
            kruskal_adjacency_list[edge[0][1]].append((edge[0][0], edge[1]))

        graph_mst = Graph(
            kruskal_adjacency_list,
            names=self.names.copy(),
            directed=False,
            weighted=True,
        )
        for vertex in graph_mst.vertices.values():
            vertex.p = self.vertices[vertex.name].p
            vertex.d = self.vertices[vertex.name].d

        if visualized:
            gif.save_gif()
            return graph_mst, gif_path

        return graph_mst

    def mst_prim(self, r, visualized=False, gif_path="mst_prim.gif"):
        if visualized:
            gif = Visual(gif_path)

        for vertex in self.vertices.values():
            vertex.d = float("inf")
            vertex.p = None
            vertex.color = "white"

        # make all edges black
        self.visualize_edge_color = {
            edge: "black" for edge in self.visualize_adjacency_list
        }

        self.vertices[r].d = 0
        Q = [(vertex.d, vertex.name) for vertex in self.vertices.values()]
        heapq.heapify(Q)

        if visualized:
            gif.add_state(self)

        while Q:
            u = heapq.heappop(Q)[1]
            self.vertices[u].color = "black"
            for v in self.adjacency_list[u]:
                if v[0] in [vertex[1] for vertex in Q] and v[1] < self.vertices[v[0]].d:
                    self.vertices[v[0]].p = u
                    self.vertices[v[0]].d = v[1]
                    for i in range(len(Q)):
                        if Q[i][1] == v[0]:
                            Q[i] = (v[1], v[0])
                            heapq.heapify(Q)
                            break

            if visualized:
                gif.add_state(self)

        prim_adjacency_list = {vertex: [] for vertex in self.names}
        for vertex in self.vertices.values():
            if vertex.p:
                prim_adjacency_list[vertex.p].append((vertex.name, vertex.d))
                prim_adjacency_list[vertex.name].append((vertex.p, vertex.d))
                self.visualize_edge_color[(vertex.p, vertex.name)] = "red"
                self.visualize_edge_color[(vertex.name, vertex.p)] = "red"
                if visualized:
                    gif.add_state(self)

        graph_prim = Graph(
            prim_adjacency_list, names=self.names.copy(), directed=False, weighted=True
        )
        for vertex in graph_prim.vertices.values():
            vertex.p = self.vertices[vertex.name].p
            vertex.d = self.vertices[vertex.name].d

        if visualized:
            gif.save_gif()
            return graph_prim, gif_path

        return graph_prim

    def relax(self, u, v, w):
        if self.vertices[v].d > self.vertices[u].d + w:
            self.vertices[v].d = self.vertices[u].d + w
            self.vertices[v].p = u
            return True
        return False

    def bellman_ford(self, source, visualized=False, gif_path="bellman_ford.gif"):
        if visualized:
            gif = Visual(gif_path)

        for vertex in self.names:
            self.vertices[vertex].d = float("inf")
            self.vertices[vertex].p = None

        self.vertices[source].d = 0

        if visualized:
            gif.add_state(self, label="Initialization")

        for i in range(len(self.names) - 1):
            for u in self.names:
                for v, w in self.adjacency_list[u]:
                    old_p = self.vertices[v].p
                    self.relax(u, v, w)
                    if old_p != self.vertices[v].p:
                        if old_p:
                            self.visualize_edge_color[(old_p, v)] = "black"
                        self.visualize_edge_color[(u, v)] = "red"

                if visualized:
                    gif.add_state(self, label=f"Iteration {i + 1} | Exploring {u}")

        flag = True
        for u in self.names:
            for v, w in self.adjacency_list[u]:
                if self.vertices[v].d > self.vertices[u].d + w:
                    flag = False
                    break

        if visualized:
            gif.save_gif()
            return flag, gif_path

        return flag

    def dag_shortest_paths(
        self, source, visualized=False, gif_path="dag_shortest_paths.gif"
    ):
        if visualized:
            gif = Visual(gif_path)

        topological_order = self.topological_sort()

        for vertex in self.names:
            self.vertices[vertex].d = float("inf")
            self.vertices[vertex].p = None

        self.vertices[source].d = 0

        if visualized:
            gif.add_state(self, label="Initialization")

        for u in topological_order:
            for v, w in self.adjacency_list[u]:
                old_p = self.vertices[v].p
                self.relax(u, v, w)
                if old_p != self.vertices[v].p:
                    if old_p:
                        self.visualize_edge_color[(old_p, v)] = "black"
                    self.visualize_edge_color[(u, v)] = "red"

                if visualized:
                    gif.add_state(self)

        if visualized:
            gif.save_gif()
            return gif_path

    def dijkstra(self, source, visualized=False, gif_path="dijkstra.gif"):
        if visualized:
            gif = Visual(gif_path)

        for vertex in self.names:
            self.vertices[vertex].d = float("inf")
            self.vertices[vertex].p = None

        self.vertices[source].d = 0
        Q = [(vertex.d, vertex.name) for vertex in self.vertices.values()]
        heapq.heapify(Q)

        if visualized:
            gif.add_state(self, label="Initialization")

        while Q:
            u = heapq.heappop(Q)[1]
            for v, w in self.adjacency_list[u]:
                old_p = self.vertices[v].p
                if self.relax(u, v, w):
                    for i in range(len(Q)):
                        if Q[i][1] == v:
                            Q[i] = (self.vertices[v].d, v)
                            heapq.heapify(Q)
                            break
                if old_p != self.vertices[v].p:
                    if old_p:
                        self.visualize_edge_color[(old_p, v)] = "black"
                    self.visualize_edge_color[(u, v)] = "red"

            if visualized:
                gif.add_state(self, label=f"Exploring {u}")

        if visualized:
            gif.save_gif()
            return gif_path

    def floyd_warshall(self):
        D = [[float("inf")] * len(self.names) for _ in range(len(self.names))]
        P = [[None] * len(self.names) for _ in range(len(self.names))]
        for i in range(len(self.names)):
            D[i][i] = 0
            for j, w in self.adjacency_list[self.names[i]]:
                j = self.names.index(j)
                D[i][j] = w
                P[i][j] = i

        for k in range(len(self.names)):
            for i in range(len(self.names)):
                for j in range(len(self.names)):
                    if D[i][j] > D[i][k] + D[k][j]:
                        D[i][j] = D[i][k] + D[k][j]
                        P[i][j] = P[k][j]

        return D, P
