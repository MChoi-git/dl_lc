


def build_graph():
    graph = [
        [1, 2, 3],
        [0, 4, 5],
        [0],
        [0, 6, 7],
        [1, 8, 9],
        [1],
        [3, 10, 11],
        [3],
        [4],
        [4],
        [6],
        [6],
    ]
    bfs_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dfs_order = [0, 1, 4, 8, 9, 5, 2, 3, 6, 10, 11, 7]

    return graph, bfs_order, dfs_order


def _dfs(graph, node, visited, _visited):
    # Visit node.
    visited.append(node)
    _visited.add(node)

    # Recurse on each neighbour that hasn't been visited yet.
    for n in graph[node]:
        if n not in _visited:
            _dfs(graph, n, visited, _visited)

    return visited


def dfs(graph, root):
    visited = []
    _visited = set()    # Avoid O(n) visited neighbour lookup.
    return _dfs(graph, root, visited, _visited)


def bfs(graph, root):
    visited = []
    _visited = set()    # Avoid O(n) visited neighbour lookup.
    queue = []

    # Enqueue root.
    queue.append(root)
    while len(queue) != 0:
        # Pop next node.
        node = queue.pop()

        # Visit the node.
        visited.append(node)
        _visited.add(node)

        # Enqueue its neigbours that havent been visited yet.
        unseen = [n for n in graph[node] if n not in _visited]
        for neighbour in unseen:
            queue.insert(0, neighbour)

    return visited


def main():
    graph, bfs_order, dfs_order = build_graph()

    my_bfs = bfs(graph, root=0)
    my_dfs = dfs(graph, root=0)

    print(bfs_order)
    print(my_bfs)
    print(dfs_order)
    print(my_dfs)

    assert bfs_order == my_bfs
    assert dfs_order == my_dfs


if __name__ == "__main__":
    main()
