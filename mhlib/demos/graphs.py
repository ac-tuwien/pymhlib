"""Functions to read or create graphs for demo problems."""

import networkx as nx


def create_or_read_simple_graph(name: str) -> nx.Graph:
    """Read a simple unweighted graph from the specified file or create random G_n,m graph if name is gnm-n-m.

    The nodes are labeled beginning with 0.

    File format:
        - ``c <comments>    #`` ignored
        - ``p <name> <number of nodes> <number of edges>``
        - ``e <node_1> <node_2>    #`` for each edge, nodes are labeled in 1...number of nodes
    """
    if name.startswith('gnm-'):
        # create random G_n,m graph
        par = name.split(sep='-')
        return nx.gnm_random_graph(int(par[1]), int(par[2]), int(par[3]) if len(par) == 4 else None)
    else:  # read from file
        graph: nx.Graph = nx.Graph()
        with open(name) as f:
            for line in f:
                flag = line[0]
                if flag == 'p':
                    split_line = line.split(' ')
                    n = int(split_line[2])
                    # m = int(split_line[3])
                    graph.add_nodes_from(range(n))
                elif flag == 'e':
                    split_line = line.split(' ')
                    u = int(split_line[1]) - 1
                    v = int(split_line[2]) - 1
                    graph.add_edge(u, v)
        return graph
