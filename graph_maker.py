import itertools

import osmnx as ox
import networkx as nx

ox.config(use_cache=True, log_console=True)
ox.__version__


# function for adding weights to edges
# needs some work for good estimations but should be good for now
def add_leaf_weight(a_graph):
    for edge in a_graph.edges(data=True):
        _, _, temp_data = edge
        temp_highway = temp_data['highway']
        if temp_highway == 'residential':
            temp_data['leaves'] = temp_data['length'] / 10
        else:
            temp_data['leaves'] = 0
    return a_graph


# going to break the graph into different sections while maintaining graph structure
def break_up_graph(a_graph):
    graphs = []

    return graphs

def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    distances = {}
    i = 0
    j = 0
    for pair in pairs:
        try:
            path = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
        except:
            j += 1


        distances[pair] = path

        if i % 1000 == 0:
            print(i)
            print(j)
        i += 1
    return distances

def create_complete_graph(pair_weights, flip_weights=True):
    """
    Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    Parameters:
        pair_weights: list[tuple] from the output of get_shortest_paths_distances
        flip_weights: Boolean. Should we negate the edge attribute in pair_weights?
    """
    g = nx.Graph()
    for k, v in pair_weights.items():
        weight = - v if flip_weights else v
        # g.add_edge(k[0], k[1], {'distance': v, 'weight': wt_i})  # deprecated after NX 1.11
        g.add_edge(k[0], k[1], **{'distance': v})
    return g

def chinese_postman(a_graph):

    nodes_more_in_degree = []
    nodes_more_out_degree = []

    for a_node in a_graph.nodes():
        d_in = a_graph.in_degree(a_node)
        d_out = a_graph.out_degree(a_node)

        if d_in > d_out:
            num_less = d_in - d_out
            for i in range(0, num_less):
                nodes_more_in_degree.append(a_node)

        if d_out > d_in:
            num_less = d_out - d_in
            for i in range(0, num_less):
                nodes_more_out_degree.append(a_node)

    print("length of in nodes: ", len(nodes_more_in_degree))
    print("length of out nodes: ", len(nodes_more_out_degree))
    odd_node_pairs = []
    for a_node in nodes_more_in_degree:
        for b_node in nodes_more_out_degree:
            odd_node_pairs.append((a_node, b_node))

    odd_node_pairs_shortest_paths = get_shortest_paths_distances(a_graph, odd_node_pairs, 'distance')
    sub_graph = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

    print("starting matching")
    odd_matching_dupes = nx.algorithms.max_weight_matching(sub_graph, True)

    print(odd_matching_dupes)

    print("total number of odd degree nodes: ", len(nodes_odd_degree))
    print("total number of odd degree pairs: ", len(odd_node_pairs))

def main():
    G = ox.graph_from_place('Worcester, Massachusetts', network_type='drive_service')
    # fig, ax = ox.plot_graph(G, fig_height=8, node_size=0, edge_linewidth=0.5)

    # ec = ox.get_edge_colors_by_attr(G, attr='length')

    dump1 = ox.get_nearest_node(G, (42.233354, -71.788443), method='euclidean')
    dump2 = ox.get_nearest_node(G, (42.261926, -71.822740), method='euclidean')
    dump3 = ox.get_nearest_node(G, (42.313539, -71.772521), method='euclidean')
    ox.core.add_edge_lengths(G)
    # ox.plot_graph(G, edge_color=ec)
    a = G.nodes(data=True)
    b = G.edges(data=True)
    G = add_leaf_weight(G)
    b = G.edges(data=True)
    print(a)
    route1 = nx.shortest_path(G, dump3, dump2, weight='length')
    route2 = nx.shortest_path(G, dump3, dump1, weight='length')
    print(route1)
    routes = [route1, route2]
    # a = chinese_postman(G)
    tour = nx.eulerian_circuit(G, dump1)
    H = G.to_directed()
    a = chinese_postman(G)

    # ox.plot_graph_routes(H, routes)


if __name__ == "__main__":
    main()
