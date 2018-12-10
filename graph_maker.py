import itertools

import osmnx as ox
import networkx as nx
import pandas as pd

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
            shortest_path = nx.shortest_path(graph, pair[0], pair[1])
            path = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
            distances[pair] = path
        except:
            j += 1

        if i % 1000 == 0:
            print(i)
            print(j)
            j = 0
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
    print("making graph now")
    for k, v in pair_weights.items():
        weight = - v if flip_weights else v
        # g.add_edge(k[0], k[1], {'distance': v, 'weight': wt_i})  # deprecated after NX 1.11
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': weight})
    return g


def create_usable_network(a_graph):
    remove_nodes = []
    edges_to_add = []

    for a_node in a_graph.nodes():
        d_in = a_graph.in_degree(a_node)
        d_out = a_graph.out_degree(a_node)
        if d_in > d_out:
            if d_out == 0:
                edges = a_graph.in_edges(nbunch=a_node, data=True)
                print("one way out")
                for an_edge in edges:
                    # neighbor, _ = an_edge
                    # other_edge = a_graph.get_edge_data(neighbor, a_node)
                    # print(other_edge)
                    edges_to_add.append([an_edge[1], an_edge[0], an_edge[2]['length'], an_edge[2]['leaves']])
                    # a_graph.add_edge(an_edge[1], an_edge[0],
                    #                  **{'length': an_edge[2]['length'] * 3, 'leaves': an_edge[2]['leaves']})

        if d_out > d_in:
            if d_in == 0:
                edges = a_graph.out_edges(nbunch=a_node, data=True)
                print("one way in")
                for an_edge in edges:
                    # neighbor, _ = an_edge
                    #
                    # other_edge = a_graph.get_edge_data(a_node, neighbor)
                    # print(other_edge)
                    length = an_edge[2]['length']
                    length = length * 3
                    edges_to_add.append([an_edge[1], an_edge[0], an_edge[2]['length'], an_edge[2]['leaves']])
                    # a_graph.add_edge(an_edge[0], an_edge[1],
                    #                  **{'length': length, 'leaves': an_edge[2]['leaves']})

        if d_out == 0 and d_in == 0:
            print("adding node: ", a_node, " to remove")
            remove_nodes.append(a_node)

    for e in edges_to_add:
        a_graph.add_edge(e[0], e[1], **{'length': e[2], 'leaves': e[3]})

    for a_node in remove_nodes:
        a_graph.remove_node(a_node)

    print("finsihed graph")
    return a_graph


def make_aug_graph(a_graph, a_matching):
    graph_aug = nx.DiGraph(a_graph.copy())

    for a_pair in a_matching:
        graph_aug.add_edge(a_pair[0], a_pair[1],
                           **{'distance': nx.dijkstra_path_length(a_graph, a_pair[0], a_pair[1]), 'trail': 'augmented'})

    return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    """Create the eulerian path using only edges from the original graph."""
    euler_circuit = []
    naive_circuit = list(nx.eulerian_circuit(graph_augmented))

    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])

        if edge_data[0]['trail'] != 'augmented':
            # If `edge` exists in original graph, grab the edge attributes and add to eulerian circuit.
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att))
        else:
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='distance')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

            print('Filling in edges for augmented edge: {}'.format(edge))
            print('Augmenting path: {}'.format(' => '.join(aug_path)))
            print('Augmenting path pairs: {}\n'.format(aug_path_pairs))

            # If `edge` does not exist in original graph, find the shortest path between its nodes and
            #  add the edge attributes for each link in the shortest path.
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))

    return euler_circuit


def chinese_postman(a_graph, start_node):
    nodes_more_in_degree = []
    nodes_more_out_degree = []

    for a_node in a_graph.nodes():
        d_in = a_graph.in_degree(a_node)
        d_out = a_graph.out_degree(a_node)

        if d_in > d_out:
            if d_out == 0:
                print("we found a one way out of the city", nx.all_neighbors(a_graph, a_node))
                print()
            num_less = d_in - d_out
            for i in range(0, num_less):
                nodes_more_in_degree.append(a_node)

        if d_out > d_in:
            num_less = d_out - d_in
            if d_in == 0:
                print("we found a one way into the city", nx.all_neighbors(a_graph, a_node))
                print()

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

    print("making the augmented graph")
    aug_graph = make_aug_graph(a_graph, odd_matching_dupes)

    second_in = []
    second_out = []
    for a_node in aug_graph.nodes():

        d_in = aug_graph.in_degree(a_node)
        d_out = aug_graph.out_degree(a_node)

        if d_in > d_out:
            num_less = d_in - d_out
            for i in range(0, num_less):
                second_in.append(a_node)

        if d_out > d_in:
            num_less = d_out - d_in
            for i in range(0, num_less):
                second_out.append(a_node)

    print("length of in nodes: ", len(second_in))
    print("length of out nodes: ", len(second_out))

    # euler_circuit = create_eulerian_circuit(aug_graph, a_graph, start_node)

    print("total number of odd degree pairs: ", len(odd_node_pairs))

    # total_mileage_of_circuit = sum([edge[2]['distance'] for edge in euler_circuit])
    # total_mileage_on_orig_trail_map = sum(nx.get_edge_attributes(a_graph, 'distance').values())
    # _vcn = pd.value_counts(pd.value_counts([(e[0]) for e in euler_circuit]), sort=False)
    # node_visits = pd.DataFrame({'n_visits': _vcn.index, 'n_nodes': _vcn.values})
    # _vce = pd.value_counts(
    #     pd.value_counts([sorted(e)[0] + sorted(e)[1] for e in nx.MultiDiGraph(euler_circuit).edges()]))
    # edge_visits = pd.DataFrame({'n_visits': _vce.index, 'n_edges': _vce.values})
    #
    # # Printing stats
    # print('Mileage of circuit: {0:.2f}'.format(total_mileage_of_circuit))
    # print('Mileage on original trail map: {0:.2f}'.format(total_mileage_on_orig_trail_map))
    # print('Mileage retracing edges: {0:.2f}'.format(total_mileage_of_circuit - total_mileage_on_orig_trail_map))
    # print('Percent of mileage retraced: {0:.2f}%\n'.format(
    #     (1 - total_mileage_of_circuit / total_mileage_on_orig_trail_map) * -100))
    #
    # print('Number of edges in circuit: {}'.format(len(euler_circuit)))
    # print('Number of edges in original graph: {}'.format(len(a_graph.edges())))
    # print('Number of nodes in original graph: {}\n'.format(len(a_graph.nodes())))
    #
    # print('Number of edges traversed more than once: {}\n'.format(len(euler_circuit) - len(g.edges())))
    #
    # print('Number of times visiting each node:')
    # print(node_visits.to_string(index=False))
    #
    # print('\nNumber of times visiting each edge:')
    # print(edge_visits.to_string(index=False))


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
    route1 = nx.shortest_path(G, dump3, dump2, weight='length')
    route2 = nx.shortest_path(G, dump3, dump1, weight='length')
    routes = [route1, route2]
    # a = chinese_postman(G)
    tour = nx.eulerian_circuit(G, dump1)
    test = G.to_undirected()
    print(nx.is_connected(test))

    H = G.to_directed()

    H = create_usable_network(H)
    H = create_usable_network(H)
    # H = create_usable_network(H)
    # H = create_usable_network(H)

    a = chinese_postman(H, dump1)

    # ox.plot_graph_routes(H, routes)


if __name__ == "__main__":
    main()
