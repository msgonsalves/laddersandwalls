import itertools

import osmnx as ox
import networkx as nx
import pandas as pd

ox.config(use_cache=True, log_console=True)
ox.__version__


class Truck:

    def __init__(self):
        self.__capacity_left = 30
        self.__path = []

    def drive_path(self, path, last_dump, dump1, dump2, dump3,  graph):

        start_node = path[0][0]
        start_path = nx.dijkstra_path(graph, last_dump, start_node)


        for a_node in start_path:
            self.__path.append(a_node)

        removed_edges = []
        print(len(self.__path))
        for an_edge in path:
            edge_data = graph.get_edge_data(an_edge[0], an_edge[1])
            self.__capacity_left -= edge_data[0]['leaves']
            if self.__capacity_left < 0:
                self.__capacity_left += edge_data[0]['leaves']
                break
            edge_data[0]['leaves'] = 0
            self.__path.append(an_edge[1])
            removed_edges.append(an_edge)

        dump1_pathl = nx.dijkstra_path_length(graph, self.__path[-1], dump1)
        dump2_pathl = nx.dijkstra_path_length(graph, self.__path[-1], dump2)
        dump3_pathl = nx.dijkstra_path_length(graph, self.__path[-1], dump3)

        end_path = []
        print(len(self.__path))
        end_node = 0
        if dump1_pathl < dump2_pathl:
            if dump1_pathl < dump3_pathl:
                end_path = nx.dijkstra_path(graph, self.__path[-1], dump1)
                end_node = dump1
            elif dump3_pathl < dump2_pathl:
                end_path = nx.dijkstra_path(graph, self.__path[-1], dump3)
                end_node = dump3
        else:
            if dump2_pathl < dump3_pathl:
                end_path = nx.dijkstra_path(graph, self.__path[-1], dump2)
                end_node = dump2
        for a_node in end_path[1:]:
            self.__path.append(a_node)

        for an_edge in removed_edges:
            path.remove(an_edge)

        return self.__path, end_node


# function for adding weights to edges
# needs some work for good estimations but should be good for now
def add_leaf_weight(a_graph):
    for edge in a_graph.edges(data=True):
        _, _, temp_data = edge
        temp_highway = temp_data['highway']
        if temp_highway == 'residential':
            temp_data['leaves'] = temp_data['length'] / 25
        else:
            temp_data['leaves'] = 0
        temp_data['trail'] = 'original'
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

        # print("degree in/out one: ", graph.in_degree(pair[0]), " / ", graph.out_degree(pair[0]))
        # print("degree in/out two: ", graph.in_degree(pair[1]), " / ", graph.out_degree(pair[1]))
        node1 = graph.nodes(pair[0])
        node2 = graph.nodes(pair[1])

        d1_in = graph.in_degree(pair[0])
        d1_out = graph.out_degree(pair[0])
        d2_in = graph.in_degree(pair[1])
        d2_out = graph.out_degree(pair[1])

        if d1_in < d1_out or d2_in > d2_out:
            if d1_in < d1_out:
                print("node1 in/out: ", d1_in, " / ", d1_out)
            else:
                print("node2 in/out: ", d2_in, " / ", d2_out)
        else:
            path = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
            distances[pair] = path

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
        g.add_edge(k[0], k[1], **{'length': v, 'weight': weight})
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
                    edges_to_add.append([an_edge[1], an_edge[0], an_edge[2]['length'], an_edge[2]['leaves'], an_edge[2]['trail']])

        if d_out > d_in:
            if d_in == 0:
                edges = a_graph.out_edges(nbunch=a_node, data=True)
                print("one way in")
                for an_edge in edges:
                    length = an_edge[2]['length']
                    length = length * 3
                    edges_to_add.append([an_edge[1], an_edge[0], an_edge[2]['length'], an_edge[2]['leaves'], an_edge[2]['trail']])

        if d_out == 0 and d_in == 0:
            print("adding node: ", a_node, " to remove")
            remove_nodes.append(a_node)

    for e in edges_to_add:
        a_graph.add_edge(e[0], e[1], **{'length': e[2], 'leaves': e[3], 'trail': e[4]})

    for a_node in remove_nodes:
        a_graph.remove_node(a_node)

    print("finsihed graph")
    return a_graph


def make_aug_graph(a_graph, a_matching):
    graph_aug = nx.MultiDiGraph(a_graph.copy())

    for a_pair in a_matching:
        # print("adding edge: ", a_pair)
        one_in = graph_aug.in_degree(a_pair[0])
        one_out = graph_aug.out_degree(a_pair[0])
        two_in = graph_aug.in_degree(a_pair[1])
        two_out = graph_aug.out_degree(a_pair[1])
        # print("vertex one in/out: ", one_in, "/", one_out)
        # print("vertex two in/out: ", two_in, "/", two_out)
        graph_aug.add_edge(a_pair[0], a_pair[1],
                           **{'length': nx.dijkstra_path_length(a_graph, a_pair[0], a_pair[1]), 'trail': 'augmented'})

        one_in = graph_aug.in_degree(a_pair[0])
        one_out = graph_aug.out_degree(a_pair[0])
        two_in = graph_aug.in_degree(a_pair[1])
        two_out = graph_aug.out_degree(a_pair[1])
        if one_in != one_out:

            print("vertex one in/out: ", one_in, "/", one_out)
            print("vertex two in/out: ", two_in, "/", two_out)

        if two_in != two_out:
            print("vertex one in/out: ", one_in, "/", one_out)
            print("vertex two in/out: ", two_in, "/", two_out)

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
            aug_path = nx.dijkstra_path(graph_original, edge[0], edge[1], weight='length')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

            # print('Filling in edges for augmented edge: {}'.format(edge))
            # print('Augmenting path: {}'.format(' => '.join(aug_path)))
            # print('Augmenting path pairs: {}\n'.format(aug_path_pairs))

            # If `edge` does not exist in original graph, find the shortest path between its nodes and
            #  add the edge attributes for each link in the shortest path.
            test_edges = graph_original.edges(aug_path, data=True)


            for i in range(0, len(aug_path)-1):
                edge_aug_att = graph_original.get_edge_data(aug_path[i], aug_path[i+1])
                euler_circuit.append((aug_path[i], aug_path[i+1], edge_aug_att))
            # for edge_aug in aug_path_pairs:
            #     edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
            #     euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))

    return euler_circuit


def check_matchin(matching, more_out, more_in):
    length = 0
    new_matching = []
    for a_pair in matching:
        length += 1

        if a_pair[0] not in more_out:
            new_matching.append((a_pair[1], a_pair[0]))
            # print("some node set was backwards")

        else:
            new_matching.append(a_pair)


    print("length of matching: ", len(matching))
    return new_matching


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
                print("degree in/out: ", d_in, " / ", d_out)
                nodes_more_in_degree.append(a_node)

        if d_out > d_in:
            num_less = d_out - d_in
            if d_in == 0:
                print("we found a one way into the city", nx.all_neighbors(a_graph, a_node))
                print()

            for i in range(0, num_less):
                print("degree in/out: ", d_in, " / ", d_out)
                nodes_more_out_degree.append(a_node)

    print("length of in nodes: ", len(nodes_more_in_degree))
    print("length of out nodes: ", len(nodes_more_out_degree))
    odd_node_pairs = []
    for a_node in nodes_more_in_degree:
        for b_node in nodes_more_out_degree:
            odd_node_pairs.append((a_node, b_node))

    odd_node_pairs_shortest_paths = get_shortest_paths_distances(a_graph, odd_node_pairs, 'length')
    sub_graph = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

    print("starting matching")
    odd_matching_dupes = nx.algorithms.max_weight_matching(sub_graph, True)
    print(odd_matching_dupes)
    new_matching = check_matchin(odd_matching_dupes, nodes_more_in_degree, nodes_more_out_degree)
    # print(new_matching)
    print("making the augmented graph")
    aug_graph = make_aug_graph(a_graph, new_matching)

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

    print("total number of odd degree pairs: ", len(odd_node_pairs))


    return aug_graph
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

    H = nx.DiGraph(G)

    H = create_usable_network(G)

    test = H.to_undirected()
    print(nx.is_connected(test))
    print(nx.is_strongly_connected(G))
    subgraphs = nx.strongly_connected_component_subgraphs(H)
    print(subgraphs)

    max = 0
    largest_graph = 0
    for sub in subgraphs:
        print("a")
        print(len(sub.nodes()))
        if len(sub.nodes()) > max:
            largest_graph = sub
            max = len(sub.nodes())

    print(len(largest_graph.nodes()))

    # a = chinese_postman(largest_graph, dump1)
    # a = chinese_postman(a, dump1)
    a = nx.read_gpickle("eularian_circuit_graph.gpickle")
    nx.write_gpickle(a, "eularian_circuit_graph.gpickle")
    euler_circuit = create_eulerian_circuit(a, H, dump1)
    total_mileage_of_circuit = 0
    for edge in euler_circuit:

        edge_data = H.get_edge_data(edge[0], edge[1])
        total_mileage_of_circuit += edge_data[0]['length']

    total_mileage_on_orig_trail_map = sum(nx.get_edge_attributes(H, 'length').values())
    _vcn = pd.value_counts(pd.value_counts([(e[0]) for e in euler_circuit]), sort=False)
    node_visits = pd.DataFrame({'n_visits': _vcn.index, 'n_nodes': _vcn.values})
    _vce = pd.value_counts(
        pd.value_counts([sorted(e)[0] + sorted(e)[1] for e in nx.MultiDiGraph(euler_circuit).edges()]))
    edge_visits = pd.DataFrame({'n_visits': _vce.index, 'n_edges': _vce.values})

    # Printing stats
    print('Mileage of circuit: {0:.2f}'.format(total_mileage_of_circuit))
    print('Mileage on original trail map: {0:.2f}'.format(total_mileage_on_orig_trail_map))
    print('Mileage retracing edges: {0:.2f}'.format(total_mileage_of_circuit - total_mileage_on_orig_trail_map))
    print('Percent of mileage retraced: {0:.2f}%\n'.format(
        (1 - total_mileage_of_circuit / total_mileage_on_orig_trail_map) * -100))

    print('Number of edges in circuit: {}'.format(len(euler_circuit)))
    print('Number of edges in original graph: {}'.format(len(a.edges())))
    print('Number of nodes in original graph: {}\n'.format(len(a.nodes())))

    print('Number of edges traversed more than once: {}\n'.format(len(euler_circuit) - len(a.edges())))

    print('Number of times visiting each node:')
    print(node_visits.to_string(index=False))

    print('\nNumber of times visiting each edge:')
    print(edge_visits.to_string(index=False))
    # ox.plot_graph_routes(H, routes)

    euler_circuit_copy = []

    for an_edge in euler_circuit:
        euler_circuit_copy.append((an_edge))

    trucks = []
    paths = []
    last_node = dump1
    while euler_circuit_copy:
        temp_truck = Truck()

        temp_path, last_node = temp_truck.drive_path(euler_circuit_copy, last_node, dump1, dump2, dump3, H)

        trucks.append(temp_truck)
        paths.append(temp_path)
        for i in range(0, len(temp_path)-1):
            if temp_path[i] == temp_path[i+1]:
                print(temp_path[i])
                print(i)
            try:
                data = min(H.get_edge_data(temp_path[i], temp_path[i+1]).values(), key=lambda x: x['length'])
            except:
                try_path = nx.dijkstra_path(H, temp_path[i], temp_path[i+1])
                print(try_path)
                print(temp_path[i])
                print(temp_path[i+1])
                print(i)

        edges = H.edges(temp_path, data=True)

        for e in edges:

            if not type(e):
                print(e)

    for a_path in paths:
        ox.plot_graph_route(H, a_path)





if __name__ == "__main__":
    main()
