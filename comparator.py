from analyzer import get_centrality_measures
from scipy import stats
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import time

centrality_names = ["max component size", "degree centrality", "betweenness centrality", "eigenvector centrality",
                    "transitivity", "clustering coefficient", "closeness"]


def ks_test(l1, l2):
    """Returns the ks statistic and the p-value That the 2 samples,
    l1 and l2 were pulled from the same distribution,
    computed using the ks test
    """
    ks_stat, pvalue = stats.ks_2samp(l1, l2)
    return ks_stat, pvalue

def mean_with_nonetype(list):
    """Calculates the mean of a list which may include some nonetype values--
    skips the nonetype values and only considers the numbers in the list
    """
    num_items = 0
    sum_items = 0
    for item in list:
        if not item == None: # if item is a number
            num_items += 1
            sum_items += item
    if num_items == 0:
        return 0
    else:
        return sum_items / num_items

def plot_error_distributions(centrality_matrix, name):
    """
    """
    (m,n) = centrality_matrix.shape
    # average 'across' rows:
    row_means = [None]*m
    for i in range(0,m):
        avg = mean_with_nonetype(centrality_matrix[i,:])
        row_means[i] = avg

    plt.clf()
    plt.hist(row_means,100)
    plt.xlabel("Vertex " + name + " Error then Average")
    plt.savefig('histograms/vertex_' + name + '_error_then_average.png')

    # average 'down' columns:
    col_means = [None]*n
    for j in range(0,n):
        avg = mean_with_nonetype(centrality_matrix[:,j])
        col_means[j] = avg

    plt.clf()
    plt.hist(col_means,25)
    plt.xlabel("Graph " + name + " Error then Average")
    plt.savefig('histograms/graph_' + name + '_error_then_average.png')

    total_avg = mean_with_nonetype(row_means)
    print("Average error overall:", total_avg)


def compare_centrality(centralities, master_cent, name):
    """ Compares a centrality measure for g1 and g2
    :param centralities: list of (dictionaries with key = node, val = centrality)
    :param master_cent: dictionary of centrality measures for master graph
    :param name: the name of the centrality measure
    :return: none
    """
    # creates a dictionary to store these values in
    # centrality_dict = {}
    vertex_list = []
    for g_dict in centralities:
        for vertex in g_dict:
            if vertex not in vertex_list:
                # centrality_dict[vertex] = [g_dict[vertex]]
                vertex_list.append(vertex)
            # else:
                # centrality_dict[vertex].append(g_dict[vertex])

    # creates a matrix to store these values in
    # initialize matrix with None values
    centrality_matrix = np.ndarray(shape=(len(vertex_list),len(centralities)), dtype=tuple)
    for g in range(len(centralities)):
        for vertex in centralities[g]:
            i = vertex_list.index(vertex)
            centrality_matrix[i,g] = centralities[g][vertex] # adds val to matrix

    # averages centrality measures for the same vertex across all graphs
    row_means = [None]*len(vertex_list)
    for i in range(0,len(vertex_list)):
        avg = mean_with_nonetype(centrality_matrix[i,:])
        row_means[i] = avg

    # apply error to row_means:
    for i in range(len(vertex_list)):
        if not row_means[i] == None:
            row_means[i] = abs(row_means[i] - master_cent[vertex_list[i]])

    # create some type of graph/print some value
    plt.clf()
    plt.hist(row_means, 100)
    plt.xlabel("Vertex " + name + " Average then Error")
    plt.savefig('histograms/vertex_' + name + '_average_then_error.png')

    # apply error to entire matrix:
    for i in range(len(vertex_list)):
        for j in range(len(centralities)):
            if not centrality_matrix[i,j] == None:
                centrality_matrix[i,j] = abs(centrality_matrix[i,j] - master_cent[vertex_list[i]])

    plot_error_distributions(centrality_matrix, name)

    # for key in centrality_dict:
    #     centrality_dict[key] = [np.mean(centrality_dict[key]), np.std(centrality_dict[key]), len(centrality_dict[key])]
    #
    # diff_dict = {}
    # for key in centrality_dict:
    #     diff_dict[key] = abs(master_cent[key] - centrality_dict[key][0])
    #
    # output = [np.mean(list(diff_dict.values())), np.std(list(diff_dict.values()))]
    # print("Mean", name, output[0])
    # print("Standard Deviation", name, output[1])

def plot_stored_data_distributions(num_graphs):
    """Calculates error and plots error distributions of the centrality data
    stored
    """
    # index of centrality measures which give values for each vertex:
    index = [1, 2, 3, 5]
    # store centralities over all graphs in a dictionary sorted by centrality type
    centralities = {"max component size":[], "degree centrality":[], "betweenness centrality":[], "eigenvector centrality":[],"transitivity":[], "clustering coefficient":[], "closeness":[]}
    # opens master centrality values file
    pkl_file = open('master_centralities1.pkl', 'rb')
    master_dict = pickle.load(pkl_file)
    pkl_file.close()

    # over all graph data files stored:
    for i in range(num_graphs):
        # opens data file
        pkl_file = open('freq_graph_data/freq_graph_' + str(i) + '.pickle', 'rb')
        g_data = pickle.load(pkl_file)
        pkl_file.close()
        # gets centrality measures
        g_cent = g_data[1]

        # over all centrality measures:
        for i in range(len(centrality_names)):
            # appends centrality measures from g to centralities dict list
            if i in index:
                centralities[centrality_names[i]].append(g_cent[centrality_names[i]])
            else:
                centralities[centrality_names[i]].append(g_cent[centrality_names[i]])

        # time to compare and create the distribution plots:
        for i in range(len(centrality_names)):
            print("Within comparator, comparing", centrality_names[i])
            if i in index:
                # sends list of specific centrality measure (over many graphs) to be compared
                compare_centrality(centralities[centrality_names[i]], master_dict[centrality_names[i]], centrality_names[i])

            #TODO: Requires testing.
            elif not centrality_names[i] == "closeness":
                avg_value = [np.mean(centralities[centrality_names[i]]), np.std(centralities[centrality_names[i]])]
                print("Mean", centrality_names[i], avg_value[0])
                print("Standard Deviation", centrality_names[i], avg_value[1])


def compare(graphs):
    """Compare a list of graphs by measuring top-n overlap for all n,
    R^2 measure, and KS measure
    :graphs: list of graphs
    :return: none
    """
    # index of centrality measures which give values for each vertex:
    index = [1, 2, 3, 5]
    # store centralities over all graphs in a dictionary sorted by centrality type
    centralities = {"max component size":[], "degree centrality":[], "betweenness centrality":[], "eigenvector centrality":[],"transitivity":[], "clustering coefficient":[], "closeness":[]}

    # over all graphs in our graph list:
    for g in graphs:
        # gets centrality measures
        g_cent = list(get_centrality_measures(g).values())
        # over all centrality measures we'll use/care about:
        for i in range(len(centrality_names)):
            # appends centrality measures from g to
            centralities[centrality_names[i]].append(g_cent[i])

    # TODO: Hard coded homophone strategy. Needs to be passed from earlier in the pipeline
    pkl_file = open('master_centralities1.pkl', 'rb')
    master_dict = pickle.load(pkl_file)


    for i in range(len(centrality_names)):
        print("Comparing", centrality_names[i])

        if i in index:
            # sends list of specific centrality measure (over many graphs) to be compared
            compare_centrality(centralities[centrality_names[i]], master_dict[centrality_names[i]], centrality_names[i])
        #TODO: Requires testing.
        # else:
        #     avg_value = [np.mean(centralities[centrality_names[i]]), np.std(centralities[centrality_names[i]])]
        #     print("Mean", centrality_names[i], avg_value[0])
        #     print("Standard Deviation", centrality_names[i], avg_value[1])
    pkl_file.close()

def compare_centrality_overlap(centrality1, centrality2, g1, g2, name):
    """ Compares a centrality measure for g1 and g2
    :param centrality1: list of centrality value for each node in g1
    :param centrality2: list of centrality value for each node in g2
    :param g1: graph 1
    :param g2: graph 2
    :param name: the name of the centrality measure
    :return:
    """
    nodes1 = [x for _, x in sorted(zip(centrality1.values(), g1.nodes()))]
    nodes2 = [x for _, x in sorted(zip(centrality2.values(), g2.nodes()))]
    nodes1.reverse()
    nodes2.reverse()
    top_n_overlap = compute_overlap(nodes1, nodes2)
    plt.clf()
    plt.plot(top_n_overlap)
    plt.savefig(centrality_names[name] + '.png')
    # print out the actual overlap of the two list of nodes
    print("Size of intersect:", len(set(nodes1) & set(nodes2)))
    # Compute pearson correlation between overlapping nodes's centrality value
    intersection = set(nodes1) & set(nodes2)
    overlap1 = [centrality1[i] for i in intersection]
    overlap2 = [centrality2[i] for i in intersection]
    print("R^2 measure", stats.pearsonr(overlap1, overlap2))
    print("KS measure", ks_test(list(centrality1.values()), list(centrality2.values())))

def compare_two_graphs(g1, g2):
    """Compare two graphs by measuring top-n overlap for all n ,R^2, and KS
    """
    print("Number of nodes in G1 and G2", nx.number_of_nodes(g1), nx.number_of_nodes(g2))
    # sort nodes by centrality measure
    centralities1 = get_centrality_measures(g1).values()
    centralities2 = get_centrality_measures(g2).values()
    # index of the centrality measure to compare
    index = [1, 2, 3, 5]
    for i in range(len(centralities1)):
        print("Comparing", centrality_names[i])
        if i in index:
            compare_centraly_overlap(centralities1[i], centralities2[i], g1, g2, i)
        else:
            print("graph1", centralities1[i], "graph2", centralities2[i])

def compute_overlap(l1, l2):
    """ Computes and returns overlap between two lists
    """
    top_n_overlap = []
    overlap = 0
    total = 0
    if len(l1) > len(l2):
        longer = l1
        shorter = l2
    else:
        longer = l2
        shorter = l1
    temp1 = []
    temp2 = []
    for i in range(len(longer)):
        total += 1
        if len(shorter) > i:
            if shorter[i] == longer[i]:
                overlap += 1
            elif shorter[i] in temp1 and longer[i] in temp2:
                overlap += 2
                total -= 1
            elif shorter[i] in temp1:
                overlap += 1
                temp1.append(longer[i])
            elif longer[i] in temp2:
                overlap += 1
                temp2.append(shorter[i])
            else:
                total += 1
                temp1.append(longer[i])
                temp2.append(shorter[i])
        top_n_overlap.append(overlap / total)
    return top_n_overlap


def store_data(graphs):
    """ Compute the centrality measures for each graph and store it
    """
    start = time.time()
    for i in range(len(graphs)):
        last = time.time()
        data = [graphs[i], get_centrality_measures(graphs[i])]

        with open('freq_graph_data/freq_graph_' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(data, f)
        print("Calculated centrality of graph", i, ",", time.time()-last, "sec")
    print("Finished centrality of ", len(graphs), "graphs,", time.time()-start, "sec")


def test_compare():
    """ tests the above functions!
    """
    print("comparing....")
    pkl_file = open('subgraph1_centralities_(2,.5,1).pkl', 'rb')
    subgraph1_cent = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('subgraph2_centralities_(2,.5,1).pkl', 'rb')
    subgraph2_cent = pickle.load(pkl_file)
    pkl_file.close()

    # print(subgraph1_cent)

    subgraph_cents = [subgraph1_cent, subgraph2_cent]
    centralities = {"max component size":[], "degree centrality":[], "betweenness centrality":[], "eigenvector centrality":[],"transitivity":[], "clustering coefficient":[], "closeness":[]}

    for g_cent in subgraph_cents:
        for i in range(len(centrality_names)):
            # appends centrality measures from g to
            centralities[centrality_names[i]].append(g_cent[centrality_names[i]])

    pkl_file = open('master_centralities1.pkl', 'rb')
    master_dict = pickle.load(pkl_file)
    # index of centrality measures which give values for each vertex:
    index = [1, 2, 3, 5]
    for i in range(len(centrality_names)):
        print("Comparing", centrality_names[i])

        if i in index:
            # sends list of specific centrality measure (over many graphs) to be compared
            compare_centrality(centralities[centrality_names[i]], master_dict[centrality_names[i]], centrality_names[i])

        #TODO: Requires testing.
        # else:
        #     avg_value = [np.mean(centralities[centrality_names[i]]), np.std(centralities[centrality_names[i]])]
        #     print("Mean", centrality_names[i], avg_value[0])
        #     print("Standard Deviation", centrality_names[i], avg_value[1])
    pkl_file.close()

if __name__ == '__main__':
    test_compare()
