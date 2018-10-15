import operator
from comparator import compute_overlap
import matplotlib.pyplot as plt
import numpy as np
from centrality_measures import *
import pickle
import csv
import lexicon_generator
from lexicon_generator import HomophoneStrat, LexModMethod
import pandas as pd
import os
import random
from sklearn.metrics import r2_score

csvfile = open("closeness.csv","w")
writer = csv.writer(csvfile)
INF = 40000


def power_law(G, nodes):
    num_edges = len(G.edges())
    edges = []
    for i in nodes:
        edges.append(num_edges)
        num_edges -= len(G[i])
        G.remove_node(i)
    edges.reverse()
    # Fit a linear regression to the data point
    x = np.log10(np.arange(70, len(edges)))
    y = np.log10(np.array(edges[70:]))
    coef = np.polyfit(x, y, 1)
    r_square = r2_score(y, coef[0] * x + coef[1])
    plt.clf()
    plt.title('Number of edges versus number of nodes in log-log scales')
    plt.loglog(edges, label='Edges')
    plt.plot(10 ** (coef[0] * x + coef[1]), label='0.00018x^(1.81), r^2 = 0.98')
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of edges')
    plt.legend()
    plt.show()


def effective_diameter(G):
    st_paths_lengths = []
    all_pairs = nx.shortest_path_length(G)
    all_pairs = list(all_pairs)
    for node in all_pairs:
        for l in node[1]:
            if node[1][l] != 0:
                st_paths_lengths.append(node[1][l])
    st_paths_lengths.sort(reverse=True)
    length = len(st_paths_lengths)
    above_90 = [st_paths_lengths[0], 1]
    for i in range(length):
        if i+1 >= len(st_paths_lengths):
            below_90 = [st_paths_lengths[i],0]
            return (0.9 - below_90[1]) / (above_90[1] - below_90[1]) + below_90[0]
        if st_paths_lengths[i] != st_paths_lengths[i + 1]:
            percentage = i / length
            if percentage < 0.1:
                above_90 = [st_paths_lengths[i + 1], 1 - percentage]
            elif percentage == 0.1:
                return st_paths_lengths[i]
            else:
                below_90 = [st_paths_lengths[i + 1], 1 - percentage]
                return (0.9 - below_90[1]) / (above_90[1] - below_90[1]) + below_90[0]


def plot_diameter_over_time(G, nodes):
    # sorted_nodes = sorted(nodes.items(), key=operator.itemgetter(1))
    # diameter = []
    # sorted_nodes.reverse()
    # age = 18
    # diameter.append(effective_diameter(G))
    # for node in sorted_nodes:
    #     if node[1] >= age:
    #         G.remove_node(node[0])
    #     if node[1] < age :
    #         G.remove_node(node[0])
    #         diameter.append(effective_diameter(G))
    #         age -= 1
    # diameter.reverse()
    # print(diameter)
    diameter = [9.840641879118506, 9.841843559181193, 9.839656983315788, 9.844144962749402, 9.83577064179414, 9.798926616756905,
     9.574529945591715, 9.495997869381393, 9.331463075887138, 9.340149464004796, 9.477943867390252, 9.633296387493282,
     9.809700610983839, 10.760147955578432, 11.913687782805429, 12.268721804511278, 1.9428571428571428]
    diameter.reverse()
    x = np.arange(3, 20)
    plt.clf()
    plt.title('Effective diameter versus ages')
    plt.plot(x, diameter, label='Effective Diameter')
    plt.xlabel('Age')
    plt.ylabel('Effective Diameter')
    plt.legend()
    plt.show()


class DisjointSet:
    """ An implementation of a disjoint set data structure, with path compression 
    and union by rank. All elements are created at once, and are consecutive 
    integers starting at zero. Also there are lists of the nodes in each set,
    stored at the index of the root node"""
    # parents[i] holds the parent of i
    parents = []
    # rank[i] holds the rank of i
    rank = []
    # sets[i] holds the nodes in the set with root i
    sets = []

    def __init__(self, size):
        self.parents = list(range(size))
        self.rank = [0] * size
        self.sets = [[i] for i in range(size)]

    def find(self, i):
        if self.parents[i] != i:
            self.parents[i] = self.find(self.parents[i])
        return self.parents[i]

    def union(self, i, j):
        iroot = self.find(i)
        jroot = self.find(j)
        if iroot == jroot:
            return
        if self.rank[iroot] < self.rank[jroot]:
            self.parents[iroot] = jroot
            self.sets[jroot].extend(self.sets[iroot])
            self.sets[iroot] = []
        elif self.rank[iroot] > self.rank[jroot]:
            self.parents[jroot] = iroot
            self.sets[iroot].extend(self.sets[jroot])
            self.sets[jroot] = []
        else:
            self.parents[iroot] = jroot
            self.rank[jroot] += 1
            self.sets[jroot].extend(self.sets[iroot])
            self.sets[iroot] = []


def analyze_predictiveness(G, nodes):
    eigenvector_centrality = nx.eigenvector_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G)
    centralities = [degree_centrality, eigenvector_centrality, clustering]
    name = ["degree centrality", "eigenvector centrality", "clustering"]
    random = np.random.permutation(nodes.copy())
    plt.clf()
    for i in range(3):
        nodes_centrality = [x for _, x in sorted(zip(centralities[i].values(), G.nodes()))]
        nodes_centrality.reverse()
        top_n_overlap = compute_overlap(nodes_centrality, nodes)
        top_n_overlap_random = compute_overlap(nodes_centrality, random)
        top_n_overlap = np.array(top_n_overlap) - np.array(top_n_overlap_random)
        plt.plot(top_n_overlap, label=name[i])
        plt.legend()
    plt.show()


def plot_average_centralities(G, nodes, ordered_by):
    compute_centrality(G, nodes, ordered_by)
    pkl_file = open("incremental_pickles/average_centrality_" + ordered_by + ".pickle", 'rb')
    average_centralities = pickle.load(pkl_file)
    average_centralities[0].reverse()
    average_centralities[1].reverse()
    plt.clf()
    plt.title('Average Degree Centrality Ordered by ' + ordered_by.title())
    plt.plot(average_centralities[0])
    plt.xlabel("Number of nodes")
    plt.ylabel("Degree centrality")
    plt.show()
    
    plt.clf()
    plt.title('Average Clustering Coefficient Ordered by ' + ordered_by.title())
    plt.plot(average_centralities[1])
    plt.xlabel("Number of nodes")
    plt.ylabel("Clustering Coefficient")
    plt.show()
    
def plot_average_centralities_comparison(G, nodes):
    # sort nodes by frequency
    freq_dict = get_freq_dict()
    nodes.sort(key=lambda node: freq_dict[node], reverse=True)
    compute_centrality(G, nodes, "frequency")
    # sort nodes by aoa
    nodes.sort(key=lambda node: node_dict[node])
    compute_centrality(G, nodes, "aoa")
    # sort nodes randomly
    random.shuffle(nodes)
    compute_centrality(G, nodes, "random")
    freq_pickle = open("incremental_pickles/average_centrality_frequency.pickle", 'rb')
    aoa_pickle = open("incremental_pickles/average_centrality_aoa.pickle", 'rb')
    random_pickle = open("incremental_pickles/average_centrality_random.pickle", 'rb')

    freq_centralities = pickle.load(freq_pickle)
    freq_centralities[0].reverse()
    freq_centralities[1].reverse()
    
    aoa_centralities = pickle.load(aoa_pickle)
    aoa_centralities[0].reverse()
    aoa_centralities[1].reverse()
    
    random_centralities = pickle.load(random_pickle)
    random_centralities[0].reverse()
    random_centralities[1].reverse()
    plt.clf()
    plt.plot(freq_centralities[0], label='ordered by frequency', color='blue')
    plt.plot(aoa_centralities[0], label='ordered by aoa', color='green')
    plt.plot(random_centralities[0], label='ordered by random', color='red')
    plt.ylim((0,4.5))
    plt.xlim((0,31000))
    plt.title("Average degree centrality")
    plt.xlabel("Number of nodes")
    plt.ylabel("Degree centrality")
    plt.legend(loc='lower right')
#    plt.show()
    
    plt.clf()
    plt.plot(freq_centralities[1], label='ordered by frequency', color='blue')
    plt.plot(aoa_centralities[1], label='ordered by aoa', color='green')
    plt.plot(random_centralities[1], label='ordered by random', color='red')
    plt.ylim((0,0.4))
    plt.xlim((0,31000))
    plt.title("Average Clustering Coefficient")
    plt.xlabel("Number of nodes")
    plt.ylabel("Clustering coefficient")
    plt.legend()
#    plt.show()
    
def plot_centrality_error_comparison(G, nodes):
    # sort nodes by frequency
    freq_dict = get_freq_dict()
    nodes.sort(key=lambda node: freq_dict[node], reverse=True)
    compute_centrality_error(G, nodes, "frequency")
    # sort nodes by aoa
    nodes.sort(key=lambda node: node_dict[node])
    compute_centrality_error(G, nodes, "aoa")
    # sort nodes randomly
    random.shuffle(nodes)
    compute_centrality_error(G, nodes, "random")  
    freq_pickle = open("incremental_pickles/centrality_error_frequency.pickle", 'rb')
    aoa_pickle = open("incremental_pickles/centrality_error_aoa.pickle", 'rb')
    random_pickle = open("incremental_pickles/centrality_error_random.pickle", 'rb')
    
    freq_centralities = pickle.load(freq_pickle)
    #degree mean
    freq_centralities[1].reverse()
    #proportion of nodes with zero degree error
    freq_centralities[2].reverse()
    #clustering mean
    freq_centralities[4].reverse()
    #prop of nodes with zero clustering error
    freq_centralities[5].reverse()
    
    aoa_centralities = pickle.load(aoa_pickle)
    #degree error
    aoa_centralities[1].reverse()
    #proportion of nodes with zero degree error
    aoa_centralities[2].reverse()
    #clustering mean
    aoa_centralities[4].reverse()
    #prop of nodes with zero clustering error
    aoa_centralities[5].reverse()
    
    random_centralities = pickle.load(random_pickle)
    #degree mean
    random_centralities[1].reverse()
    #proportion of nodes with zero degree error
    random_centralities[2].reverse()
    #clustering mean
    random_centralities[4].reverse()
    #prop of nodes with zero clustering error
    random_centralities[5].reverse()
    
    plt.clf()
    plt.title('Average Relative Degree Centrality Error')
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(random_centralities[1], label='ordered by random', color='red')
    plt.plot(aoa_centralities[1], label='ordered by aoa', color='green')
    plt.plot(freq_centralities[1], label='ordered by frequency', color='blue')
    plt.ylim((0,1))
    plt.xlim((0,31000))
    plt.legend()
#    plt.show()

    plt.clf()
    plt.title("Proportions of Nodes with Correct Degree Centrality")
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(freq_centralities[2], label='ordered by frequency', color='blue')
    plt.plot(aoa_centralities[2], label='ordered by aoa', color='green')
    plt.plot(random_centralities[2], label='ordered by random', color='red')
    plt.legend(loc='lower right')
    plt.xlim((0,31000))
#    plt.show()
    
    plt.clf()
    plt.title('Average Relative Clustering Coefficient Error')
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(random_centralities[4], label='ordered by random', color='red')
    plt.plot(aoa_centralities[4], label='ordered by aoa', color='green')
    plt.plot(freq_centralities[4], label='ordered by frequency', color='blue')
    plt.legend()
    plt.xlim((0,31000))
#    plt.show()

    plt.clf()
    plt.title("Proportions of Nodes with Correct Clustering Coefficient")
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(freq_centralities[5], label='ordered by frequency', color='blue')
    plt.plot(aoa_centralities[5], label='ordered by aoa', color='green')
    plt.plot(random_centralities[5], label='ordered by random', color='red')
    plt.legend(loc='lower right')
    plt.xlim((0,31000))
#    plt.show()
        

    
def plot_centrality_error(G, nodes, ordered_by):
    compute_centrality_error(G, nodes, ordered_by)
    pkl_file = open("incremental_pickles/centrality_error_" + ordered_by + ".pickle", 'rb')
    average_centralities = pickle.load(pkl_file)
    #degree percentiles
    for list in average_centralities[0]:
        list.reverse()
    #degree mean
    average_centralities[1].reverse()
    #proportion of nodes with zero degree error
    average_centralities[2].reverse()
    #clustering percentiles
    for list in average_centralities[3]:
        list.reverse()
    #clustering mean
    average_centralities[4].reverse()
    #prop of nodes with zero clustering error
    average_centralities[5].reverse()
    plt.clf()
    plt.title('Average Relative Degree Centrality Error Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(average_centralities[1],label="Mean")
    plt.show()
    plt.clf()
    plt.title('Relative Degree Centrality Error Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(average_centralities[0][4], label='Maximum')
    plt.plot(average_centralities[0][3], label='75th percentile')
    plt.plot(average_centralities[1],label="Mean")
    plt.plot(average_centralities[0][2], label='Median')
    plt.plot(average_centralities[0][1], label='25th percentile')
    plt.plot(average_centralities[0][0], label='Minimum')
    plt.legend()
    plt.show()
    plt.clf()
    plt.title("Proportions of Nodes with Correct Degree Centrality Ordered by " + ordered_by.title())
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(average_centralities[2])
    plt.show()
    plt.clf()
    plt.title('Average Relative Clustering Coefficient Error Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(average_centralities[4],label="Mean")
    plt.show()
    plt.clf()
    plt.title('Relative Clustering Coefficient Error Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(average_centralities[3][4], label='Maximum')
    plt.plot(average_centralities[3][3], label='75th percentile')
    plt.plot(average_centralities[4],label="Mean")
    plt.plot(average_centralities[3][2], label='Median')
    plt.plot(average_centralities[3][1], label='25th percentile')
    plt.plot(average_centralities[3][0], label='Minimum')
    plt.legend()
    plt.show()
    plt.clf()
    plt.title("Proportions of Nodes with Correct Clustering Coefficient Ordered by " + ordered_by.title())
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(average_centralities[5])
    plt.show()
    

def compute_centrality(G, nodes, ordered_by):
    if not os.path.isfile('incremental_pickles/average_centrality_' + ordered_by + '.pickle'):
        degree_aoa = []
        clustering_aoa = []
        nodes = nodes.copy()
        nodes.reverse()
        degree = nx.degree_centrality(G)
        for i in degree:
            degree[i] *= nx.number_of_nodes(G) - 1
        clustering = nx.clustering(G)
        for i in range(len(nodes) - 2):
            degree_aoa.append(np.average(list(degree.values())))
            clustering_aoa.append(np.average(list(clustering.values())))
            degree = compute_degree_centrality(G, nodes[i], degree)
            clustering = compute_clustering(G, nodes[i], clustering)
            G.remove_node(nodes[i])
            if i % 5000 == 0:
                print(i)
        with open('incremental_pickles/average_centrality_' + ordered_by + '.pickle', 'wb') as f:
            pickle.dump([degree_aoa, clustering_aoa], f)

def compute_centrality_error(G, nodes, ordered_by):
    if not os.path.isfile("incremental_pickles/centrality_error_" + ordered_by + ".pickle"):
        degree_error_aoa_avg = []
        #contains lists of min, 25%, median, 75%, max
        degree_error_aoa_quartiles = [[] for i in range(5)]
        #prop of nodes with 0 error
        correct_degree_prop = []
        clustering_error_aoa_avg = []
        #contains lists of min, 25%, median, 75%, max
        clustering_error_aoa_quartiles = [[] for i in range(5)]
        #prop of nodes with 0 error
        correct_clustering_prop = []
        nodes = nodes.copy()
        nodes.reverse()
        true_degree = nx.degree_centrality(G)
        for i in true_degree:
            true_degree[i] *= nx.number_of_nodes(G) - 1
        degree = true_degree.copy()
        true_clustering = nx.clustering(G)
        clustering = true_clustering.copy()
        for i in range(len(nodes) - 2):
            #clustering_aoa.append(np.average(list(clustering.values())))
            #degree_error = {key: abs(degree[key] - true_degree.get(key, 0)) for key in degree.keys()}
            degree_error_list = []
            for key in degree.keys():
                if true_degree.get(key, 0) != 0:
                    degree_error_list.append(abs(degree[key] - true_degree.get(key, 0))/true_degree.get(key, 0))
                else:
                    assert degree[key] == 0
                    degree_error_list.append(0)

            if len(degree.keys()) != 0:
                degree_error_aoa_avg.append(sum(degree_error_list)/len(degree_error_list))
                degree_error_aoa_quartiles[0].append(min(degree_error_list))
                degree_error_aoa_quartiles[1].append(np.percentile(degree_error_list,25))
                degree_error_aoa_quartiles[2].append(np.percentile(degree_error_list,50))
                degree_error_aoa_quartiles[3].append(np.percentile(degree_error_list,75))
                degree_error_aoa_quartiles[4].append(max(degree_error_list))
                correct_degree_prop.append(degree_error_list.count(0)/len(degree_error_list))

            clustering_error_list = []
            for key in clustering.keys():
                if true_clustering.get(key, 0) != 0:
                    clustering_error_list.append(abs(clustering[key] - true_clustering.get(key, 0))/true_clustering.get(key, 0))
                else:
                    assert clustering[key] == 0
                    clustering_error_list.append(0)

            if len(degree.keys()) != 0:
                clustering_error_aoa_avg.append(sum(clustering_error_list)/len(clustering_error_list))
                clustering_error_aoa_quartiles[0].append(min(clustering_error_list))
                clustering_error_aoa_quartiles[1].append(np.percentile(clustering_error_list,25))
                clustering_error_aoa_quartiles[2].append(np.percentile(clustering_error_list,50))
                clustering_error_aoa_quartiles[3].append(np.percentile(clustering_error_list,75))
                clustering_error_aoa_quartiles[4].append(max(clustering_error_list))
                correct_clustering_prop.append(clustering_error_list.count(0)/len(clustering_error_list))

            degree = compute_degree_centrality(G, nodes[i], degree)
            clustering = compute_clustering(G, nodes[i], clustering)
            G.remove_node(nodes[i])
            if i % 5000 == 0:
                print(i)
        with open("incremental_pickles/centrality_error_" + ordered_by + ".pickle", 'wb') as f:
            pickle.dump([degree_error_aoa_quartiles, degree_error_aoa_avg, correct_degree_prop, \
                         clustering_error_aoa_quartiles, clustering_error_aoa_avg, correct_clustering_prop], f)
    return

    
    
def plot_closeness_comparison(G, nodes, binsize):
    # sort nodes by frequency
    freq_dict = get_freq_dict()
    nodes.sort(key=lambda node: freq_dict[node], reverse=True)
    compute_centrality_bins(G, nodes, "frequency", "closeness", binsize)
    # sort nodes by aoa
    nodes.sort(key=lambda node: node_dict[node])
    compute_centrality_bins(G, nodes, "aoa", "closeness", binsize)
    # sort nodes randomly
    random.shuffle(nodes)
    compute_centrality_bins(G, nodes, "random", "closeness", binsize)  
    freq_pickle = open('incremental_pickles/closeness_frequency_binsize_' + str(binsize) + '.pickle', 'rb')
    aoa_pickle = open('incremental_pickles/closeness_aoa_binsize_' + str(binsize) + '.pickle', 'rb')
    random_pickle = open('incremental_pickles/closeness_random_binsize_' + str(binsize) + '.pickle', 'rb')
    
    freq_centralities = pickle.load(freq_pickle)
    aoa_centralities = pickle.load(aoa_pickle)
    random_centralities = pickle.load(random_pickle)
#    print("loaded all pickles")
    
    # at max number nodes everything is correct
    for ordering_centrality in [freq_centralities, aoa_centralities, random_centralities]:
        for proportion in ['10p', '5p', '1p']:
            ordering_centrality['prop correct ' + proportion].reverse()
            ordering_centrality['prop correct ' + proportion].append(1)
    
    plt.clf()
    plt.plot(freq_centralities['num nodes'], freq_centralities['average'], label='ordered by frequency')
    plt.plot(aoa_centralities['num nodes'], aoa_centralities['average'], label='ordered by aoa')
    plt.plot(random_centralities['num nodes'], random_centralities['average'], label='ordered by random')
    plt.title("Average Closeness Centrality")
    plt.xlabel("Number of nodes")
    plt.ylabel("Closeness centrality")
    plt.legend(loc='upper left')
    plt.xlim((0,31000))
#    plt.show()
    
    plt.clf()
    plt.title('Average Relative Closeness Centrality Error')
    plt.xlabel('Number of words')
    plt.ylabel('average relative error')
    plt.plot(freq_centralities['num nodes'], freq_centralities['average error'],label="ordered by frequency")
    plt.plot(aoa_centralities['num nodes'], aoa_centralities['average error'],label="ordered by aoa")
    plt.plot(random_centralities['num nodes'], random_centralities['average error'],label="ordered by random")
    plt.xlim((0,31000))
    plt.legend()
#    plt.show()
    
    plt.clf()
    plt.title("Proportions of Nodes with Correct Closeness Centrality")
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(freq_centralities['num nodes'], freq_centralities['prop correct'],label="ordered by frequency")
    plt.plot(aoa_centralities['num nodes'], aoa_centralities['prop correct'],label="ordered by frequency")
    plt.plot(random_centralities['num nodes'], random_centralities['prop correct'],label="ordered by frequency")
    plt.legend(loc='upper left')
    plt.xlim((0,31000))
    plt.show()

    plt.clf()
    plt.title("Proportions of Nodes with Correct Closeness Centrality (5% error tolerance)")
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")

#    plt.plot(freq_centralities['num nodes'], freq_centralities['prop correct 10p'],label="ordered by frequency (10% error)")
    plt.plot(freq_centralities['num nodes'], freq_centralities['prop correct 5p'],label="ordered by frequency")
#    plt.plot(freq_centralities['num nodes'], freq_centralities['prop correct 1p'],label="ordered by frequency (1% error)")
#    plt.plot(aoa_centralities['num nodes'], aoa_centralities['prop correct 10p'],label="ordered by aoa (10% error)")
    plt.plot(aoa_centralities['num nodes'], aoa_centralities['prop correct 5p'],label="ordered by aoa")
#    plt.plot(aoa_centralities['num nodes'], aoa_centralities['prop correct 1p'],label="ordered by aoa (1% error)")
#    plt.plot(random_centralities['num nodes'], random_centralities['prop correct 10p'],label="ordered by random (10% error)")
    plt.plot(random_centralities['num nodes'], random_centralities['prop correct 5p'],label="ordered by random")
#    plt.plot(random_centralities['num nodes'], random_centralities['prop correct 1p'],label="ordered by random (1% error)")
    plt.legend(loc='upper left')
    plt.xlim((0,31000))
    plt.show()
    return

def plot_error_comparison_ordered_by_aoa(G, nodes, closeness_binsize):
    closeness_pickle = open('incremental_pickles/closeness_aoa_binsize_' + str(closeness_binsize) + '.pickle', 'rb')
    local_measures_pickle = open("incremental_pickles/centrality_error_aoa.pickle", 'rb')
    closeness = pickle.load(closeness_pickle)
    local_measures = pickle.load(local_measures_pickle)
    # reverse degree error
    local_measures[1].reverse()
    local_measures[4].reverse()

    plt.clf()
    plt.title('Average Relative Error (ordered by AoA)')
    plt.xlabel('Number of words')
    plt.ylabel('average relative error')
    plt.plot(local_measures[1], label="degree centrality", color='purple')
    plt.plot(local_measures[4], label="clustering coefficient", color='orange')
    plt.plot(closeness['num nodes'], closeness['average error'],label="closeness centrality", color='green')
    plt.xlim((0,31000))
    plt.legend()
    plt.show()


    
def plot_centrality_bins(G, nodes, ordered_by, measure, binsize):
    compute_centrality_bins(G, nodes, ordered_by, measure, binsize)
    pkl_file = open('incremental_pickles/' + measure + '_' + ordered_by + '_binsize_' + str(binsize) + ".pickle", 'rb')
    centralities = pickle.load(pkl_file)
    print(centralities['average error'])
    plt.clf()
    plt.title('Average ' + measure.title() + ' Centrality Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel(measure.title() + ' centrality')
    plt.plot(centralities['num nodes'], centralities['average'])
#    plt.show()
    plt.clf()
    plt.title('Average Relative ' + measure.title() + ' Centrality Error Ordered by ' + ordered_by.title(),y=1.05)
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
#    plt.plot(centralities['num nodes'][4:], centralities['average error'][4:])
    plt.plot(centralities['num nodes'], centralities['average error'])
#    plt.show()
    plt.clf()
    plt.title('Relative ' + measure.title() + ' Centrality Error Ordered by ' + ordered_by.title())
    plt.xlabel('Number of words')
    plt.ylabel('Average relative error')
    plt.plot(centralities['num nodes'], centralities['error quartiles'][4], label='Maximum')
    plt.plot(centralities['num nodes'], centralities['error quartiles'][3], label='75th percentile')
    plt.plot(centralities['num nodes'], centralities['average error'], label='Mean')
    plt.plot(centralities['num nodes'], centralities['error quartiles'][2], label='Median')
    plt.plot(centralities['num nodes'], centralities['error quartiles'][1], label='25th percentile')
    plt.plot(centralities['num nodes'], centralities['error quartiles'][0], label='Minimum')
    plt.legend()
#    plt.show()
    plt.clf()
    plt.title('Proportions of Nodes with Correct ' + measure.title() + ' Centrality Ordered by ' + ordered_by.title())
    plt.xlabel("Number of nodes")
    plt.ylabel("Proportion of nodes")
    plt.plot(centralities['num nodes'], centralities['prop correct'])
#    plt.show()
    plt.clf()
    
    
        
def compute_centrality_bins(G_original, nodes, ordered_by, measure, binsize):
    if not os.path.isfile('incremental_pickles/' + measure + '_' + ordered_by + '_binsize_' + str(binsize) + '.pickle'):
        G = G_original.copy()
        average = []
        average_error = []
        #prop of nodes with 0 error
        prop_correct = []
        #prop of nodes with 1% error
        prop_correct_1p = []
        #prop of nodes with 5% error
        prop_correct_5p = []
        #prop of nodes with 10% error
        prop_correct_10p = []
        #contains lists of min, 25%, median, 75%, max
        error_quartiles = [[] for i in range(5)]
        #contains the list of the number of nodes included
        num_nodes = []
        
        nodes = nodes.copy()
        nodes.reverse()
        
        #compute reference centrality
        if measure == 'eigenvector':
            ref_centrality = nx.eigenvector_centrality(G)
        else:
            ref_centrality = nx.closeness_centrality(G)
        #put in first values
        average.append(sum(ref_centrality.values())/len(ref_centrality.values()))
        average_error.append(0)
        prop_correct.append(1)
        for i in range(5):
            error_quartiles[i].append(0)
        num_nodes.append(len(ref_centrality))
        
        #remove first batch
        first_dec = len(nodes) % binsize
        G.remove_nodes_from(nodes[-1*first_dec:])
        nodes = nodes[:-1*first_dec]
        while len(nodes) > 0:
            try:
                centrality = {}
                if measure == 'eigenvector':
                    centrality = nx.eigenvector_centrality(G)
                else:
                    centrality = nx.closeness_centrality(G)
                average.append(sum(centrality.values())/len(centrality.values()))
                errors = []
                num_correct_within_1p = 0
                num_correct_within_5p = 0
                num_correct_within_10p = 0
                for key in centrality.keys():
                    if ref_centrality.get(key, 0) != 0:
                        error = abs(centrality[key]-ref_centrality[key])/ref_centrality[key]
                        errors.append(error)
                        if error < .1:
                            num_correct_within_10p += 1
                            if error < .05:
                                num_correct_within_5p += 1
                                if error < .01:
                                    num_correct_within_1p += 1
                                    
                average_error.append(sum(errors)/len(errors))
                error_quartiles[0].append(min(errors))
                error_quartiles[1].append(np.percentile(errors,25))
                error_quartiles[2].append(np.percentile(errors,50))
                error_quartiles[3].append(np.percentile(errors,75))
                error_quartiles[4].append(max(errors))
                prop_correct.append(errors.count(0)/len(errors))
                prop_correct_10p.append(num_correct_within_10p / len(errors))
                prop_correct_5p.append(num_correct_within_5p / len(errors))
                prop_correct_1p.append(num_correct_within_1p / len(errors))
                num_nodes.append(len(centrality.keys()))
                print(num_nodes[-1])
            except:
                print("failed to converge for " + len(nodes))
            
            G.remove_nodes_from(nodes[-1*binsize:])
            nodes = nodes[:-1*binsize]
            
        average.reverse()
        average_error.reverse()
        prop_correct.reverse()
        num_nodes.reverse()
        for list in error_quartiles:
            list.reverse()
        with open('incremental_pickles/' + measure + '_' + ordered_by + '_binsize_' + str(binsize) + '.pickle', 'wb') as f:
            pickle.dump({'average':average, 'average error':average_error, 'error quartiles':error_quartiles,
                         'prop correct':prop_correct, 'num nodes':num_nodes, 'prop correct 1p': prop_correct_1p,
                        'prop correct 5p': prop_correct_5p, 'prop correct 10p': prop_correct_10p}, f)
    
        
def plot_average_word_length(nodes):
    avg_lens = []
    tot_len = 0
    num_words = 0
    for word in nodes:
        num_words += 1
        tot_len += len(word)
        avg_lens.append(tot_len/num_words)
    plt.clf()
    plt.title('Average Word Length')
    plt.xlabel('Number of words')
    plt.ylabel('Average word length')
    plt.plot(avg_lens)
    plt.show()       

def getcloseness(pathlens):
    closenesslist = []
    # print(pathlens)
    for i in range(len(pathlens)):
        pathlentotal = 0
        connectednum = 0
        for j in range(len(pathlens)):
            if pathlens[i, j] != INF:
                pathlentotal += pathlens[i, j]
                connectednum += 1
        closenesslist.append(connectednum / pathlentotal)
    return closenesslist


def getclosenessreciprocal(pathlens):
    closenesslist = []
    # print(nodes[:len(pathlens)])
    # print(pathlens)
    if (len(pathlens) % 100 == 0):
        print(len(pathlens))
    for i in range(len(pathlens)):
        reciprocaltotal = 0
        for j in range(len(pathlens)):
            if pathlens[i, j] != INF and i != j:
                reciprocaltotal += 1 / pathlens[i, j]
                # else add 0
        closenesslist.append(reciprocaltotal)
    return closenesslist


def getASPL(pathlens):
    print("pathlens", pathlens)
    numpairs = 0
    sumpaths = 0
    for i in range(len(pathlens)):
        for j in range(i + 1, len(pathlens)):
            numpairs += 1
            sumpaths += pathlens[i][j]
    if not numpairs:
        return 0
    return sumpaths / numpairs


def analyzematrices(pathlens, numpaths):
    # inputs are the nodelists for each component, shortest paths, and number of shortest paths
    closenesslist = getclosenessreciprocal(pathlens)
    writer.writerow(closenesslist)
    # print(closenesslist)


def floydWarshall(G, nodes):
    nodesets = DisjointSet(len(nodes))
    # length of shortest path
    pathlens = np.full((V, V), INF)
    # number of shortest paths
    numpaths = np.full((V, V), INF)
    for i in range(V):
        pathlens[i, i] = 0
        numpaths[i, i] = 1
    for node1, node2 in G.edges():
        i = nodes.index(node1)
        j = nodes.index(node2)
        pathlens[i, j] = 1
        pathlens[j, i] = 1
        numpaths[i, j] = 1
        numpaths[j, i] = 1

    for k in range(len(nodes)):
        for node in G.neighbors(nodes[k]):
            nodesets.union(k, nodes.index(node))
        candidates = nodesets.sets[nodesets.find(k)]
        # update path lengths, number of paths
        for i in candidates:
            for j in candidates:
                newlen = pathlens[i, k] + pathlens[k, j]
                if (k != i) and (k != j) and (pathlens[i, j] == newlen) and (newlen != INF):
                    numpaths[i, j] += numpaths[i, k] * numpaths[k, j]
                elif pathlens[i, j] > newlen:
                    pathlens[i, j] = newlen
                    numpaths[i, j] = numpaths[i, k] * numpaths[k, j]
        # do stuff with matrices
        analyzematrices(pathlens[np.ix_(range(k + 1), range(k + 1))], numpaths[np.ix_(range(k + 1), range(k + 1))])
    return numpaths, pathlens

def get_freq_dict():
    '''Returns a dictionary from a csv where key = pronunciation, value = frequency'''
     # import data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/Freq_Lexicons.csv'),
                    header=0, delimiter=",", quoting=3)
    # last line with pronunciation info
    max_index = 40411
    # delete words with no pronunciation info
    data = data[:max_index]
    return lexicon_generator.get_freq_dict(HomophoneStrat.addprobs.value, data, data['Log_Freq_SUBTLEX'])




if __name__ == "__main__":
    G = nx.read_gpickle("graph_aoa" + ".gpickle")
    pkl_file = open("aoa_nodes_dict.pickle", 'rb')
    node_dict = pickle.load(pkl_file)
    nodes = list(node_dict.keys())
    
#    graphs that have already been run and saved:
#    plot_diameter_over_time(G, node_dict)
#    plot_average_word_length(nodes)
#    floydWarshall(G,nodes)
#    power_law(G, nodes)
#    analyze_predictiveness(G, nodes)
    
#    sort by aoa and run graphs
#    nodes.sort(key=lambda node: node_dict[node])
#    plot_average_centralities(G, nodes, "aoa")
#    plot_centrality_error(G, nodes, "aoa")
#    plot_centrality_bins(G, nodes, "aoa", "closeness", 3000)
    
#    sort by frequency and run graphs
#    freq_dict = get_freq_dict()
#    nodes.sort(key=lambda node: freq_dict[node], reverse=True)
#    plot_centrality_error(G, nodes, "frequency")
#    plot_centrality_bins(G, nodes, "frequency", "closeness", 3000)
#    plot_centrality_bins(G, nodes, "frequency", "eigenvector", 500)

#    sort by random and run graphs
#    random.shuffle(nodes)
#    plot_centrality_error(G, nodes, "random")
#    plot_centrality_bins(G, nodes, "random", "closeness", 3000)    
#    plot_average_centralities_comparison(G, nodes)
#    plot_centrality_error_comparison(G, nodes)
#    plot_closeness_comparison(G, nodes, 3000)
    plot_error_comparison_ordered_by_aoa(G, nodes, 3000)