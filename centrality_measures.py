import networkx as nx
import itertools


def compute_degree_centrality(G, index, centrality):
    for j in G[index].keys():
        centrality[j] -= 1
    centrality.pop(index)
    return centrality


def get_all_pairs(list):
    l = []
    for i in itertools.combinations(list, 2):
        l.append(i)
    return l


def compute_clustering(G, index, clustering):
    neighbors = G[index].keys()
    neighbor_dic = {}
    for i in neighbors:
        neighbor_dic[i] = clustering[i] * G.degree(i) * (G.degree(i) - 1) / 2
    neighbor_pairs = get_all_pairs(neighbors)
    for j in neighbor_pairs:
        if G.has_edge(j[0], j[1]):
            neighbor_dic[j[0]] -= 1
            neighbor_dic[j[1]] -= 1
    for j in neighbors:
        degree = G.degree(j) - 1
        if degree <= 1:
            clustering[j] = 0
        else:
            clustering[j] = 2 * neighbor_dic[j] / (degree * (degree - 1))
    clustering.pop(index)
    return clustering
