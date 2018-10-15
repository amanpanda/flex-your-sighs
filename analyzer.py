import networkx as nx
from betweeness import *

def get_centrality_measures(G):
    """Returns a list of centrality measures for a graph G
    """
    degree = nx.degree_centrality(G)
    betweenness, shortest_paths, largest_component = betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)
    transitivity = nx.transitivity(G)
    cluster_coefficient = nx.clustering(G)
    closeness = closeness_centrality(nx.number_of_nodes(G), shortest_paths)
    all_measures = {}
    all_measures['max component size'] = largest_component
    all_measures['degree centrality'] = degree
    all_measures['betweenness centrality'] = betweenness
    all_measures['eigenvector centrality'] = eigenvector
    all_measures['transitivity'] = transitivity
    all_measures['clustering coefficient'] = cluster_coefficient
    all_measures['closeness'] = closeness
    return all_measures
