from analyzer import get_centrality_measures
from scipy import stats
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import csv

# This unnormalizes the degree centralities in our frequency probability graphs
def unnormalize_degree_centrality():
	#for i in range(0, 200):
	#print("Computing for graph: ", i)
	#pkl_file = open("freq_prob_graph_data/freq_graph_" + str(i) + ".pickle", 'rb')
	pkl_file = open("master_centralities.pickle", "rb")
	freq_centralities = pickle.load(pkl_file)
	pkl_file.close()
	degree_centrality_dict = freq_centralities['degree centrality']
	#degree_centrality_dict = freq_centralities[1]['degree centrality']
	unnormalize_factor = len(degree_centrality_dict) - 1
	for key, value in degree_centrality_dict.items():
		degree_centrality_dict[key] = value * unnormalize_factor
	with open("master_centralities_unnormalized.pickle", "wb") as f:
		#with open("freq_prob_graph_data/freq_graph_" + str(i) + ".pickle", 'wb') as f:
		pickle.dump(freq_centralities, f)
		f.close()

#This computes mean, median, and standard deviation for the centrality measures
def calc_centrality_stats():
	all_centralities_dict = {'degree centrality': {},
							 'betweenness centrality': {},
							 'eigenvector centrality': {},
							 'transitivity': [],
							 'clustering coefficient':{},
							 'closeness': {},
							 'max component size': []}
	#for i in range(0, 200):
	#print("Computing for graph ", i)
	#pkl_file = open("freq_prob_graph_data/freq_graph_" + str(i) + ".pickle", 'rb')
	pkl_file = open("master_centralities_unnormalized.pickle", 'rb')
	freq_centralities = pickle.load(pkl_file)
	#for centrality_type, centralities in freq_centralities[1].items():
	for centrality_type, centralities in freq_centralities.items():
		if centrality_type != 'max component size' and centrality_type != 'transitivity':
			for word, measure in centralities.items():
				all_centralities_dict[centrality_type].setdefault(word, []).append(measure)
		else:
			all_centralities_dict[centrality_type].append(centralities)
	for centrality_type, centralities in all_centralities_dict.items():
		print("Computing statistics for ", centrality_type)
		if centrality_type != 'max component size' and centrality_type != 'transitivity':
			for word, measure in centralities.items():
				all_centralities_dict[centrality_type][word] = (np.mean(measure), 
																np.median(measure), 
																np.std(measure))
		else:
			all_centralities_dict[centrality_type] = (np.mean(all_centralities_dict[centrality_type]), 
													  np.median(all_centralities_dict[centrality_type]), 
													  np.std(all_centralities_dict[centrality_type]))
	with open("master_dsn_stats.pickle", 'wb') as f:
	 	pickle.dump(all_centralities_dict, f)
	 	f.close()
	return all_centralities_dict

def group_centralities(all_centralities):
	grouped_centralities = {'degree centrality': [],
							'betweenness centrality': [],
							'eigenvector centrality': [],
							'transitivity': [],
							'clustering coefficient':[],
							'closeness': [],
							'max component size': []}
	for centrality_type, centralities in all_centralities_dict.items():
		if centrality_type != 'max component size' and centrality_type != 'transitivity':
			for word, measure in centralities.items():
				grouped_centralities[centrality_type].append(measure[0])
		else:
			grouped_centralities[centrality_type].append(centralities[0])
	return grouped_centralities

def plot_distribution(grouped_centralities, measure):
	plt.clf()
	plt.hist(grouped_centralities[measure],bins=45)
	plt.xlabel(measure)
	plt.ylabel("Number of Nodes")
	plt.show()

def output_to_csv(centrality_list):
	csvfile = 'output.csv'
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		for val in centrality_list:
			writer.writerow([val]) 

# calc_centrality_stats()
# file = open('freq_prob_dsn_stats.pickle','rb')
# all_centralities_dict = pickle.load(file)
# grouped_centralities = group_centralities(all_centralities_dict)
# plot_distribution()

#unnormalize_degree_centrality()
#calc_centrality_stats()
file = open('master_dsn_stats.pickle','rb')
all_centralities_dict = pickle.load(file)
grouped = group_centralities(all_centralities_dict)
plot_distribution(grouped, "degree centrality")


