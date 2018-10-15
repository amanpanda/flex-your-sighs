import comparator, analyzer, graph_generator
import numpy as np
from lexicon_generator import HomophoneStrat, LexModMethod
import networkx as nx
import pickle


def gen_and_compare_two_graphs(lex_method, strat_parameter, homophone_strat):
    """ Generates and compares two graphs with the given lexicon generation
    method, parameter, and homophone strategy. Prints comparison information.
    
    strat_parameter: parameter required for given method
    - threshold: threshold frequency
    - random: proportion of words to removed
    - freqprob, forgetting: number of word instances
    """
    print("Comparing two graphs for method %s, strat_parameter %f, homophone_strat %s"
          % (LexModMethod(lex_method).name, strat_parameter, HomophoneStrat(homophone_strat).name))
    # Generates and writes graph pickles
    print("Generating graphs...")
    graphs = graph_generator.get_graphs(lex_method, strat_parameter, homophone_strat, 2)
    print("Comparing graphs...")
    # Prints comparison information for the two graphs
    comparator.compare_two_graphs(graphs[0], graph[1])

def gen_data(lex_method, strat_parameter, homophone_strat, num_graphs):
    """ Generates a specified number of graphs with the given lexicon generation
    method, parameter, and homophone strategy. Stores centrality measures for each.
    
    strat_parameter: parameter required for given method
    - threshold: threshold frequency
    - random: proportion of words to removed
    - freqprob, forgetting: number of word instances
    """
    print("Generating graphs...")
    graphs = graph_generator.get_graphs(lex_method, strat_parameter, homophone_strat, num_graphs)
    print("Comparing graphs...")
    comparator.store_data(graphs)
    comparator.plot_stored_data_distributions(num_graphs)


gen_data(LexModMethod.freqprob.value, 1000000, HomophoneStrat.addprobs.value, 200)

# Before comparing subgraphs, we must build centrality measures of the
# master graph we're using and pickle it. This is used in the compare function in the
# comparator.

#G = nx.read_gpickle("graph1.gpickle")
#graph_generator.construct_master_centralities(G, 1)

## For each lexicon method:
#for lex_method in range(5):
#    # for each homophone strategy:
#    for homophone_strat in range(3):
#        # test graph generation and comparison for a range of values:
#        if lex_method == LexModMethod.same.value:
#            strat_parameter = 0
#            gen_and_compare(lex_method, strat_parameter, homophone_strat)
#        elif lex_method == LexModMethod.threshold.value:
#            for strat_parameter in range(10):
#                gen_and_compare(lex_method, strat_parameter, homophone_strat)
#        elif lex_method == LexModMethod.random.value:
#            for strat_parameter in np.arange(0.0,1.0,0.1):
#                gen_and_compare(lex_method, strat_parameter, homophone_strat)
#        elif lex_method == LexModMethod.freqprob.value or lex_method == LexModMethod.forgetting.value:
#            for strat_parameter in range(100000, 1000000, 100000):
#                gen_and_compare(lex_method, strat_parameter, homophone_strat)
