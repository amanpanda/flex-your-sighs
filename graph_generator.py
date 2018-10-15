import networkx as nx
from lexicon_generator import *
from analyzer import get_centrality_measures
import pickle
import pandas as pd
import os
import time


def isOneEditDistance(s, t):
    """Returns a boolean: true if the two words are one edit distance or less
    away in pronunciation, false if not
    """
    if s == t:
        return True
    l1, l2 = len(s), len(t)
    for i in range(len(s)):
        if s[i] != t[i]:
            if l1 == l2:
                s = s[:i] + t[i] + s[i + 1:]
            else:
                s = s[:i] + t[i] + s[i:]
            break
    return s == t or s == t[:-1]


def get_graphs(method, strat_parameter, homophone_strat, num_graphs):
    """Returns a list of graphs with different type of perturbations
    num_graphs: number of graphs

    method: value of LexModMethod Enum associated with lexicon modification methods
    (LexModMethod.same, LexModMethod.threshold, LexModMethod.random, LexModMethod.freqprob, LexModMethod.forgetting)
    to get the enum, use LexModMethod(method)

    strat_parameter: parameter required for given method
    - threshold: threshold frequency
    - random: proportion of words to removed
    - freqprob, forgetting: number of word instances

    homophone_strat: strat_parameter of HomophoneStrat Enum associated with homophone strategy
    (HomophoneStrat.maxprob, HomophoneStrat.addprobs, HomophoneStrat.compete)
    to get the enum, use HomophoneStrat(homophone_strat)
    
    num_graphs: number of graphs
    """
    G = nx.read_gpickle("graph" + str(homophone_strat) + ".gpickle")
    graphs = []
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/Freq_Lexicons.csv'),
                       header=0, delimiter=",", quoting=3)
    data = data[:40411]
    frequencies = list(data['Log_Freq_SUBTLEX'])
    # shuffle(frequencies)
    start = time.time()
    for i in range(num_graphs):
        last = time.time()
        lexicon_to_remove = get_lexicon(method, strat_parameter, homophone_strat, data, frequencies)
        E = G.copy()
        E.remove_nodes_from(lexicon_to_remove)
        graphs.append(E)
        print("Finished graph", i, ",", time.time()-last, "sec")
    print("Finished", num_graphs, "graphs,", time.time()-start, "sec")
    return graphs

def strip_digits(pron):
    '''Strips digits from pron. Digits appear when homophones are treated as
    competitors to ensure uniqueness of keys is freq_dic'''
    new_list = []
    for i in range(len(pron)):
        # adjusts to get rid of digits (for uniqueness of homophone keys)
        proni = pron[i].split(":")[0]
        new_list.append(proni)
    return new_list


def get_len_strip_digits(key):
    '''returns length of key after homophone competitior digits have been stripped'''
    key = key.split(":")[0]
    return len(key)


def get_complete_graph(homophone_strat):
    """
    Note: this function is currently out of date, functions that it calls have changed.

    Given a homophone strategy, gets the full lexicon, calculates the
    graph associated with the full lexicon, and writes it to a gpickle file

    homophone_strat: value of HomophoneStrat Enum associated with homophone strategy
    (HomophoneStrat.maxprob, HomophoneStrat.addprobs, HomophoneStrat.compete)
    to get the enum, use HomophoneStrat(homophone_strat)
    """
    pron = get_lexicon(0, 0, homophone_strat)
    pron.sort(key=get_len_strip_digits)  # sorts on length after stripping digits
    stripped = strip_digits(pron)
    G = nx.Graph()
    G.add_nodes_from(pron)
    E = []
    for i in range(len(pron)):
        j = i + 1
        if i % 1000 == 0:
            print("Progress", i, i)
        while j < len(pron) and (len(stripped[j]) == len(stripped[i])
                                 or len(stripped[j]) == len(stripped[i]) + 1):
            if isOneEditDistance(stripped[i], stripped[j]):
                E.append((pron[i], pron[j]))
            j += 1
    G.add_edges_from(E)
    # pickles the graph
    nx.write_gpickle(G, "graph" + str(homophone_strat) + ".gpickle")

def construct_master_centralities(G, homophone_strat):
    # pickles the centrality measures of the graph (in a dictionary)
    G_centralities = get_centrality_measures(G)
    output = open('master_centralities'+ str(homophone_strat) +'.pkl', 'wb')
    pickle.dump(G_centralities, output)
    output.close()

def get_spanish_graph(homophone_strat):
    """
    Given a homophone strategy, gets the full Spanish lexicon, calculates the
    graph associated with the full lexicon, and writes it to a gpickle file
    (homophone stuff not implemented because some other stuff is hardcoded for
    the English dataset)

    homophone_strat: value of HomophoneStrat Enum associated with homophone strategy
    (HomophoneStrat.maxprob, HomophoneStrat.addprobs, HomophoneStrat.compete)
    to get the enum, use HomophoneStrat(homophone_strat)
    """
    pron = get_spanish_lexicon()
    pron.sort(key=get_len_strip_digits)  # sorts on length after stripping digits
    stripped = strip_digits(pron)
    G = nx.Graph()
    G.add_nodes_from(pron)
    E = []
    for i in range(len(pron)):
        j = i + 1
        if i % 1000 == 0:
            print("Progress", i)
        while j < len(pron) and (len(stripped[j]) == len(stripped[i])
                                 or len(stripped[j]) == len(stripped[i]) + 1):
            if isOneEditDistance(stripped[i], stripped[j]):
                E.append((pron[i], pron[j]))
            j += 1
    G.add_edges_from(E)
    # pickles the graph
    nx.write_gpickle(G, "span_graph" + str(homophone_strat) + ".gpickle")

# get_spanish_graph('test')
