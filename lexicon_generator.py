from random import sample
from numpy.random import choice
from numpy.random import random
import math
from enum import Enum
import pandas as pd
import os

MAX_CHOICE_SIZE = 100000000


# Approaches for homophones
class HomophoneStrat(Enum):
    # max probability between homophones
    maxprob = 0
    # add probabilities of homophones
    addprobs = 1
    # treat homophones as competitors
    compete = 2


# Methods of selecting words in a sublexicon    
class LexModMethod(Enum):
    # graph doesn't change
    same = 0
    # all nodes below given threshold frequency are removed
    threshold = 1
    # given proportion of words are removed (uniformly random)
    random = 2
    # words chosen using probability of having been seen given num word instances
    freqprob = 3
    # words chosen using probability of encountering and remembering gien num word instances
    forgetting = 4


def lexicon_by_threshold(lexicon, freq_dict, threshold):
    """Removes all words in the lexicon with a frequency lower
    than 'threshold', returns the new lexicon
    """
    l = []
    for word in lexicon:
        if freq_dict[word] >= threshold:
            l.append(word)
    return l


def remove_by_threshold(lexicon, freq_dict, threshold):
    """Removes all words in the lexicon with a frequency lower
    than 'threshold', returns the words to be removed from the
    full lexicon to create the new lexicon
    """
    l = []
    for word in lexicon:
        if freq_dict[word] <= threshold:
            l.append(word)
    return l


def lexicon_by_random(pron, percentage):
    """Remove percentage*100% of the given lexicon pseudorandomly,
    returns the new lexicon
    """
    pron = sample(pron, round(len(pron) * (1 - percentage)))
    return pron


def remove_by_random(pron, percentage):
    """Remove percentage*100% of the given lexicon pseudorandomly,
    returns the list of words to remove from the full lexicon to
    create the new lexicon
    """
    pron = sample(pron, round(len(pron) * (percentage)))
    return pron


def lexicon_by_freqprob(lexicon, freq_dict, word_insts):
    """Generates a lexicon by randomly drawing word_insts words from
    the lexicon with replacement with weights directly proportional to their
    frequencies, and then removing duplicates. Loop is there to prevent
    memory error b/c of too large value given to numpy.random.choice()
    """
    probabilities = []
    total_freq = sum(freq_dict.values())
    for word in lexicon:
        word_prob = freq_dict[word] / total_freq
        probabilities.append(word_prob)
    newlex = set()
    while word_insts > 0:
        if word_insts < MAX_CHOICE_SIZE:
            words_seen = choice(lexicon, word_insts, replace=True, p=probabilities)
            word_insts = 0
        else:
            words_seen = choice(lexicon, MAX_CHOICE_SIZE, replace=True, p=probabilities)
            word_insts = word_insts - MAX_CHOICE_SIZE
        newlex.update(words_seen)
    newlex = list(newlex)
    return newlex


def remove_by_freqprob(lexicon, freq_dict, word_insts):
    """Does the same thing as lexicon_by_freqprob except returns the words
    to remove from the full lexicon as opposed to the words to include
    """
    newlex = lexicon_by_freqprob(lexicon, freq_dict, word_insts)
    for item in newlex:
        lexicon.remove(item)
    return lexicon


def lexicon_by_forget(lexicon, freq_dict, word_insts, memory_stability):
    """Generates a lexicon by including each word with the probability
    that someone who encounters and remembers.
    Memory stability meaning:
        One unit of time is used to encounter word_insts number of words
        Memory stability is the amount of time needed to forget about 2 / 3 (63.2%) of the material
        This parameter might be subject to change.
    """
    newlex = []
    total_freq = sum(freq_dict.values())
    for word in lexicon:
        word_prob = freq_dict[word] / total_freq
        if word_prob != 0:
            # Poisson process, expected number of events
            lmb = word_insts * word_prob + (random() * 2 - 1) * math.sqrt(word_insts * word_prob)
            # time from last refresh
            last_refresh = 1 / lmb + (random() * 2 - 1) * (2 / (l * l))
            # forget curve
            prob_remem = math.exp(-last_refresh / memory_stability)
        else:
            prob_remem = 0
        if random() <= prob_remem:
            newlex.append(word)
    return newlex


def remove_by_forget(lexicon, freq_dict, word_insts, memory_stability):
    """Generates a lexicon by removing each word with the probability
    that someone who did not both encounter and remember.
    """
    removelist = []
    total_freq = sum(freq_dict.values())
    for word in lexicon:
        word_prob = freq_dict[word] / total_freq
        if word_prob != 0:
            # Poisson process, expected number of events
            lmb = max(word_insts * word_prob + (random() * 2 - 1) * math.sqrt(word_insts * word_prob), 0)
            if lmb != 0:
                # time from last refresh
                last_refresh = 1 / lmb
                # forget curve.
                prob_remem = math.exp(-last_refresh / memory_stability)
            else:
                prob_remem = 0
        else:
            prob_remem = 0
        if random() >= prob_remem:
            removelist.append(word)
    return removelist


def expected_overlap(lexicon, freq_dict, word_insts, homophone_strat):
    # expectation of sum of prob = sum of expectation of prob
    total_freq = sum(freq_dict.values())
    expected_overlap = 0
    # for word in freq_dict:
    for word in lexicon:
        # pr = prob(w drawn once, its freq prob)
        word_prob = freq_dict[word] / total_freq
        if word_prob != 0:
            # prob(w not drawn once) = 1 - pr
            # prob(w not in lexicon 1) = prob(w not drawn once)^word_insts
            # prob(w in lexicon 1) = 1 - prob(w not in lexicon 1)
            prob_in_lexicon = 1 - ((1 - word_prob) ** word_insts)

            # prob(w in lexicon 1 and lexicon 2) = prob(w in lexicon 1)^2
            prob_in_both = prob_in_lexicon ** 2
            # prob(w in lexicon 1 or lexicon 2) = 1 - prob(w not in lexicon 1)^2
            prob_in_either = 1 - ((1 - prob_in_lexicon) ** 2)

            if prob_in_either == 0:
                print(prob_in_lexicon, prob_in_both, prob_in_either)
                return 0

            # expected probability of overlap = prob(w in both) / prob(w in either)
            # sum the above value over all words to get expected overlap size!
            expected = (prob_in_both)
        else:
            expected = 0
        expected_overlap += expected
    if homophone_strat == 0:
        homophone = "use max frequency"
    elif homophone_strat == 1:
        homophone = "use sum frequency"
    else:
        homophone = "treat as competitor"
    print("Expected overlap using homophone strategy -", homophone, ":", expected_overlap)


def get_freq_dict(homophone_strat, data, frequencies):
    # calculate frequency only based on SUBTLEX
    freq_dict = {}
    for i in range(len(data)):
        log_freq = frequencies[i]
        if not math.isnan(log_freq):
            log_freq = round(math.pow(10, log_freq))
        else:
            log_freq = 0
        if data['Pron'][i] in freq_dict:
            if homophone_strat == HomophoneStrat.maxprob.value:
                freq_dict[data['Pron'][i]] = max(log_freq, freq_dict[data['Pron'][i]])
            elif homophone_strat == HomophoneStrat.addprobs.value:
                freq_dict[data['Pron'][i]] = log_freq + freq_dict[data['Pron'][i]]
            elif homophone_strat == HomophoneStrat.compete.value:
                homophone_num = 1
                while (data['Pron'][i] + ":" + str(homophone_num)) in freq_dict:
                    homophone_num += 1
                freq_dict[(data['Pron'][i] + ":" + str(homophone_num))] = log_freq
        else:
            freq_dict[data['Pron'][i]] = log_freq
    return freq_dict


def get_lexicon(method, strat_parameter, homophone_strat, data, frequencies):
    """Returns a lexicon from a csv and optionally alters the lexicon
    
    method: value of LexModMethod Enum associated with lexicon modification methods
    (LexModMethod.same, LexModMethod.threshold, LexModMethod.random, LexModMethod.freqprob, LexModMethod.forgetting)
    to get the enum, use LexModMethod(method)
    
    strat_parameter: value associated with method
    - threshold: threshold frequency
    - random: proportion of words to removed
    - freqprob, forgetting: number of word instances
    
    homophone_strat: value of HomophoneStrat Enum associated with homophone strategy
    (HomophoneStrat.maxprob, HomophoneStrat.addprobs, HomophoneStrat.compete)
    to get the enum, use HomophoneStrat(homophone_strat)
    
    Returns a list of all the words in the generated lexicon
    """
    # if an enum is passed in, converts to appropriate int    
    if isinstance(method, Enum):
        method = method.value
    if isinstance(homophone_strat, Enum):
        homophone_strat = method.value

    # gets dictionary where keys: word pronunciations and values: frequencies    
    freq_dict = get_freq_dict(homophone_strat, data, frequencies)

    # delete repeated words if not treating homophones as competitors
    # pron is a list containing all the words in the full lexicon
    if homophone_strat != HomophoneStrat.compete.value:
        pron = list(set(data['Pron']))
    else:
        pron = list(freq_dict.keys())

    if method == LexModMethod.threshold.value:
        pron = remove_by_threshold(pron, freq_dict, strat_parameter)
    elif method == LexModMethod.random.value:
        pron = remove_by_random(pron, strat_parameter)
    elif method == LexModMethod.freqprob.value:
        pron = remove_by_freqprob(pron, freq_dict, strat_parameter)
    elif method == LexModMethod.forgetting.value:
        pron = remove_by_forget(pron, freq_dict, strat_parameter, 1)

    return pron


def get_spanish_data():
    # import data
    freq_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/spanish/nwfreq.txt'),
                            sep="\t", header=None, encoding='latin-1')
    freq_data.columns = ["word", "freq"]
    pron_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/spanish/nwphono.txt'),
                            sep="\t", header=None, encoding='latin-1')
    pron_data = pron_data.iloc[:, 0:2]
    pron_data.columns = ["word", "pron"]
    # get rid of dashes in pron, which are syllable markers
    pron_data['pron'] = pron_data['pron'].str.replace('-', '')
    full_data = pd.merge(freq_data, pron_data, on='word')
    return full_data


def get_spanish_lexicon():
    """Returns a Spanish lexicon, in the same format as the get_lexicon function returns.
    Mostly temporary, need to make the other functions so things aren't hardcoded.
    """
    data = get_spanish_data()
    pron = list(data['pron'])
    # do the homophone stuff

    return pron
