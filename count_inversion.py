import pickle
from analyzer import *
import operator
import matplotlib.pyplot as plt

centrality_names = ['closeness', 'eigenvector centrality', 'degree centrality', 'betweenness centrality']


def count_inversions(lst):
    return merge_count_inversion(lst)[1]


def merge_count_inversion(lst):
    if len(lst) <= 1:
        return lst, 0
    middle = int(len(lst) / 2)
    left, a = merge_count_inversion(lst[:middle])
    right, b = merge_count_inversion(lst[middle:])
    result, c = merge_count_split_inversion(left, right)
    return result, (a + b + c)


def merge_count_split_inversion(left, right):
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]
    return result, count


def count_inversion(l1, l2):
    index_dict = {}
    for i in range(len(l2)):
        index_dict[l2[i]] = i
    l = []
    for i in l1:
        l.append(index_dict[i])
    inv = count_inversions(l)
    inv = 2 * inv / (len(l2) * (len(l2) - 1))
    return inv


def preprocess(centralities):
    ordered = []
    for i in centrality_names:
        ordered_list = sorted(centralities[i].items(), key=operator.itemgetter(1))
        ordered_list = [i[0] for i in ordered_list]
        ordered_list.reverse()
        ordered.append(ordered_list)
    return ordered


# pkl_file = open("graph1.gpickle", 'rb')
# G = pickle.load(pkl_file)
# centralities = get_centrality_measures(G)
# with open("master_centralities"+ ".pickle", 'wb') as f:
#     pickle.dump(centralities, f)

pkl_file = open("master_centralities.pickle", 'rb')
centralities = pickle.load(pkl_file)
ordered_centralities = preprocess(centralities)
closeness = []
eigenvector = []
degree = []
betweenness = []
for i in range(200):
    print(i)
    pkl_file = open("freq_prob_graph_data/freq_graph_" + str(i) + ".pickle", 'rb')
    freq_centralities = pickle.load(pkl_file)[1]
    ordered_list_freq = preprocess(freq_centralities)
    closeness.append(count_inversion(ordered_list_freq[0], ordered_centralities[0]))
    eigenvector.append(count_inversion(ordered_list_freq[1], ordered_centralities[1]))
    degree.append(count_inversion(ordered_list_freq[2], ordered_centralities[2]))
    betweenness.append(count_inversion(ordered_list_freq[3], ordered_centralities[3]))
l = [closeness, eigenvector, degree, betweenness]

for i in range(len(centrality_names)):
    plt.clf()
    plt.hist(l[i], bins='auto')
    plt.title('Distribution of Number of Inversions for ' + centrality_names[i])
    plt.xlabel("Relative Number of Inversion of " + centrality_names[i])
    plt.savefig(centrality_names[i] + '_percentage_inversion.png')
