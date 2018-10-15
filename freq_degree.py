import networkx as nx
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter

max_index = 40411

G = nx.read_gpickle("graph0.gpickle")

deg = G.degree()

deg_dict = {}
for k in deg:
    deg_dict[k[0]] = k[1]

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/Freq_Lexicons.csv'),
                   header=0, delimiter=",", quoting=3)
data = data[:max_index]

pron = list(set(data['Pron']))

freq_dic = {}
for i in range(len(data)):
    log_freq = data['Log_Freq_SUBTLEX'][i]
    if not math.isnan(log_freq):
        # log_freq = round(math.pow(10, log_freq))
        pass
    else:
        log_freq = 0
    if data['Pron'][i] in freq_dic:
        freq_dic[data['Pron'][i]] = max(log_freq, freq_dic[data['Pron'][i]])
    else:
        freq_dic[data['Pron'][i]] = log_freq
deg_list = []
freq_list = []
for key in freq_dic:
    deg_list.append(deg[key])
    freq_list.append(freq_dic[key])
slope, intercept, r_value, p_value, std_err = stats.linregress(freq_list, deg_list)
print("r-squared:", r_value ** 2)
print("p:", p_value)
plt.plot(freq_list, deg_list, 'ro', markersize=1)
plt.xlabel('log(frequency)')
plt.ylabel('degree')
plt.show()

deg = sorted(deg, key=itemgetter(1), reverse=True)
average = []
sum = 0
count = 0
for i in range(1, len(deg)):
    sum += freq_dic[deg[i][0]]
    count += 1
    average.append(sum / count)
plt.plot(average)
plt.xlabel('top n degree words')
plt.ylabel('average frequency')
plt.show()
