import matplotlib.pyplot as plt
import csv
import numpy as np

reader = csv.reader(open("test.csv"))
avg_closeness = []
for row in reader:
    if len(row) != 0:
        sum = 0
        for elt in row:
            sum += float(elt)
        avg_closeness.append(sum/len(row))
plt.clf()
plt.plot(avg_closeness, label='average closeness centrality')
plt.legend()
plt.show()