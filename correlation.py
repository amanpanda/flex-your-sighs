from numpy.linalg import norm
import numpy as np
from numpy import array

"""Below are various functions to calculate the similarity of two lists
(lexicons)
"""


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def jaccard_similarity(a, b):
    return len(list(set(a) & set(b))) / len(list(set(a) | set(b)))


def euclidean_similarity(a, b):
    return 1 / (1 + norm(array(a) - array(b)))


def minkowski_similarity(a, b):
    return 1 / (1 + np.sum(np.abs(array(a) - array(b))))


def pearson_correlation_coefficient(a, b):
    return np.corrcoef(array(a), array(b))
