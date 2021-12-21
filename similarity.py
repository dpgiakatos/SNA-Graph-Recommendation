import numpy as np
from scipy.spatial import distance


class Similarity:
    def __init__(self):
        pass

    @staticmethod
    def cosine(a, b):
        return np.sqrt(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def jaccard(a, b):
        nominator = a.intersection(b)
        denominator = a.union(b)
        similarity = len(nominator) / len(denominator)
        return similarity

    @staticmethod
    def euclidean_distance(a, b):
        return distance.euclidean(a, b)

    @staticmethod
    def manhattan_distance(a, b):
        return distance.cityblock(a, b)
