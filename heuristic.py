import numpy as np


class Heuristic:
    """This class contains some heuristics algorithms for a graph recommendation system."""

    def __init__(self, graph):
        self.graph = graph

    def __col_fil(self, user, item):
        """Calculates possible rating using collaborative filtering technique."""
        k = 0
        sim_r = 0
        for edge in self.graph.edges([user['label']], data=True):
            if self.graph.nodes[edge[0]]['type'] == 'user':
                movie = self.graph.nodes[edge[1]]
            else:
                movie = self.graph.nodes[edge[0]]
            data = self.graph.get_edge_data(item['label'], movie['label'])
            # print(data)
            if data is None:
                continue
            # print(edge, data)
            k += data['similarity']
            sim_r += data['similarity'] * edge[2]['rating']
        return (1 / k) * sim_r if k else 0

    def collaborative_filtering(self, edges, mode='movies'):
        """Movie collaborative filtering algorithm"""
        r = []
        for edge in edges:
            if self.graph.nodes[edge[0]]['type'] == 'user':
                user = self.graph.nodes[edge[0]]
                movie = self.graph.nodes[edge[1]]
            else:
                user = self.graph.nodes[edge[1]]
                movie = self.graph.nodes[edge[0]]
            r.append(self.__col_fil(user, movie))
        return np.array(r)
