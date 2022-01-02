import numpy as np
import pandas as pd

class Heuristic:
    """This class contains some heuristics algorithms for a graph recommendation system."""
    def __init__(self, graph):
        self.graph = graph


    def __col_fil_movies(self, user, item):
        """Calculates possible rating using collaborative filtering technique."""
        k = 0
        sim_r = 0
        for edge in self.graph.edges([user['label']], data=True):
            if self.graph.nodes[edge[0]]['type'] == 'movie':
                movie = self.graph.nodes[edge[0]]
            elif self.graph.nodes[edge[1]]['type'] == 'movie':
                movie = self.graph.nodes[edge[1]]
            else:
                continue
            data = self.graph.get_edge_data(movie['label'], item['label'])
            print(data)
            if data is None:
                continue
            print(edge, data)
            k += data['similarity']
            sim_r += data['similarity'] * edge[2]['rating']
        return (1 / k) * sim_r if k else 0


    def __col_fil_users(self, user, item):
        """Calculates possible rating using collaborative filtering technique."""
        k = 0
        sim_r = 0
        for edge in self.graph.edges([item['label']], data=True):
            if self.graph.nodes[edge[0]]['type'] == 'user':
                user_1 = self.graph.nodes[edge[0]]
            elif self.graph.nodes[edge[1]]['type'] == 'user':
                user_1 = self.graph.nodes[edge[1]]
            else:
                continue
            data = self.graph.get_edge_data(user_1['label'], user['label'])
            print(data)
            if data is None:
                continue
            print(edge, data)
            k += data['similarity']
            sim_r += data['similarity'] * edge[2]['rating']
        return (1 / k) * sim_r if k else 0




    def collaborative_filtering(self, edges, mode):
        """Movie collaborative filtering algorithm"""
        r = []
        for edge in edges:
            if self.graph.nodes[edge[0]]['type'] == 'user':
                user = self.graph.nodes[edge[0]]
                movie = self.graph.nodes[edge[1]]
            else:
                user = self.graph.nodes[edge[1]]
                movie = self.graph.nodes[edge[0]]
            if mode == 'movies':
                r.append(self.__col_fil_movies(user, movie))
            else:
                r.append(self.__col_fil_users(user, movie))
        return np.array(r)
