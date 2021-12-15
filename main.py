import pandas as pd
import networkx as nx
import numpy as np


class Dataset:
    def __init__(self, directory):
        print('Loading local database...')
        self.links = pd.read_csv(directory+'links.csv', sep=',', encoding='utf-8')
        self.movies = pd.read_csv(directory+'movies.csv', sep=',', encoding='utf-8')
        self.ratings = pd.read_csv(directory+'ratings.csv', sep=',', encoding='utf-8')
        self.tags = pd.read_csv(directory+'tags.csv', sep=',', encoding='utf-8')
        print('Database loaded')

    def get_links(self):
        return self.links

    def get_movies(self):
        return self.movies

    def get_ratings(self):
        return self.ratings

    def get_tags(self):
        return self.tags


class Similarity:
    def __init__(self):
        pass


class Embedding:
    def __init__(self):
        pass


class Metric:
    def __init__(self):
        pass


class Graph:
    def __init__(self):
        self.graph = nx.Graph()

    def __print_graph_info(self):
        print(f'Bipartite graph: {nx.bipartite.is_bipartite(self.graph)}')
        print(f'Total nodes: {nx.number_of_nodes(self.graph)}')
        print(f'Total edges: {nx.number_of_edges(self.graph)}')

    def init(self, dataset_directory):
        dataset = Dataset(dataset_directory)
        print('Graph initialing...')
        movies = dataset.get_movies()
        tags = dataset.get_tags()
        for index, value in dataset.get_ratings().iterrows():
            self.graph.add_node(value['userId'], type='user')
            movie = movies.loc[movies['movieId'] == value['movieId']].iloc[0]
            self.graph.add_node(movie['title'], type='movie', genres=movie['genres'].replace('|', ' '))
            tag = tags.loc[(tags['userId'] == value['userId']) & (tags['movieId'] == value['movieId'])]
            tag_value = '' if tag.empty else tag.iloc[0]['tag']
            self.graph.add_edge(value['userId'], movie['title'], rating=value['rating'], tag=tag_value)
        print('Graph created')
        self.__print_graph_info()

    def export_gexf(self, directory):
        nx.write_gexf(self.graph, directory+'graph.gexf')

    def read_gexf(self, file_path):
        print('Graph loading...')
        self.graph = nx.read_gexf(file_path)
        print('Graph loaded')
        self.__print_graph_info()

    def get_graph(self):
        return self.graph

    def get_adjacency_matrix(self):
        adj = nx.adjacency_matrix(self.graph)
        return adj.todense()

    def split_train_test(self, split=0.1):
        print('Splitting started. This may take time. Please wait...')
        graph_copy = self.graph.copy()
        train_graph = self.graph.copy()
        test = []
        test_size = nx.number_of_edges(self.graph) * split
        for edge in self.graph.edges(data=True):
            if len(test) > test_size:
                break
            graph_copy.remove_edge(*edge[:2])
            if nx.number_connected_components(graph_copy) == 1:
                train_graph.remove_edge(*edge[:2])
                test.append(edge)
            graph_copy.add_edge(edge[0], edge[1], **edge[2])
        return train_graph, test


class Heuristic:
    def __init__(self):
        pass


class Learning:
    def __init__(self):
        pass


class Hybrid:
    def __init__(self):
        pass


class Evaluation:
    def __init__(self):
        pass


if __name__ == '__main__':
    graph = Graph()
    # graph.init(dataset_directory='data/dataset/')
    # graph.export_gexf(directory='data/graph/')
    graph.read_gexf('data/graph/graph.gexf')
    graph.split_train_test()
