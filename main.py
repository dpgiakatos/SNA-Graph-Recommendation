import pandas as pd
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset:
    def __init__(self, directory):
        print('Loading local database...')
        self.links = pd.read_csv(directory + 'links.csv', sep=',', encoding='utf-8')
        self.movies = pd.read_csv(directory + 'movies.csv', sep=',', encoding='utf-8')
        self.ratings = pd.read_csv(directory + 'ratings.csv', sep=',', encoding='utf-8')
        self.tags = pd.read_csv(directory + 'tags.csv', sep=',', encoding='utf-8')
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

    @staticmethod
    def cosine(a, b):
        return np.sqrt(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


class Embedding:
    def __init__(self, method='tf'):
        self.corpus = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                       'War', 'Western', '(no genres listed)']
        if method == 'tf-idf':
            self.tfidf = TfidfVectorizer()
            self.tfidf.fit(self.corpus)

    def get_tf_idf(self, document):
        return self.tfidf.transform([document])

    def get_tf(self, document):
        return [1 if value in document else 0 for value in self.corpus]


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

    def __clean_graph(self):
        print('Checking if graph contains isolated nodes...')
        if nx.number_connected_components(self.graph) > 1:
            pass

    def __insert_movies_nodes(self, movies):
        for index, movie in movies.iterrows():
            self.graph.add_node(movie['title'], type='movie', genres=movie['genres'])

    def __create_movies_graph(self, movies):
        embedding = Embedding(method='tf')
        for index_i, movie_i in movies.iterrows():
            index_j = index_i
            self.graph.add_node(movie_i['title'], type='movie', genres=movie_i['genres'])
            for _, movie_j in movies.iterrows():
                if index_i != index_j:
                    self.graph.add_node(movie_j['title'], type='movie', genres=movie_j['genres'])
                    similarity = Similarity.cosine(embedding.get_tf(movie_i['genres']), embedding.get_tf(movie_j['genres']))
                    if similarity > 0.8:
                        self.graph.add_edge(movie_i['title'], movie_j['title'], similarity=similarity)
                index_j += 1

    def init(self, dataset_directory, keep_only_good_ratings=False, bipartite=False):
        dataset = Dataset(dataset_directory)
        print('Graph initialing...')
        movies = dataset.get_movies()
        tags = dataset.get_tags()
        if bipartite:
            self.__insert_movies_nodes(movies.head(500))
        else:
            self.__create_movies_graph(movies.head(500))
        for index, value in dataset.get_ratings().iterrows():
            movie = movies.loc[movies['movieId'] == value['movieId']].iloc[0]
            if movie['title'] not in self.graph:
                continue
            if keep_only_good_ratings and value['rating'] < 3.5:
                continue
            self.graph.add_node(value['userId'], type='user')
            # self.graph.add_node(movie['title'], type='movie', genres=movie['genres'])
            tag = tags.loc[(tags['userId'] == value['userId']) & (tags['movieId'] == value['movieId'])]
            tag_value = '' if tag.empty else tag.iloc[0]['tag']
            self.graph.add_edge(value['userId'], movie['title'], rating=value['rating'], tag=tag_value)
        print('Graph created')
        self.__print_graph_info()

    def export_gexf(self, directory):
        nx.write_gexf(self.graph, directory + 'graph.gexf')

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
        print('Splitting started. This may take a while. Please wait...')
        graph_copy = self.graph.copy()
        train_graph = self.graph.copy()
        test = []
        test_size = nx.number_of_edges(self.graph) * split
        for edge in self.graph.edges(data=True):
            if len(test) > test_size:
                break
            graph_copy.remove_edge(*edge[:2])
            # print(nx.number_connected_components(graph_copy))
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
    graph.init(dataset_directory='data/dataset/', keep_only_good_ratings=True, bipartite=False)
    graph.export_gexf(directory='data/graph/')
    # graph.read_gexf('data/graph/graph.gexf')
    graph_train, test_edges = graph.split_train_test()
    print(len(test_edges))
    # embedding = Embedding('tf')
    # test1 = embedding.get_tf('Animation|War|Thriller')
    # print(test1)
    # test2 = embedding.get_tf('War|Thriller')
    # print(test2)
    # sim = Similarity.cosine(test1, test2)
    # print(sim)
