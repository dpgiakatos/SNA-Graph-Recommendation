import random
import networkx as nx
from embedding import Embedding
from similarity import Similarity
from dataset import Dataset


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
            self.graph.add_node(movie['title'], type='movie', genres=movie['genres'], movieId=movie['movieId'])

    def __create_movies_graph(self, movies, embedding):
        for index_i, movie_i in movies.iterrows():
            index_j = index_i
            self.graph.add_node(movie_i['title'], type='movie', genres=movie_i['genres'], movieId=movie_i['movieId'])
            for _, movie_j in movies.iterrows():
                if index_i != index_j:
                    self.graph.add_node(movie_j['title'], type='movie', genres=movie_j['genres'], movieId=movie_j['movieId'])
                    similarity = Similarity.cosine(embedding.transform(movie_i['genres']), embedding.transform(movie_j['genres']))
                    if similarity > 0.8:
                        self.graph.add_edge(movie_i['title'], movie_j['title'], similarity=similarity)
                index_j += 1

    def init(self, dataset_directory, keep_only_good_ratings=False, bipartite=False, embedding=Embedding('tf')):
        dataset = Dataset(dataset_directory)
        print('Graph initialing...')
        movies = dataset.get_movies()
        tags = dataset.get_tags()
        if bipartite:
            self.__insert_movies_nodes(movies.head(500))
        else:
            self.__create_movies_graph(movies.head(100), embedding)
        for index, value in dataset.get_ratings().iterrows():
            movie = movies.loc[movies['movieId'] == value['movieId']].iloc[0]
            if movie['title'] not in self.graph:
                continue
            if keep_only_good_ratings and value['rating'] < 3.5:
                continue
            self.graph.add_node(value['userId'], type='user', userId=value['userId'])
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
        edges = list(self.graph.edges(data=True))
        random.shuffle(edges)
        for edge in edges:
            if 'similarity' in edge[2]:
                continue
            if len(test) > test_size:
                break
            if edge[2]['rating'] < 3.5:
                continue
            graph_copy.remove_edge(*edge[:2])
            # print(nx.number_connected_components(graph_copy))
            if nx.number_connected_components(graph_copy) == 1:
                train_graph.remove_edge(*edge[:2])
                test.append(edge)
            graph_copy.add_edge(edge[0], edge[1], **edge[2])
        print('Splitting completed')
        return train_graph, test
