import pandas as pd
import networkx as nx


class Dataset:
    def __init__(self, directory):
        print('Loading local database...')
        self.links = pd.read_csv(directory+'links.csv', sep=',', encoding='utf-8')
        self.movies = pd.read_csv(directory+'movies.csv', sep=',', encoding='utf-8')
        self.ratings = pd.read_csv(directory+'ratings.csv', sep=',', encoding='utf-8')
        self.tags = pd.read_csv(directory+'tags.csv', sep=',', encoding='utf-8')
        print('Database loaded')

    def getLinks(self):
        return self.links

    def getMovies(self):
        return self.movies

    def getRatings(self):
        return self.ratings

    def getTags(self):
        return self.tags

    @staticmethod
    def splitTrainTest():
        pass


class Similarity:
    def __init__(self):
        pass


class Embeddings:
    def __init__(self):
        pass


class Metrics:
    def __init__(self):
        pass


class Graph:
    def __init__(self, dataset_directory):
        self.dataset = Dataset(dataset_directory)
        self.graph = nx.Graph()
        self.__initGraph()

    def __initGraph(self):
        print('Graph initialing...')
        movies = self.dataset.getMovies()
        for index, value in self.dataset.getRatings().iterrows():
            self.graph.add_node(value['userId'], type='user')
            movie = movies.loc[movies['movieId'] == value['movieId']].iloc[0]
            self.graph.add_node(movie['title'], type='movie')
            self.graph.add_edge(value['userId'], movie['title'], rating=value['rating'])
        print('Graph created')
        print(f'Bipartite graph: {nx.bipartite.is_bipartite(self.graph)}')
        print(f'Total nodes: {nx.number_of_nodes(self.graph)}')
        print(f'Total edges: {nx.number_of_edges(self.graph)}')


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
    graph = Graph(dataset_directory='data/dataset/')
