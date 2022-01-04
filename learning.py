import networkx as nx
import numpy as np
from embedding import Embedding
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures


class Learning:
    """The class contains some learning based algorithms."""

    def __init__(self, graph, model='svm-rbf', embedding=Embedding('tf')):
        self.recommended_nodes = None
        self.graph = graph
        self.embedding = embedding
        self.clustering = nx.clustering(graph)
        self.degree_centrality = nx.degree_centrality(graph)
        self.closeness_centrality = nx.closeness_centrality(graph)
        self.betweenness_centrality = nx.betweenness_centrality(graph)
        self.__select_model(model)
        self.x, self.y = self.get_x_y(self.graph.edges(data=True))

    def __select_model(self, model):
        """Model initialization from the given parameter."""
        # self.m = model
        if model == 'svm-rbf':
            self.model = SVR(kernel='rbf')
        elif model == 'svm-linear':
            self.model = SVR(kernel='linear')
        elif model == 'tree':
            self.model = DecisionTreeRegressor(random_state=0)
        elif model == 'linear':
            self.model = LinearRegression()
        elif model == 'forest':
            self.model = RandomForestRegressor(n_estimators=10, random_state=0)
        elif model == 'knn':
            self.model = KNeighborsRegressor()

    def get_x_y(self, edges):
        """The method extracts the features from two nodes (user-movie) that connect with a rating edge.
        Returns the features for each graph, the corresponding ratings and movie titles."""
        x = []
        y = []
        movies_title = {}
        for edge in edges:
            # print(edge)
            if 'similarity' in edge[2]:
                continue
            if self.graph.nodes[edge[0]]['type'] == 'user':
                user = self.graph.nodes[edge[0]]
                movie = self.graph.nodes[edge[1]]
            else:
                user = self.graph.nodes[edge[1]]
                movie = self.graph.nodes[edge[0]]
            movies_title[movie['movieId']] = movie['label']
            features = [user['userId'], float(movie['movieId']),
                        self.clustering[str(user['userId'])], self.degree_centrality[str(user['userId'])],
                        self.closeness_centrality[str(user['userId'])],
                        self.betweenness_centrality[str(user['userId'])],
                        self.clustering[movie['label']], self.degree_centrality[movie['label']],
                        self.closeness_centrality[movie['label']], self.betweenness_centrality[movie['label']]]
            for genre in self.embedding.transform(movie['genres']):
                # print(genre)
                features.append(float(genre))
            # print(features)
            x.append(features)  # [useId, movieId, genre1, genre2, ...]
            y.append(edge[2]['rating'])
        return np.array(x), np.array(y)

    def fit(self):
        """The method fits the SVM model with the extracted features and the corresponding ratings."""
        self.model.fit(self.x, self.y)

    def predict(self, x):
        """Returns the predicts the possible rating for a potential edge between a user and a movie."""
        pred = self.model.predict(x)
        return pred
