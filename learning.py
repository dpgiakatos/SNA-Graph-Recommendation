import numpy as np
from embedding import Embedding
from sklearn.svm import SVR


class Learning:
    """The class contains some learning based algorithms."""
    def __init__(self, graph, model='svm-rbf', embedding=Embedding('tf')):
        self.recommended_nodes = None
        self.graph = graph
        self.embedding = embedding
        self.__select_model(model)
        self.x, self.y = self.get_x_y(self.graph.edges(data=True))

    def __select_model(self, model):
        """Model initialization from the given parameter."""
        if model == 'svm-rbf':
            self.model = SVR(kernel='rbf')

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
            features = [int(user['userId']), movie['movieId']]
            for genre in self.embedding.transform(movie['genres']):
                features.append(genre)
            # print(features)
            x.append(features)
            y.append(edge[2]['rating'])
        return np.array(x), np.array(y)

    def fit(self):
        """The method fits the SVM model with the extracted features and the corresponding ratings."""
        # print(self.x)
        self.model.fit(self.x, self.y)

    def predict(self, x):
        """Returns the predicts the possible rating for a potential edge between a user and a movie."""
        pred = self.model.predict(x)
        return pred
