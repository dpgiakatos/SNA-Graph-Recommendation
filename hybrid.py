import numpy as np
from embedding import Embedding
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from node2vec import Node2Vec


class Hybrid:
    """This class contains some hybrid algorithms. A hybrid algorithm is a combination with a heuristic and a learning
    based model. With the heuristic algorithm we extract the graph structure related features and then these features
    we use them as input in our learning based model."""

    def __init__(self, graph, model='svm-rbf', embedding=Embedding('tf'), node_dimensions=100):
        self.node2vec = None
        self.graph = graph
        self.embedding = embedding
        self.node_dimensions = node_dimensions
        self.__node2vec()
        self.__select_model(model)
        self.x, self.y = self.get_x_y(self.graph.edges(data=True))

    def __select_model(self, model):
        """Model initialization from the given parameter."""
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

    def __node2vec(self):
        """Extract node features using graph structure (node2vec)."""
        self.node2vec = Node2Vec(self.graph, dimensions=self.node_dimensions, walk_length=5, num_walks=100).fit()

    def get_x_y(self, edges):
        """From the edge list (parameter), returns the features for each edge that has a rating,
        the corresponding ratings, movie titles and users"""
        x = []
        y = []
        for edge in edges:
            if 'similarity' in edge[2]:
                continue
            x.append(self.node2vec.wv[str(edge[0])] + self.node2vec.wv[str(edge[1])])
            y.append(edge[2]['rating'])
        return np.array(x), np.array(y)

    def fit(self):
        """The method fits the SVM model with the extracted features and the corresponding ratings."""
        print('Fitting model. Please wait...')
        self.model.fit(self.x, self.y)
        print('Model trained')

    def predict(self, x):
        """Returns the predicts the possible rating for a potential edge between a user and a movie."""
        pred = self.model.predict(x)
        return pred
