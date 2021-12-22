import numpy as np
import csrgraph as cg
from embedding import Embedding
from sklearn.svm import SVR
from node2vec import Node2Vec


class Hybrid:
    """This class contains some hybrid algorithms. A hybrid algorithm is a combination with a heuristic and a learning
    based model. With the heuristic algorithm we extract the graph structure related features and then these features
    we use them as input in our learning based model."""
    def __init__(self, graph, embedding=Embedding('tf')):
        self.svr = None
        self.node2vec = None
        self.graph = graph
        self.embedding = embedding
        self.__node2vec()
        self.x, self.y = self.get_x_y(self.graph.edges(data=True))

    def __get_features(self, random_walks):
        """This solution may be wrong!!!
        The method extracts the features from the walks that have been created from the random walk."""
        print('Extracting features...')
        x = []
        for walk in random_walks:
            features = []
            for node in walk:
                if self.graph.nodes[node]['type'] == 'user':
                    user = self.graph.nodes[node]
                    features.append(int(user['userId']))
                else:
                    movie = self.graph.nodes[node]
                    features.append(movie['movieId'])
                    for genre in self.embedding.transform(movie['genres']):
                        features.append(genre)
            x.append(features)
        print('Features extracted')
        return np.array(x)

    def __random_walk(self, edges):
        """This solution may be wrong!!!
        For each edge that connects a user with a movie, we generate some random walks that contains that edge."""
        print('Random walk starting...')
        G = cg.csrgraph(self.graph, threads=12)
        node_names = G.names
        random_walks = []
        y = []
        movies_title = []
        users = []
        for edge in edges:
            if self.graph.nodes[edge[0]]['type'] == 'user':
                user = self.graph.nodes[edge[0]]
                movie = self.graph.nodes[edge[1]]
            else:
                user = self.graph.nodes[edge[1]]
                movie = self.graph.nodes[edge[0]]
            node_1 = node_names[node_names == user['label']].index[0]
            node_2 = node_names[node_names == movie['label']].index[0]
            walks = np.vectorize(lambda x: node_names[x])(G.random_walks(walklen=2, epochs=2, start_nodes=[node_1, node_2], return_weight=1.5, neighbor_weight=1))
            for i in range(0, len(walks), 2):
                random_walks.append(list(walks[i][::-1]) + list(walks[i+1]))
                y.append(edge[2]['rating']/5)
                movies_title.append(movie['label'])
                users.append(user['userId'])
        print('Random walk completed')
        return self.__get_features(random_walks), np.array(y), movies_title, users

    def __node2vec(self):
        """Extract node features using graph structure (node2vec)."""
        self.node2vec = Node2Vec(self.graph, dimensions=100, walk_length=5, num_walks=100).fit(window=7, min_count=1)

    def get_x_y(self, edges):
        """From the edge list (parameter), returns the features for each edge that has a rating,
        the corresponding ratings, movie titles and users"""
        x = []
        y = []
        for edge in edges:
            x.append(self.node2vec.wv[str(edge[0])] + self.node2vec.wv[str(edge[1])])
            y.append(edge[2]['rating'] / 5)
        return np.array(x), np.array(y)

    def svm_fit(self):
        """The method fits the SVM model with the extracted features and the corresponding ratings."""
        print('Fitting SVM model. Please wait...')
        self.svr = SVR()
        # print(self.x.shape)
        # print(self.x)
        self.svr.fit(self.x, self.y)
        print('SVM trained')

    def svm_predict(self, x):
        """Returns the predicts the possible rating for a potential edge between a user and a movie."""
        pred = self.svr.predict(x)
        return pred
