import numpy as np
import csrgraph as cg
from embedding import Embedding
from sklearn.svm import SVR


class Hybrid:
    """This class contains some hybrid algorithms. A hybrid algorithm is a combination with a heuristic and a learning
    based model. With the heuristic algorithm we extract the graph structure related features and then these features
    we use them as input in our learning based model."""
    def __init__(self, graph, embedding=Embedding('tf')):
        self.graph = graph
        self.embedding = embedding
        self.x, self.y, self.movies_title, self.users = self.__random_walk(self.graph.edges(data=True))

    def __get_features(self, random_walks):
        """The method extracts the features from the walks that have been created from the random walk."""
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
        """For each edge that connects a user with a movie, we generate some random walks that contains that edge."""
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

    def get_x_y(self, edges):
        """From the edge list (parameter), returns the features for each edge that has a rating,
        the corresponding ratings, movie titles and users"""
        x, y, movies_title, users = self.__random_walk(edges)
        return x, y, movies_title, users

    def svm_fit(self):
        """The method fits the SVM model with the extracted features and the corresponding ratings."""
        print('Fitting SVM model. Please wait...')
        self.svr = SVR()
        # print(self.x.shape)
        # print(self.x)
        self.svr.fit(self.x, self.y)
        print('SVM trained')

    def svm_predict(self, x, movies_title, users, top=10):
        """The method predicts the possible rating for a potential edge between a user and a movie.
        Returns the top k (default k=10) recommendations for the users."""
        # print(x)
        pred = self.svr.predict(x)
        # print(pred, len(pred))
        self.__get_predictions(pred, movies_title, users)
        self.__get_top(top)
        return self.recommended_nodes

    def __get_predictions(self, predicted, movies_title, users):
        """This method creates a dictionary that contains the user ids and for each id a list with the movie titles
        that we will recommend to that user. We recommend a movie that will have a possible rating greater than 3.25."""
        self.recommended_nodes = {}
        for i in range(len(users)):
            # print(users[i], movies_title[i], predicted[i])
            user = str(float(users[i]))
            if predicted[i] < 0.66:
                continue
            if user not in self.recommended_nodes:
                self.recommended_nodes[user] = {
                    movies_title[i]: predicted[i]
                }
            else:
                self.recommended_nodes[user][movies_title[i]] = predicted[i]

    def __get_top(self, top=10):
        """Selects the top k (default k=10) movies with max similarity."""
        for user in self.recommended_nodes:
            top_n = []
            for i in range(top):
                if len(self.recommended_nodes[user]) == 0:
                    break
                key = max(self.recommended_nodes[user], key=self.recommended_nodes[user].get)
                top_n.append(key)
                self.recommended_nodes[user].pop(key)
            self.recommended_nodes[user] = top_n
