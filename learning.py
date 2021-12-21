import numpy as np
from embedding import Embedding
from sklearn.svm import SVR


class Learning:
    def __init__(self, graph, embedding=Embedding('tf')):
        self.svr = None
        self.recommended_nodes = None
        self.graph = graph
        self.embedding = embedding
        self.x, self.y, self.movies_title = self.get_x_y(self.graph.edges(data=True))

    def get_x_y(self, edges):
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
            y.append(edge[2]['rating']/5)
        return np.array(x), np.array(y), movies_title

    def svm_fit(self):
        self.svr = SVR()
        # print(self.x)
        self.svr.fit(self.x, self.y)

    def svm_predict(self, x, movies_title, top=10):
        pred = self.svr.predict(x)
        # print(pred, len(pred))
        self.__get_predictions(pred, x, movies_title)
        self.__get_top(top)
        return self.recommended_nodes

    def __get_predictions(self, predicted, x, movies_title):
        self.recommended_nodes = {}
        for i in range(len(x)):
            # print(x[i][0], x[i][1], movies_title[x[i][1]], predicted[i])
            user = str(float(x[i][0]))
            if predicted[i] < 0.66:
                continue
            if user not in self.recommended_nodes:
                self.recommended_nodes[user] = {
                    movies_title[x[i][1]]: predicted[i]
                }
            else:
                self.recommended_nodes[user][movies_title[x[i][1]]] = predicted[i]

    def __get_top(self, top=10):
        for user in self.recommended_nodes:
            top_n = []
            for i in range(top):
                if len(self.recommended_nodes[user]) == 0:
                    break
                key = max(self.recommended_nodes[user], key=self.recommended_nodes[user].get)
                top_n.append(key)
                self.recommended_nodes[user].pop(key)
            self.recommended_nodes[user] = top_n
