import pandas as pd
import networkx as nx
import numpy as np
import csrgraph as cg
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from scipy.spatial import distance


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
   
    @staticmethod
    def jaccard(a, b):
        nominator = a.intersection(b)
        denominator = a.union(b)
        similarity = len(nominator)/len(denominator)
        return similarity

    @staticmethod
    def euclidean_distance(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def manhattan_distance(a, b):
        return distance.cityblock(a, b)

class Embedding:
    def __init__(self, method='tf'):
        self.corpus = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                       'War', 'Western', '(no genres listed)']
        self.method = method
        if method == 'tf-idf':
            self.tfidf = TfidfVectorizer()
            self.tfidf.fit(self.corpus)

    def transform(self, document):
        if self.method == 'tf':
            return [1 if value in document else 0 for value in self.corpus]
        elif self.method == 'tf-idf':
            return self.tfidf.transform([document])


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


class Heuristic:
    def __init__(self, graph):
        self.graph = graph
        self.recommended_nodes = {}

    def __sim_cal(self, degree_as, hops, value, node, user):
        if hops == degree_as:
            if node not in self.recommended_nodes[user]:
                self.recommended_nodes[user][node] = value
            else:
                self.recommended_nodes[user][node] += value
            return
        edges = self.graph[node]
        for key in edges:
            if 'rating' in edges[key]:
                continue
            self.__sim_cal(degree_as, hops+1, value*edges[key]['similarity'], key, user)

    def degree_association(self, degree=2):
        print(f'Searching for potential links between nodes with degree association {degree}. This may take a while. Please wait...')
        for node in self.graph.nodes(data=True):
            if node[1]['type'] == 'user':
                edges = self.graph[node[0]]
                for key in edges:
                    if edges[key]['rating'] < 3.5:
                        continue
                    val = edges[key]['rating']/5
                    self.recommended_nodes[node[0]] = {}
                    self.__sim_cal(degree, 1, val, key, node[0])
        print('Searching completed')

    def get_recommended_nodes(self, top=10):
        self.__get_top(top)
        return self.recommended_nodes

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
            if predicted[i] < 0.7:
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


class Hybrid:
    def __init__(self, graph, embedding=Embedding('tf')):
        self.graph = graph
        self.embedding = embedding
        self.x, self.y, self.movies_title, self.users = self.__random_walk(self.graph.edges(data=True))

    def __get_features(self, random_walks):
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
            walks = np.vectorize(lambda x: node_names[x])(G.random_walks(walklen=5, epochs=1, start_nodes=[node_1, node_2], return_weight=1., neighbor_weight=1.))
            for i in range(0, len(walks), 2):
                random_walks.append(list(walks[i][::-1]) + list(walks[i+1]))
                y.append(edge[2]['rating']/5)
                movies_title.append(movie['label'])
                users.append(user['userId'])
        print('Random walk completed')
        return self.__get_features(random_walks), np.array(y), movies_title, users

    def get_x_y(self, edges):
        x, y, movies_title, users = self.__random_walk(edges)
        return x, y, movies_title, users

    def svm_fit(self):
        print('Fitting SVM model. Please wait...')
        self.svr = SVR()
        print(self.x.shape)
        print(self.x)
        self.svr.fit(self.x, self.y)
        print('SVM trained')

    def svm_predict(self, x, movies_title, users, top=10):
        # print(x)
        pred = self.svr.predict(x)
        # print(pred, len(pred))
        self.__get_predictions(pred, x, movies_title, users)
        self.__get_top(top)
        return self.recommended_nodes

    def __get_predictions(self, predicted, x, movies_title, users):
        self.recommended_nodes = {}
        for i in range(len(users)):
            # print(users[i], movies_title[i], predicted[i])
            user = str(float(users[i]))
            if predicted[i] < 0.7:
                continue
            if user not in self.recommended_nodes:
                self.recommended_nodes[user] = {
                    movies_title[i]: predicted[i]
                }
            else:
                self.recommended_nodes[user][movies_title[i]] = predicted[i]

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




class Evaluation:
    def __init__(self):
        pass


if __name__ == '__main__':
    graph = Graph()
    # graph.init(dataset_directory='data/dataset/', keep_only_good_ratings=False, bipartite=True)
    # graph.export_gexf(directory='data/graph/')
    graph.read_gexf('data/graph/graph.gexf')
    graph_train, test_edges = graph.split_train_test(0.3)
    print(len(test_edges))
    ##### Heuristic #####
    # heuristic = Heuristic(graph_train)
    # heuristic.degree_association(4)
    # rec = heuristic.get_recommended_nodes(20)

    ##### Learning #####
    # learning = Learning(graph_train)
    # learning.svm_fit()
    # embedding = Embedding('tf')
    # test_x, test_y, movies_title = learning.get_x_y(test_edges)
    # rec = learning.svm_predict(np.array(test_x), movies_title, top=20)

    ##### Hybrid #####
    hybrid = Hybrid(graph_train)
    hybrid.svm_fit()
    embedding = Embedding('tf')
    test_x, test_y, movies_title, users = hybrid.get_x_y(test_edges)
    rec = hybrid.svm_predict(test_x, movies_title, users, top=20)

    print(rec)
    link = []
    for edge in test_edges:
        # print(edge)
        if edge[0] in rec and edge[1] in rec[edge[0]]:
            link.append((edge[0], edge[1], rec[edge[0]]))
            # print(edge[0], edge[1], rec[edge[0]], len(rec[edge[0]]))
        elif edge[1] in rec and edge[0] in rec[edge[1]]:
            link.append((edge[1], edge[0], rec[edge[1]]))
            # print(edge[1], edge[0], rec[edge[1]], len(rec[edge[1]]))
    print(len(link))


    # embedding = Embedding('tf')
    # test1 = embedding.transform('Animation|War|Thriller')
    # print(test1)
    # test2 = embedding.transform('War|Thriller')
    # print(test2)
    # sim = Similarity.cosine(test1, test2)
    # print(sim)
