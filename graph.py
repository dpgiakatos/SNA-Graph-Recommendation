import random
import networkx as nx
from embedding import Embedding
from similarity import Similarity
from dataset import Dataset
from heuristic import Heuristic


class Graph:
    """This class initialize the graph from the MovieLens dataset."""
    def __init__(self):
        self.graph = nx.Graph()

    def __print_graph_info(self):
        """Prints some infos for the graph."""
        print(f'Bipartite graph: {nx.bipartite.is_bipartite(self.graph)}')
        print(f'Total nodes: {nx.number_of_nodes(self.graph)}')
        print(f'Total edges: {nx.number_of_edges(self.graph)}')

    def __clean_graph(self):
        """Cleaning graph from isolated nodes. Then, the graph will have only one component."""
        print('Checking if graph contains isolated nodes...')
        if nx.number_connected_components(self.graph) > 1:
            pass

    def __insert_movies_nodes(self, movies):
        """Inserting the movies nodes, with id the movie title, and attributes the type, genres and id.
        This method will run only for bipartite graph."""
        for index, movie in movies.iterrows():
            self.graph.add_node(movie['title'], type='movie', genres=movie['genres'], movieId=movie['movieId'])

    def __create_movies_graph(self, movies, embedding):
        """Inserting the movies nodes, with id the movie title, and attributes the type, genre and id.
        Also, the movie nodes that have similar nodes (>0.8), they will connect with a node with attributes the
        similarity between the movies. So, this method creates a graph between the movies. The final graph will not
         be bipartite."""
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
        """Initialize the final graph (user-movie). First, the movie nodes will be created. Second, the user nodes
        will be created with id the user id and attributes the type and user id. Then the nodes will be connected
        with the rating that a user has put to a movie. Also, the edges that connect the users with the movies with have
        as attributes the user's tag for the movie and the rating."""
        dataset = Dataset(dataset_directory)
        print('Graph initialing...')
        movies = dataset.get_movies()
        tags = dataset.get_tags()
        if bipartite:
            # We keep a subsample from the dataset. The 500 first rows.
            self.__insert_movies_nodes(movies.head(500))
        else:
            # We keep a subsample from the dataset. The 100 first rows.
            self.__create_movies_graph(movies.head(500), embedding)
        for index, value in dataset.get_ratings(normalize=True).iterrows():
            movie = movies.loc[movies['movieId'] == value['movieId']].iloc[0]
            if movie['title'] not in self.graph:
                continue
            self.graph.add_node(value['userId'], type='user', userId=value['userId'])
            # self.graph.add_node(movie['title'], type='movie', genres=movie['genres'])
            tag = tags.loc[(tags['userId'] == value['userId']) & (tags['movieId'] == value['movieId'])]
            tag_value = '' if tag.empty else tag.iloc[0]['tag']
            self.graph.add_edge(value['userId'], movie['title'], rating=value['rating'], tag=tag_value)
        self.connect_users(dataset.get_similarity_users())

        print('Graph created')
        self.__print_graph_info()

    def connect_users(self, df):
        for index, value in df.iterrows():
            if value['User1'] in self.graph and value['User2'] in self.graph:
                self.graph.add_edge(value['User1'], value['User2'], similarity=value['Similarity'])




    def export_gexf(self, directory):
        """Export the graph to file."""
        nx.write_gexf(self.graph, directory + 'graph.gexf')

    def read_gexf(self, file_path):
        """Create the graph from file."""
        print('Graph loading...')
        self.graph = nx.read_gexf(file_path)
        print('Graph loaded')
        self.__print_graph_info()

    def get_graph(self):
        """Return the graph object."""
        return self.graph

    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        adj = nx.adjacency_matrix(self.graph)
        return adj.todense()

    def split_train_test(self, split=0.1):
        """Split the graph to a train, test set. The method removes the edges from the graph that have a rating
        grater that 3.5. The method returns the graph with the removed edges and a list of the removed edges. The
        default splitting is 10% (0.1) for the test set. This means that the 10% of the edges will be removed."""
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
            graph_copy.remove_edge(*edge[:2])
            # print(nx.number_connected_components(graph_copy))
            if nx.number_connected_components(graph_copy) == 1:
                train_graph.remove_edge(*edge[:2])
                test.append(edge)
            graph_copy.add_edge(edge[0], edge[1], **edge[2])
        print('Splitting completed')
        return train_graph, test
