class Heuristic:
    """This class contains some heuristics algorithms for a graph recommendation system. Each algorithm returns
    a dictionary that contains the user ids and for each id a list with the movie titles that we will recommend to that
    user. We recommend a movie that will have a possible rating greater than 3.25."""
    def __init__(self, graph):
        self.graph = graph
        self.recommended_nodes = {}

    def __sim_cal(self, degree_as, hops, value, node, user):
        """This a recursive method that calculates the similarity between the movies that have a specific depth."""
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
        """For each user we calculate the similarity association based to the depth degree. By default, the depth degree
        is 2. This algorithm inspired from the heuristic algorithm presented by Huang Z. et al. in the paper with title
        A Graph-based Recommender System for Digital Library."""
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
        """Returns the top k (default k=10) recommendations for the users."""
        self.__get_top(top)
        return self.recommended_nodes

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
