from graph import Graph
from embedding import Embedding
from heuristic import Heuristic
from learning import Learning
from hybrid import Hybrid
from sklearn.metrics import mean_squared_error
from dataset import Dataset


if __name__ == '__main__':
    graph = Graph()
    # graph.init(dataset_directory='data/dataset/', keep_only_good_ratings=False, bipartite=False)
    # graph.export_gexf(directory='data/graph/')
    # exit(0)
    graph.read_gexf('data/graph/graph.gexf')
    graph_train, test_edges = graph.split_train_test(0.1)
    print(len(test_edges))
    ##### Heuristic #####
    heuristic = Heuristic(graph_train)
    rec_e = heuristic.collaborative_filtering(test_edges)

    ##### Learning #####
    learning = Learning(graph_train, model='svm-rbf')
    learning.fit()
    test_x_l, test_y_l = learning.get_x_y(test_edges)
    rec_l = learning.predict(test_x_l)

    ##### Hybrid #####
    hybrid = Hybrid(graph_train, model='knn')
    hybrid.fit()
    test_x_h, test_y_h = hybrid.get_x_y(test_edges)
    rec_h = hybrid.predict(test_x_h)

    # print(rec)
    ratings = []
    for edge in test_edges:
        # print(edge)
        ratings.append(edge[2]['rating'])
    print(f"Mean squared error: {mean_squared_error(ratings, rec_e)}")
    print(f"Mean squared error: {mean_squared_error(ratings, rec_l)}")
    print(f"Mean squared error: {mean_squared_error(ratings, rec_h)}")


    # embedding = Embedding('tf')
    # test1 = embedding.transform('Animation|War|Thriller')
    # print(test1)
    # test2 = embedding.transform('War|Thriller')
    # print(test2)
    # sim = Similarity.cosine(test1, test2)
    # print(sim)
