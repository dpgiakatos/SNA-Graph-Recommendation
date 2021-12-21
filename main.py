from graph import Graph
from embedding import Embedding
from heuristic import Heuristic
from learning import Learning
from hybrid import Hybrid


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

    # print(rec)
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
