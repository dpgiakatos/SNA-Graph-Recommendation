from graph import Graph
from embedding import Embedding
from heuristic import Heuristic
from learning import Learning
from hybrid import Hybrid
from sklearn.metrics import mean_squared_error, r2_score
from dataset import Dataset
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers.merge import concatenate


def get_mlps(input_1, input_2):
    model_1 = Sequential()
    model_1.add(Dense(10, input_dim=input_1.shape[1], activation='sigmoid'))
    model_2 = Sequential()
    model_2.add(Dense(50, input_dim=input_2.shape[1], activation='sigmoid'))
    model_3 = concatenate([model_1.output, model_2.output])
    model_3 = Dense(1, activation='sigmoid')(model_3)
    model = Model(inputs=[model_1.input, model_2.input], outputs=model_3)
    opt = tf.keras.optimizers.Adam(learning_rate=0.03)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model


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
    learning = Learning(graph_train, model='linear')
    learning.fit()
    test_x_l, test_y_l = learning.get_x_y(test_edges)
    rec_l = learning.predict(test_x_l)

    ##### Hybrid #####
    hybrid = Hybrid(graph_train, model='knn')
    hybrid.fit()
    test_x_h, test_y_h = hybrid.get_x_y(test_edges)
    rec_h = hybrid.predict(test_x_h)

    ##### MLPS #####
    learning_features, learning_target = learning.get_x_y(graph_train.edges(data=True))
    hybrid_features, hybrid_target = hybrid.get_x_y(graph_train.edges(data=True))
    mlps = get_mlps(learning_features, hybrid_features)
    mlps.fit([learning_features, hybrid_features], learning_target, batch_size=1000, epochs=70)
    loss = mlps.evaluate([test_x_l, test_x_h], test_y_h)

    ratings = []
    for edge in test_edges:
        ratings.append(edge[2]['rating'])
    print(f"Mean squared error: {mean_squared_error(ratings, rec_e)} (Heuristic)")
    print(f"Mean squared error: {mean_squared_error(ratings, rec_l)} (Learning)")
    print(f"Mean squared error: {mean_squared_error(ratings, rec_h)} (Hybrid)")
    print(f"Mean squared error: {loss}")

    print(f"R2 score: {r2_score(ratings, rec_e)} (Heuristic)")
    print(f"R2 score: {r2_score(ratings, rec_l)} (Learning)")
    print(f"R2 score: {r2_score(ratings, rec_h)} (Hybrid)")

    # embedding = Embedding('tf')
    # test1 = embedding.transform('Animation|War|Thriller')
    # print(test1)
    # test2 = embedding.transform('War|Thriller')
    # print(test2)
    # sim = Similarity.cosine(test1, test2)
    # print(sim)
