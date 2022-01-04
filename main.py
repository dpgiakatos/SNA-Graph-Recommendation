from graph import Graph
from evaluation import Evaluation
from heuristic import Heuristic
from learning import Learning
from hybrid import Hybrid
from sklearn.metrics import mean_squared_error, r2_score
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
    # graph.init(dataset_directory='data/dataset/')
    # graph.export_gexf(directory='data/graph/')
    # exit(0)
    graph.read_gexf('data/graph/graph0_6.gexf')
    graph_train, test_edges = graph.split_train_test(0.2)
    print(len(test_edges))

    ##### Heuristic #####
    heuristic = Heuristic(graph_train)
    rec_e = heuristic.collaborative_filtering(test_edges, mode='movies')

    ##### Learning #####
    learning = Learning(graph_train, model='linear')
    learning.fit()
    test_x_l, test_y_l = learning.get_x_y(test_edges)
    rec_l = learning.predict(test_x_l)

    ##### Hybrid #####
    hybrid = Hybrid(graph_train, model='linear', node_dimensions=128)
    hybrid.fit()
    test_x_h, test_y_h = hybrid.get_x_y(test_edges)
    rec_h = hybrid.predict(test_x_h)

    ##### MLPS #####
    learning_features, learning_target = learning.get_x_y(graph_train.edges(data=True))
    hybrid_features, hybrid_target = hybrid.get_x_y(graph_train.edges(data=True))
    mlps = get_mlps(learning_features, hybrid_features)
    mlps.fit([learning_features, hybrid_features], learning_target, batch_size=1000, epochs=70)
    rec_m = mlps.predict([test_x_l, test_x_h])
    # loss = mlps.evaluate([test_x_l, test_x_h], test_y_h)

    ##### EVALUATION #####
    Evaluation.model_evaluation(test_edges, rec_e)
    Evaluation.model_evaluation(test_edges, rec_l)
    Evaluation.model_evaluation(test_edges, rec_h)
    Evaluation.model_evaluation(test_edges, rec_m)

    # print(f"Mean squared error: {loss}")
