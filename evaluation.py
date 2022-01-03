from sklearn import metrics


class Evaluation:

    @staticmethod
    def model_evaluation(y_test, y_predicted):
        ratings = []
        for edge in y_test:
            ratings.append(edge[2]['rating'])
        print(f"Mean Squared Error: {metrics.mean_squared_error(ratings, y_predicted)}")
        print(f"Root Mean Squared Error: {metrics.mean_squared_error(ratings, y_predicted, squared=False)}")
        print(f"R-Squared: {metrics.r2_score(ratings, y_predicted)}")
