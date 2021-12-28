from sklearn import metrics


class Evaluation:

    def model_evaluation(y_test, y_predicted):
        print("Correlation: %2f, p-value: %2f" % stats.pearsonr(y_test, y_predicted))
        print("Mean Squared Error: %2f" % metrics.mean_squared_error(y_test, y_predicted))
        RMSE = metrics.mean_squared_error(y_test, y_predicted, squared=False)
        print("Root Mean Squared Error: %2f" % RMSE)
        print("R-Squared: %2f" % metrics.r2_score(y_test, y_predicted))
