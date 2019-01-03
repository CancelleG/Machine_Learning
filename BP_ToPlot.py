import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)
    X_train = MinMaxScaler().fit_transform(X_train)
    return X_train, y_train


def predict_data():
    df_X = pd.read_csv("./data/Testing-set.csv")
    df_y = pd.read_csv("./data/Testing-set-label.csv")
    X_test = df_X.drop('id', axis=1)
    X_test = MinMaxScaler().fit_transform(X_test)
    y_test = df_y.iloc[:, -1].values
    return X_test, y_test

def main():
    params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
               'learning_rate_init': 1e-4},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 1e-4},
              {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 1e-4},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
               'learning_rate_init': 1e-4},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': True, 'learning_rate_init': 1e-4},
              {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
               'nesterovs_momentum': False, 'learning_rate_init': 1e-4},
              {'solver': 'adam', 'learning_rate_init': 6e-3}]

    labels = ["constant learning-rate", "constant with momentum",
              "constant with Nesterov's momentum",
              "inv-scaling learning-rate", "inv-scaling with momentum",
              "inv-scaling with Nesterov's momentum", "adam"]

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'black', 'linestyle': '-'}]


    X_train, y_train = input_data()
    X_test, y_test = predict_data()
    # mlp = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(5, 10),
    #                     random_state=1, max_iter=2000)
    mlps = []
    for param in params:
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=2000, **param)
        mlp.fit(X_train, y_train)
        mlps.append(mlp)
        # print(mlp.score(X_test, y_test))
        print("Training set score: %f" % mlp.score(X_test, y_test))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        plt.plot(mlp.loss_curve_, label=label, **args)
    plt.legend(labels, ncol=3, loc="upper center")
    plt.show()


if __name__ == '__main__':
    main()