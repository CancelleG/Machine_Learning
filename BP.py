import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)
    # X_train = X_train.sample(frac=1).reset_index(drop=True)
    # X_train = shuffle(X_train)

    #############采用计算椭球方程的方式处理X###################
    X_train = np.multiply(X_train, X_train)
    return X_train, y_train


def predict_data():
    df = pd.read_csv("./data/Testing-set-label.csv")
    y_test = df.iloc[:, -1].values
    X_test = df.drop(['id', 't'], axis=1)

    #############采用计算椭球方程的方式处理X###################
    X_test = np.multiply(X_test, X_test)
    return X_test, y_test


def main():
    X_train, y_train = input_data()
    X_test, y_test = predict_data()
    mlp = MLPClassifier(verbose=1, random_state=0,
                        max_iter=50, solver='adam', learning_rate_init=4e-5, tol=1e-150, hidden_layer_sizes=(300, 300))
    # mlp = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(5, 10),
    #                     random_state=1, max_iter=200000)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_test, y_test))
    print("Training set loss: %f" % mlp.loss_)


    #plt.plot(mlp.loss_curve_)
    #plt.show()


if __name__ == '__main__':
    main()