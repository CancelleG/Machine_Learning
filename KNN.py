import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)

    ###########对X的特殊处理############
    X_train = X_train.abs()
    X_train = MinMaxScaler(feature_range=(0, 100)).fit_transform(X_train)
    return X_train, y_train


def predict_data():
    df_X = pd.read_csv("./data/Testing-set.csv")
    df_y = pd.read_csv("./data/Testing-set-label.csv")
    X_test = df_X.drop('id', axis=1)
    y_test = df_y.iloc[:, -1].values
    ###########对X的特殊处理############
    X_test = X_test.abs()
    X_test = MinMaxScaler(feature_range=(0, 10000)).fit_transform(X_test)
    return X_test, y_test

def main():
    X_train, y_train = input_data()
    X_test, y_test = predict_data()

    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    # y_pred = GaussianNB().fit(X_train, y_train).predict(X_test)
    print("total %d points, Suceecd: %d"
          % (len(y_test), (y_test == y_pred).sum()))
    print("Training set score: %f" % clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
