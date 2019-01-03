import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn import tree


def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)
    # X_train = MinMaxScaler().fit_transform(X_train)
    return X_train, y_train


def predict_data():
    df_X = pd.read_csv("./data/Testing-set.csv")
    df_y = pd.read_csv("./data/Testing-set-label.csv")
    X_test = df_X.drop('id', axis=1)
    # X_test = MinMaxScaler().fit_transform(X_test)
    y_test = df_y.iloc[:, -1].values
    return X_test, y_test

def main():
    X_train, y_train = input_data()
    X_test, y_test = predict_data()
    clf = tree.DecisionTreeClassifier(random_state=0, splitter='random', max_depth=16)
    #criterion='entropy',  , , max_leaf_nodes=414, max_leaf_nodes=1000
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    # y_pred = GaussianNB().fit(X_train, y_train).predict(X_test)
    print("total %d points, Suceecd: %d"
          % (len(y_test), (y_test == y_pred).sum()))
    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Testing set score: %f" % clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
