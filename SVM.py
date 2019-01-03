import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn import svm

def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)
    return X_train, y_train


def predict_data():
    df_X = pd.read_csv("./data/Testing-set.csv")
    df_y = pd.read_csv("./data/Testing-set-label.csv")
    X_test = df_X.drop('id', axis=1)
    y_test = df_y.iloc[:, -1].values
    return X_test, y_test

def main():
    X_train, y_train = input_data()
    X_test, y_test = predict_data()
    # clf = svm.SVC(C=0.7, kernel='rbf', coef0=5,
    #               decision_function_shape='ovo', verbose=1, max_iter=-1, tol=1e-3)
    clf = svm.SVC(C=1, kernel='rbf', verbose=1)
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    # y_pred = GaussianNB().fit(X_train, y_train).predict(X_test)
    print("total %d points, Suceecd: %d"
          % (len(y_test), (y_test == y_pred).sum()))
    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Testing set score: %f" % clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
