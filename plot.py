import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def input_data():
    df = pd.read_csv("./data/Training-set.csv")
    y_train = df.iloc[:, -1].values
    X_train = df.drop(['id', 't'], axis=1)
    # X_train = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_train)
    # X_train = StandardScaler().fit_transform(X_train)
    return X_train, y_train


def predict_data():
    df_X = pd.read_csv("./data/Testing-set.csv")
    df_y = pd.read_csv("./data/Testing-set-label.csv")
    X_test = df_X.drop('id', axis=1)
    # X_test = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_test)
    # X_test = StandardScaler().fit_transform(X_test)
    y_test = df_y.iloc[:, -1].values
    return X_test, y_test

def main():
    X_train, y_train = input_data()
    X_test, y_test = predict_data()
###########################seaborn散点####################################
    # sns.scatterplot(x=X_train.iloc[:, 0].values, y=X_train.iloc[:, 1].values, hue=y_train)
    # plt.xlabel('a')
    # plt.ylabel('b')
    # plt.show()


###########################plt散点####################################

    # plt.scatter(X_train.iloc[:, 0].values, X_train.iloc[:, 1].values, s=5)
    # plt.xlabel('a')
    # plt.ylabel('b')
    # plt.show()

###########################plt 3维散点  train####################################
    # x = X_train.iloc[:, 0].values
    # y = X_train.iloc[:, 1].values
    # z = X_train.iloc[:, 2].values
    # ax = plt.subplot(111, projection='3d')
    # for index_y in range(len(y_train)):
    #     if y_train[index_y] == 0:
    #         ax.scatter(x[index_y], y[index_y], z[index_y], c='royalblue', alpha=0.7, s=5)  # 绘制数据点
    #     else:
    #         ax.scatter(x[index_y], y[index_y], z[index_y], c='orange', alpha=0.7, s=5)  # 绘制数据点
    # # edgecolor='w'
    # ax.set_zlabel('a')  # 坐标轴
    # ax.set_ylabel('b')
    # ax.set_xlabel('c')
    # plt.show()
###########################plt 3维散点  test####################################
    x = X_test.iloc[:, 0].values
    y = X_test.iloc[:, 1].values
    z = X_test.iloc[:, 2].values
    ax = plt.subplot(111, projection='3d')
    for index_y in range(len(y_test)):
        if y_test[index_y] == 0:
            ax.scatter(x[index_y], y[index_y], z[index_y], c='royalblue', alpha=0.7, s=5)  # 绘制数据点
        else:
            ax.scatter(x[index_y], y[index_y], z[index_y], c='royalblue', alpha=0.7, s=5)  # 绘制数据点
    # edgecolor='w'
    ax.set_zlabel('a')  # 坐标轴
    ax.set_ylabel('b')
    ax.set_xlabel('c')
    plt.show()

if __name__ == '__main__':
    main()
