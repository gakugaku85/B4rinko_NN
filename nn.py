import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

np.random.seed(5)

SAVE_RESULT_PATH = "./result/"
if not os.path.isdir(SAVE_RESULT_PATH):
    os.makedirs(SAVE_RESULT_PATH)

iris = load_iris()

feature_names = iris.feature_names
target_names = iris.target_names

class NeuralNetwork:

    def __init__(self, eta=0.1):
        self.w = np.random.randn(4)  # wの初期値はランダム
        self.b = np.random.randn(1)  # bも初期値を0.1にする。
        self.eta = eta

    def get_weight(self):
        return self.w

    def get_bias(self):
        return self.b

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def activation(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def loss(self, X, y):
        dif = y - self.activation(X)
        return np.sum(dif**2/(2*len(y)), keepdims=True)

    def accuracy(self, X, y):
        pre = self.predict(X)
        return np.sum(np.where(pre==y, 1, 0))/len(y)

    def predict(self, X):
        result = np.where(self.activation(X)<0.5, 0.0, 2.0)
        return result

    # 微分
    def update(self, X, y):
        a = (self.activation(X) - y)*self.activation(X)*(1 - self.activation(X))
        a = a.reshape(-1, 1)
        self.w -= self.eta * 1/float(len(y))*np.sum(a*X,axis=0)
        self.b -= self.eta * 1/float(len(y))*np.sum(a)

def plot(x, y, title, x_label, y_label, label ,file_name):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, y_i in enumerate(y):
        plt.plot(x, y_i, label=label[i])

    plt.grid()
    plt.legend()

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def load_data():

    data = iris.data
    label = iris.target

    x1 = data[:30]
    x2 = data[100:130]
    x_train = np.concatenate((x1, x2), axis=0)

    ltr1 = label[:30]
    ltr2 = label[100:130]
    l_train = np.concatenate((ltr1, ltr2), axis=0)

    val1 = data[30:40]
    val2 = data[130:140]
    y_val = np.concatenate((val1, val2), axis=0)

    ga1 = label[30:40]
    ga2 = label[130:140]
    l_val = np.concatenate((ga1, ga2), axis=0)

    y1 = data[40:50]
    y2 = data[140:150]
    y_test = np.concatenate((y1, y2), axis=0)

    lte1 = label[40:50]
    lte2 = label[140:150]
    l_test = np.concatenate((lte1, lte2), axis=0)

    return x_train, l_train, y_val, l_val, y_test, l_test

def get_df_and_csv(result, columns, file_name):

    df = pd.DataFrame(result, columns=columns)

    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)

    return df

def main():

    nn = NeuralNetwork() #解析的微分

    epochs = 100

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    result = []
    columns = ["epoch", "acc_train", "loss_train", "acc_val", "loss_val", "acc_test", "loss_test"]
    weights = nn.get_weight()
    bias = nn.get_bias()
    print('weights = ', weights, 'bias = ', bias)
    for epoch in range(epochs):
        nn.update(X_train, y_train)
        acc_train = nn.accuracy(X_train, y_train)
        loss_train = nn.loss(X_train, y_train)
        acc_val = nn.accuracy(X_val, y_val)
        loss_val = nn.loss(X_val, y_val)
        acc_test = nn.accuracy(X_test, y_test)
        loss_test = nn.loss(X_test, y_test)
        print('epoch  %d, acc_test  %.4f, loss_test %.4f' % (epoch+1, acc_test, loss_test))
        result.append([epoch+1, acc_train, loss_train[0], acc_val, loss_val[0] ,acc_test, loss_test[0]])

    file_name = os.path.join(SAVE_RESULT_PATH, "nn_result.csv")
    df = get_df_and_csv(result, columns, file_name)

    plot(df["epoch"], [df["acc_train"], df["acc_val"]], "Accuracy", "epoch", "accuracy", ["train", "validation"], os.path.join(SAVE_RESULT_PATH, "nn_accuracy.png"))
    plot(df["epoch"], [df["loss_train"], df["loss_val"]], "Loss", "epoch", "loss", ["train", "validation"], os.path.join(SAVE_RESULT_PATH, "nn_loss.png"))

    weights = nn.get_weight()
    bias = nn.get_bias()
    print('weights = ', weights, 'bias = ', bias)

    file_name = os.path.join(SAVE_RESULT_PATH, "nn_weight_bias.txt")
    print("Saving ==>> {}".format(file_name))
    with open(file_name, mode="w") as f:
        f.write("weights: {}, bias: {}\n".format(weights, bias))

if __name__ == "__main__":
    main()
