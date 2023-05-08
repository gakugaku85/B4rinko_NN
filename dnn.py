import argparse
import ast
import gzip
import os
import pickle
import random
from ipaddress import v4_int_to_packed
from time import timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from urllib3 import request

# from tkinter.tix import X_REGION

seed_value = 0
np.random.seed(seed_value)
random.seed(seed_value)

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'x_train':'train-images.idx3-ubyte',
    't_train':'train-labels.idx1-ubyte',
    'x_test':'t10k-images.idx3-ubyte',
    't_test':'t10k-labels.idx1-ubyte'
}

DATA_PATH = './data/'

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

    for v in key_file.values():
        file_path = DATA_PATH + v
        print("Loading >> {}".format(file_path))
        request.urlretrieve(url_base + v, file_path)

def get_timestamp():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')

SAVE_RESULT_PATH = "./result/" + get_timestamp() + "/"

if not os.path.isdir(SAVE_RESULT_PATH):
    os.makedirs(SAVE_RESULT_PATH)


def load_label(file_name):
    file_path = DATA_PATH + file_name
    with open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
            # 最初の８バイト分はデータ本体ではないので飛ばす
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def load_image(file_name):
    file_path = DATA_PATH + file_name
    with open(file_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        # 画像本体の方は16バイト分飛ばす必要がある
    return images

def convert_into_numpy(key_file):
    dataset = {}

    dataset['x_train'] = load_image(key_file['x_train'])
    dataset['t_train'] = load_label(key_file['t_train'])
    dataset['x_test']  = load_image(key_file['x_test'])
    dataset['t_test']  = load_label(key_file['t_test'])

    return dataset

def load_mnist():
    # mnistを読み込みNumPy配列として出力する
    dataset = convert_into_numpy(key_file)
    dataset['x_train'] = dataset['x_train'].astype(np.float32) # データ型を`float32`型に指定しておく
    dataset['x_test'] = dataset['x_test'].astype(np.float32)
    dataset['x_train'] /= 255.0
    dataset['x_test'] /= 255.0 # 簡単な標準化
    dataset['x_train'] = dataset['x_train'].reshape(60000, 28*28)
    dataset['x_test']  = dataset['x_test'].reshape(10000, 28*28)
    return dataset


def sigmoid(x):  # シグモイド関数
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x) # 最大値
    exp_a = np.exp(i-c) # 分子:オーバーフロー対策
    for i in range(len(exp_a)):
        print(i)
        input()
        sum_exp_a = np.sum(exp_a) # 分母
    y = exp_a / sum_exp_a # 式(3.10)
    return y

def inner_product(X, w, b):  # ここは内積とバイアスを足し合わせる
    return np.dot(X, w) + b

def activation(X, w, b):
    return sigmoid(inner_product(X, w, b))

def cross_entropy_error(y, t):
    delta = 1e-7 # log(0)にならないように微小値を足しておく
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

class DeepNeuralNetwork:

    def __init__(self, shape_list=[784, 100, 10], eta=0.01):
        self.weight_list, self.bias_list = self.make_params(shape_list)
        self.eta = eta

    def make_params(self, shape_list): # shape_list = [784, 100, 10]のように層ごとにニューロンの数を配列にしたものを入力する
        weight_list = []
        bias_list = []
        for i in range(len(shape_list)-1):
            weight = np.random.randn(shape_list[i], shape_list[i+1]) # 標準正規分布に従った乱数を初期値とする
            bias = np.ones(shape_list[i+1])/10.0 # 初期値はすべて0.1にする
            weight_list.append(weight)
            bias_list.append(bias)
        return weight_list, bias_list

    def calculate(self, X, t):

        val_list = {}
        a_1 = inner_product(X, self.weight_list[0], self.bias_list[0]) # (N, 1000)
        y_1 = sigmoid(a_1)
        a_2 = inner_product(y_1, self.weight_list[1], self.bias_list[1]) # (N, 10)
        y_2 = sigmoid(a_2)
        # y_2 /= np.sum(y_2, axis=1, keepdims=True) # ここで簡単な正規化をはさむ これを消してみた
        S = 1/(2*len(y_2))*(y_2 - t)**2 
        L = np.sum(S)
        # L = cross_entropy_error(y_2, t)
        val_list['a_1'] = a_1
        val_list['y_1'] = y_1
        val_list['a_2'] = a_2
        val_list['y_2'] = y_2
        val_list['S'] = S
        val_list['L'] = L

        return val_list

    def predict(self, X, t):
        val_list = self.calculate(X, t)
        y_2 = val_list['y_2']
        result = np.zeros_like(y_2)
        for i in range(y_2.shape[0]): # サンプル数にあたる
            result[i, np.argmax(y_2[i])] = 1
        # print(result.shape)
        return result

    def predict_confu(self, X, t):
        val_list = self.calculate(X, t)
        y_2 = val_list['y_2']
        result = []
        for i in range(y_2.shape[0]): # サンプル数にあたる
            result.append(np.argmax(y_2[i]))
        # print(result.shape)
        return result

    def accuracy(self, X, t):
        pre = self.predict(X, t)
        result = np.where(np.argmax(t, axis=1)==np.argmax(pre, axis=1), 1, 0)
        acc = np.mean(result)
        return acc

    def loss(self, X, t):
        L = self.calculate(X, t)['L']
        return L

    def update(self, X, t): # etaは学習率。ここでパラメータの更新を行う
        val_list = self.calculate(X, t)
        a_1 = val_list['a_1']
        y_1 = val_list['y_1']
        a_2 = val_list['a_2']
        y_2 = val_list['y_2']
        # S = val_list['S']
        L = val_list['L']

        dL_dS = 1.0
        dS_dy_2 = 1/X.shape[0]*(y_2 - t)
        dy_2_da_2 = y_2*(1.0 - y_2)
        da_2_dw_2 = np.transpose(y_1)
        da_2_db_2 = 1.0
        da_2_dy_1 = np.transpose(self.weight_list[1])
        dy_1_da_1 = y_1 * (1 - y_1)
        da_1_dw_1 = np.transpose(X)
        da_1_db_1 = 1.0

        # ここからパラメータの更新を行っていく。
        dL_da_2 =  dL_dS * dS_dy_2 * dy_2_da_2
        self.bias_list[1] -= self.eta*np.sum(dL_da_2 * da_2_db_2, axis=0)
        self.weight_list[1] -= self.eta*np.dot(da_2_dw_2, dL_da_2)
        dL_dy_1 = np.dot(dL_da_2, da_2_dy_1)
        dL_da_1 = dL_dy_1 * dy_1_da_1
        self.bias_list[0] -= self.eta*np.sum(dL_da_1 * da_1_db_1, axis=0)
        self.weight_list[0] -= self.eta*np.dot(da_1_dw_1, dL_da_1)

def get_df_and_csv(result, columns, file_name):

    df = pd.DataFrame(result, columns=columns)

    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)

    return df

def visualize_data(dataset, file_name):
    plt.figure(figsize=(10, 8))

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(dataset['x_train'][i, :].reshape(28, 28))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)

def plot(x, y, title, x_label, y_label, label,file_name):

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

def max_vali_cnf(dnn, X_test, t_test, test_num):
    cnf_labels = [0,1,2,3,4,5,6,7,8,9]
    predict_X = dnn.predict_confu(X_test, t_test)
    true_t = []
    for i in range(test_num):
        true_t.append(np.argmax(t_test[i]))
    cm = confusion_matrix(true_t, predict_X, labels=cnf_labels)
    df = pd.DataFrame(cm)
    file_name = os.path.join(SAVE_RESULT_PATH, "cnf_mat.csv")
    df.to_csv(file_name)
    print("Saving ==>> {}".format(file_name))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=str, default='10',
                        help='batch size for training')
    parser.add_argument('-eta', '--eta', type=str, default='0.1',
                        help='learning rate for training')
    parser.add_argument('-e', '--epochs', type=str, default='100',
                        help='Number of epochs for training')
    parser.add_argument('-shape_list', '--shape_list', type=str, default='[784, 100, 10]',
                        help='network shape')
    parser.add_argument('-train_num', '--train_num', type=str, default='50000',
                        help='Number of training data')
    parser.add_argument('-vali_num', '--vali_num', type=str, default='10000',
                        help='Number of validation data')
    parser.add_argument('-test_num', '--test_num', type=str, default='10000',
                        help='Number of test data')
    

    # parse configs
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    eta = float(args.eta)
    epochs = int(args.epochs)
    shape_list = ast.literal_eval(args.shape_list)

    printargs = vars(args)
    print("Arguments are:")
    for k, v in printargs.items():
        print("\t{}: {}".format(k, v))

    output_args = os.path.join(SAVE_RESULT_PATH, "args.txt")
    with open(output_args, 'w') as f:
        for k, v in printargs.items():
            f.write("{}: {}\n".format(k, v))

    dnn = DeepNeuralNetwork(shape_list, eta)

    dataset = load_mnist()

    file_name = os.path.join(SAVE_RESULT_PATH, "mnist.png")
    visualize_data(dataset, file_name)

    X_train_val, t_train_val, X_test, t_test = dataset["x_train"], dataset["t_train"], dataset["x_test"], dataset["t_test"]

    train_num = int(args.train_num)
    test_num = int(args.test_num)
    vali_num = int(args.vali_num)

    X_train = X_train_val[:train_num, :].copy()
    t_train = t_train_val[:train_num, :].copy()
    X_val = X_train_val[train_num:train_num + vali_num, :].copy()
    t_val = t_train_val[train_num:train_num + vali_num, :].copy()
    X_test = X_test[:test_num, :].copy()
    t_test = t_test[:test_num, :].copy()

    columns = ["Epoch", "acc_train", "loss_train", "acc_vali", "loss_vali"]

    results = []
    dnn_list = []
    acc_vali_list = []

    for epoch in range(epochs):

        acc_list_train = []
        loss_list_train = []

        batch_list = list(range(train_num))
        
        for i in range(train_num//batch_size):
            random.shuffle(batch_list)
            ra = [batch_list.pop(0) for i in range(batch_size)]
            
            x_batch, t_batch = X_train[ra,:], t_train[ra,:]
            acc_val_tr = dnn.accuracy(x_batch, t_batch)
            loss_val_tr = dnn.loss(x_batch, t_batch)
            acc_list_train.append(acc_val_tr)
            loss_list_train.append(loss_val_tr)
            dnn.update(x_batch, t_batch)

        acc_train = np.mean(acc_list_train)   # 精度は平均で求める
        loss_train = np.mean(loss_list_train) # 損失は平均で求める。

        acc_vali = dnn.accuracy(X_val, t_val)
        loss_vali = dnn.loss(X_val, t_val)

        # print("Epoch: %d, Accuracy: %f, Loss: %f" % (epoch+1, acc_vali, loss_vali))
        print("Epoch: %d, train_Accuracy: %f, train_Loss: %f, vali_Accuracy: %f, vali_Loss: %f" % (epoch+1, acc_train, loss_train, acc_vali, loss_vali))
        results.append([epoch+1, acc_train, loss_train, acc_vali, loss_vali])

        acc_vali_list.append(acc_vali)
        dnn_list.append(dnn)
    
    vali_max_index = np.argmax(acc_vali_list)
    max_dnn = dnn_list[vali_max_index]
    acc_test = max_dnn.accuracy(X_test, t_test)
    loss_test = max_dnn.loss(X_test, t_test)
    print("MAX_Epoch: %d, test_Accuracy: %f, test_Loss: %f" % (vali_max_index+1, acc_test, loss_test))
    file_name = os.path.join(SAVE_RESULT_PATH, "MAX_EPOCH.txt")
    f = open(file_name, "w")
    f.write("MAX_Epoch: %d, test_Accuracy: %f, test_Loss: %f" % (vali_max_index+1, acc_test, loss_test))
    f.close()
    max_vali_cnf(max_dnn, X_test, t_test, test_num)

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_result.csv")
    df = get_df_and_csv(results, columns, file_name)

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_accuracy.png")
    plot(df["Epoch"], [df["acc_train"], df["acc_vali"]], "Accuracy", "epoch", "accuracy",["train", "validation"], file_name)

    file_name = os.path.join(SAVE_RESULT_PATH, "dnn_loss.png")
    plot(df["Epoch"], [df["loss_train"], df["loss_vali"]], "Loss", "epoch", "loss",["train", "validation"], file_name)


if __name__ == "__main__":
    main()
