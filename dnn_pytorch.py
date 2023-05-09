import argparse
import os
import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from urllib3 import request
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


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

def get_timestamp():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')

SAVE_RESULT_PATH = "./result/dnn_" + get_timestamp() + "/"

if not os.path.isdir(SAVE_RESULT_PATH):
    os.makedirs(SAVE_RESULT_PATH)

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

    for v in key_file.values():
        file_path = DATA_PATH + v
        print("Loading >> {}".format(file_path))
        request.urlretrieve(url_base + v, file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 28x28の画像を1次元に変換
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def make_train_dataset(size):
    x_train_path = DATA_PATH + key_file['x_train']
    t_train_path = DATA_PATH + key_file['t_train']

    x_train = np.fromfile(x_train_path, np.uint8)
    x_train = x_train[16:]
    x_train = x_train.reshape(-1, 784)
    x_train = x_train[:size]

    t_train = np.fromfile(t_train_path, np.uint8)
    t_train = t_train[8:]
    t_train = t_train[:size]

    return x_train, t_train

def make_test_dataset(size):
    x_test_path = DATA_PATH + key_file['x_test']
    t_test_path = DATA_PATH + key_file['t_test']

    x_test = np.fromfile(x_test_path, np.uint8)
    x_test = x_test[16:]
    x_test = x_test.reshape(-1, 784)
    x_test = x_test[:size]

    t_test = np.fromfile(t_test_path, np.uint8)
    t_test = t_test[8:]
    t_test = t_test[:size]

    return x_test, t_test

def make_validation_dataset(validation_size, x_train, t_train):
    x_train, t_train = shuffle_dataset(x_train, t_train)

    x_validation = x_train[:validation_size]
    t_validation = t_train[:validation_size]

    x_train = x_train[validation_size:]
    t_train = t_train[validation_size:]

    return x_train, t_train, x_validation, t_validation

def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation]
    t = t[permutation]

    return x, t

def show_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig('loss.png')

def show_accuracy(accuracy_list):
    plt.plot(accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.savefig('accuracy.png')

def plot_result(path, images, labels, predicts):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title('label:{}, predict:{}'.format(labels[i], predicts[i]))
        plt.axis('off')
    plt.savefig(path + 'result.png')

def calculate_loss_and_accuracy(model, criterion, x, t):
    model.eval()
    with torch.no_grad():
        y = model(x)
        loss = criterion(y, t)
        correct = (y.max(1)[1] == t).sum().item()
        data_num = len(x)
        accuracy = correct / data_num
    model.train()
    return loss, accuracy

def output_confusion_matrix_csv(path, t, y):
    confusion_matrix = metrics.confusion_matrix(t, y)
    df = pd.DataFrame(confusion_matrix)
    df.to_csv(path+'confusion_matrix.csv')

def plot_loss_train_val(path, t_loss, v_loss):
    plt.plot(t_loss, label='train_loss')
    plt.plot(v_loss, label='validation_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.savefig(path + 'loss.png')
    plt.cla()

def plot_accuracy_train_val(path, t_acc, v_acc):
    plt.plot(t_acc, label='train_accuracy')
    plt.plot(v_acc, label='validation_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(path + 'accuracy.png')
    plt.cla()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=1000)
    args = parser.parse_args()

    printargs = vars(args)
    print('---args---')
    for k, v in printargs.items():
        print('%s: %s' % (k, v))
    print('---args---')

    output_args = os.path.join(SAVE_RESULT_PATH, "args.txt")
    with open(output_args, 'w') as f:
        for k, v in printargs.items():
            f.write("{}: {}\n".format(k, v))


    batch_size = args.batch_size
    epochs = args.epoch
    lr = args.lr
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size

    x_train, t_train = make_train_dataset(train_size + val_size)
    x_test, t_test = make_test_dataset(test_size)

    x_train, t_train, x_validation, t_validation = make_validation_dataset(val_size, x_train, t_train)

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).long()
    x_validation = torch.from_numpy(x_validation).float()
    t_validation = torch.from_numpy(t_validation).long()
    x_test = torch.from_numpy(x_test).float()
    t_test = torch.from_numpy(t_test).long()

    print('x_train.shape:', x_train.shape)
    print('t_train.shape:', t_train.shape)
    print('x_validation.shape:', x_validation.shape)
    print('t_validation.shape:', t_validation.shape)
    print('x_test.shape:', x_test.shape)
    print('t_test.shape:', t_test.shape)

    x_train = x_train.to(device)
    t_train = t_train.to(device)
    x_validation = x_validation.to(device)
    t_validation = t_validation.to(device)
    x_test = x_test.to(device)
    t_test = t_test.to(device)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    iteration = x_train.shape[0] // batch_size

    train_loss_list = []
    train_accuracy_list = []
    validation_loss_list = []
    validation_accuracy_list = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-------------')

        for i in range(iteration):
            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
            t_batch = t_train[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()

            outputs = net(x_batch)
            loss = criterion(outputs, t_batch)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print('Batch {} / {} Loss {}'.format(i + 1, iteration, loss.item()))

        train_loss, train_accuracy = calculate_loss_and_accuracy(net, criterion, x_train, t_train)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        print('Train Loss: {} Accuracy: {}'.format(train_loss, train_accuracy))

        validation_loss, validation_accuracy = calculate_loss_and_accuracy(net, criterion, x_validation, t_validation)
        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)
        print('Validation Loss: {} Accuracy: {}'.format(validation_loss, validation_accuracy))

    plot_loss_train_val(SAVE_RESULT_PATH, train_loss_list, validation_loss_list)
    plot_accuracy_train_val(SAVE_RESULT_PATH, train_accuracy_list, validation_accuracy_list)

    test_loss, test_accuracy = calculate_loss_and_accuracy(net, criterion, x_test, t_test)
    print('Test Loss: {} Accuracy: {}'.format(test_loss, test_accuracy))

    outputs = net(x_test)
    _, predicts = torch.max(outputs, 1)
    output_confusion_matrix_csv(SAVE_RESULT_PATH, t_test.cpu().numpy(), predicts.cpu().numpy())

    outputs = net(x_test[:25])
    _, predicts = torch.max(outputs, 1)
    plot_result(SAVE_RESULT_PATH, x_test[:25].cpu().numpy(), t_test[:25].cpu().numpy(), predicts.cpu().numpy())
    
if __name__ == '__main__':
    main()
    