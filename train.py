from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math


def train_neural():
    # load data
    mnist_data = fetch_openml('mnist_784', version=1)

    # assign features and labels
    X, y = mnist_data['data'], mnist_data['target']
    print('Shape of X:', X.shape, '\n', 'Shape of y:', y.shape)

    # scale set
    X = X / 255.0

    # split into test and train into 75/25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)
    # X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    print('train after split: ', X_train.shape, '\n', 'test after split: ', X_test.shape)

    # plt.imshow(X_train.shape)

    b = np.argwhere(y_train[0:1000] == '7')
    #d = np.where(y_train[0:100] == '6')

    rows = 20
    idx = 1
    fig = plt.figure()
    for val in np.nditer(b):

        fig.add_subplot(rows, math.ceil(b.size / rows), idx)
        plt.imshow(np.resize(X_train[val], (28, 28)))
        idx += 1


    plt.show()

    # create NN model
    model = MLPClassifier(hidden_layer_sizes=(300, 150, 50), max_iter=300, verbose=True, activation='tanh',
                          solver='adam',
                          early_stopping=True, learning_rate='adaptive', random_state=1)
    # train NN
    model.fit(X_train, y_train)

    print('train score: ', model.score(X_train, y_train))
    pred = model.predict(X_test)
    print('test score: ', model.score(X_test, pred))
    #print('test score: ', model.score(X_test, y_test))

    # save model to disk
    filename = 'mnist_ML_model.sav'
    pickle.dump(model, open(filename, 'wb'))


train_neural()
# print(classification_report(y_test, pred))
