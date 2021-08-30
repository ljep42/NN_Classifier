from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

def display_graph(x, y, num):
    '''
    Used for debugging/learning.  Displays 100 resized images of MNIST target/true number specified by the parameter num
    :param x: features array/matrix, 784 pixel (columns) by 60000 examples (rows)
    :param y: target array/matrix, 1 number (column) by 60000 examples (rows)
    :param num: number of interest you want to see displayed
    :return: outputs matplotlib window object
    '''

    number = num
    indices = np.argwhere(y.to_numpy() == number)[:100]
    rows = 10
    idx = 1
    fig = plt.figure()
    for val in np.nditer(indices):
        fig.add_subplot(rows, math.ceil(indices.size / rows), idx)
        plt.imshow(np.resize(x[val.item():val.item()+1], (28, 28)))
        idx += 1

    plt.show()

def train_neural():
    '''
    Function that loads MNIST data and trains model prints metrics and saves model as pickle file
    :return: pickle file save file in project folder path
    '''

    # load data from openML.org
    mnist_data = fetch_openml('mnist_784', version=1)

    # assign features and labels
    X, y = mnist_data['data'], mnist_data['target']
    #print('Shape of X:', X.shape, '\n', 'Shape of y:', y.shape)

    # scale set to range [0,1] like MinMaxScaler
    X = X / 255.0

    # split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

    # use for debugging
    #display_graph(X_train, y_train, '7')

    # instantiate estimator (Multi Layer Perceptron Classifier)
    model = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=100, verbose=False, activation='relu',
                          solver='adam', early_stopping=True, learning_rate='adaptive', random_state=1)
    # fit with data
    print('Training...')
    model.fit(X_train, y_train)

    # display metrics
    print('train score: ', model.score(X_train, y_train))
    pred = model.predict(X_test)
    print('test score: ', model.score(X_test, pred))
    print(classification_report(y_test, pred))

    # save model to disk
    filename = 'mnist_ML_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def main():
    train_neural()
    print('Complete!')

if __name__ == "__main__":
    main()


