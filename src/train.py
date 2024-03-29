import math
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def display_graph(x, y, num):
    """
    Used for debugging/learning.  Displays 100 resized images of MNIST target/true number specified by the parameter num
    :param x: features array/matrix, 784 pixel (columns) by 60000 examples (rows)
    :param y: target array/matrix, 1 number (column) by 60000 examples (rows)
    :param num: number of interest you want to see displayed
    :return: outputs matplotlib window object
    """

    number = num
    indices = np.argwhere(y.to_numpy() == number)[:100]
    rows = 10
    idx = 1
    fig = plt.figure()
    for val in np.nditer(indices):
        fig.add_subplot(rows, math.ceil(indices.size / rows), idx)
        plt.imshow(np.resize(x[val.item():val.item() + 1], (28, 28)))
        idx += 1

    plt.show()


def get_data():
    """
    User can get data from openML.org using the id of the dataset or load csv file
    :return: dictionary object, features and target as keys
    """
    response = input("Do you want to load data from openML.org (Y/N)?\n")
    assert response in ['Y', 'N']
    datafile = "data.csv"
    file_path = os.path.join("../data", datafile)
    results = dict()

    if response == 'Y':
        print('Loading Data from openML...')
        # load data from openML.org name='mnist_784'
        mnist_data = fetch_openml(data_id=554)

        data = mnist_data['data']
        series = mnist_data['target']
        results['features'] = mnist_data['data']
        results['target'] = mnist_data['target']

        # merge series into df and write to csv
        df = data.merge(series, left_index=True, right_index=True)
        print('Writing data to disk...')
        df.to_csv(file_path)
        print('Complete!')
    else:
        print('Loading Data from {}...'.format(file_path))
        df = pd.read_csv(file_path, dtype={'class': 'object'})
        features = df.filter(like='pixel', axis=1)
        target = pd.Series(df['class'], dtype='category')

        results['features'] = features
        results['target'] = target

    return results


def train_neural(data, metrics=True, **kwargs):
    """
    Function that loads MNIST data and trains MLP
    :param data: dictionary with features & target as keys
    :param metrics: bool whether you want to display classification metrics
    :return: model object
    """

    # assign features and labels
    X, y = data['features'], data['target']
    # print('Shape of X:', X.shape, '\n', 'Shape of y:', y.shape)

    # scale data
    # set to range [0,1] like MinMaxScaler
    X = X / 255.0

    # split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

    # use for debugging
    # display_graph(X_train, y_train, '7')

    # instantiate estimator (Multi Layer Perceptron Classifier)
    model = MLPClassifier(hidden_layer_sizes=(100, 50, 20), **kwargs)
    # fit with data
    print('Training Model...')
    model.fit(X_train.values, y_train)

    if metrics:
        # display metrics for classification
        pred = model.predict(X_test)
        print(classification_report(y_test, pred))

    return model


def save_model(model):
    """
    function that just saves model to disk using pickle
    :param model: model object from sklearn
    :return: N/A
    """
    # save model to disk
    print('Saving Model to disk...')
    filename = 'mnist_ML_model.sav'
    filepath = os.path.join('../model', filename)
    pickle.dump(model, open(filepath, 'wb'))


def main():
    results = get_data()
    model = train_neural(results, metrics=False, activation='relu', solver='adam', early_stopping=True, max_iter=20,
                         learning_rate='adaptive', random_state=1, verbose=True)
    save_model(model)
    print('Complete!')


if __name__ == "__main__":
    main()
