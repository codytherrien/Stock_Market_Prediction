import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

def load_data():
    (X_train_, y_train_), (X_test_, y_test_) = fashion_mnist.load_data()

    counter = 0
    i = 0
    X_train = np.empty((12000, 28, 28))
    y_train = np.empty((12000))
    while i < len(y_train_):
        if y_train_[i] == 5 or y_train_[i] == 7:
            X_train[counter] = X_train_[i]
            y_train[counter] = y_train_[i]
            counter += 1
        i += 1

    i = 0
    while i < len(y_train):
        if y_train[i] == 5:
            y_train[i] = 0
        else:
            y_train[i] = 1
        i += 1

    counter = 0
    i = 0
    X_test = np.empty((2000, 28, 28))
    y_test = np.empty((2000))
    while i < len(y_test_):
        if y_test_[i] == 5 or y_test_[i] == 7:
            X_test[counter] = X_test_[i]
            y_test[counter] = y_test_[i]
            counter += 1
        i += 1

    i = 0
    while i < len(y_test):
        if y_test[i] == 5:
            y_test[i] = 0
        else:
            y_test[i] = 1
        i += 1

    X_train = X_train.reshape(12000, -1)
    X_test = X_test.reshape(2000, -1)

    return X_train, y_train, X_test, y_test


def k_fold_cross_validation(X_train, y_train, k):
    if k == 1:
        return (X_train, y_train)
    split = len(y_train) // k
    total = len(y_train)
    y_train = np.vstack(y_train)
    train = np.append(X_train, y_train, axis=1)
    np.random.shuffle(train)

    k_splits = []

    for v in range(k):
        if v == k - 1:
            curr_test = train[v * split:, :]
            curr_train = train[:v * split, :]
        else:
            curr_test = train[v * split:(v + 1) * split, :]
            curr_train = train[:v * split, :]
            curr_train = np.vstack((curr_train, train[(v + 1) * split:, :]))
        k_splits.append((curr_train, curr_test))

    return k_splits


def train_multiple_log_regression(X_train, y_train, X_test, y_test, C_min, C_max, alpha=10):
    columns = ['C', 'Training Error', 'Test Error']
    df = pd.DataFrame(columns=columns)
    while C_min <= C_max:
        lrm = LogisticRegression(C=C_min).fit(X_train, y_train)
        train_error = 1 - lrm.score(X_train, y_train)
        test_error = 1 - lrm.score(X_test, y_test)
        df.loc[len(df)] = [C_min, train_error, test_error]
        C_min = C_min * alpha

    plt.figure()
    df.plot(x='C', logx=True)

    plt.ylabel('Error')
    plt.xlabel('Regularization (log Scale)')
    plt.title('Training and Test Errors vs regularization')
    plt.show()

    return df


def train_multiple_SVM(X_train, y_train, X_test, y_test, C_min, C_max, alpha=10, kernel='linear'):
    columns = ['C', 'Training Error', 'Test Error']
    df = pd.DataFrame(columns=columns)
    while C_min <= C_max:
        svm = SVC(kernel=kernel, C=C_min).fit(X_train, y_train)
        train_error = 1 - svm.score(X_train, y_train)
        test_error = 1 - svm.score(X_test, y_test)
        df.loc[len(df)] = [C_min, train_error, test_error]
        C_min = C_min * alpha

    plt.figure()
    df.plot(x='C', logx=True)

    plt.ylabel('Error')
    plt.xlabel('Regularization (log Scale)')
    plt.title('Training and Test Errors vs regularization')
    plt.show()

    return df


def train_multiple_log_regression_with_KFCV(X_train, y_train, C_min, C_max, alpha=10):
    k = math.log(C_max * alpha, alpha) - math.log(C_min, alpha)
    k = int(round(k))
    folds = k_fold_cross_validation(X_train, y_train, k)

    columns = ['C', 'Training Error', 'Test Error']
    df = pd.DataFrame(columns=columns)
    for curr_train, curr_test in folds:
        curr_X_train = curr_train[:, :-1]
        curr_y_train = curr_train[:, -1:]
        curr_X_test = curr_test[:, :-1]
        curr_y_test = curr_test[:, -1:]
        lrm = LogisticRegression(C=C_min).fit(curr_X_train, curr_y_train)
        train_error = 1 - lrm.score(curr_X_train, curr_y_train)
        test_error = 1 - lrm.score(curr_X_test, curr_y_test)
        df.loc[len(df)] = [C_min, train_error, test_error]
        C_min = C_min * alpha

    plt.figure()
    df.plot(x='C', logx=True)

    plt.ylabel('Error')
    plt.xlabel('Regularization (log Scale)')
    plt.title('Training and Test Errors vs regularization')
    plt.show()

    return df


def train_multiple_SVM_with_KFCV(X_train, y_train, C_min, C_max, alpha=10, kernel='linear'):
    k = math.log(C_max * alpha, alpha) - math.log(C_min, alpha)
    k = int(round(k))
    folds = k_fold_cross_validation(X_train, y_train, k)

    columns = ['C', 'Training Error', 'Test Error']
    df = pd.DataFrame(columns=columns)
    for curr_train, curr_test in folds:
        curr_X_train = curr_train[:, :-1]
        curr_y_train = curr_train[:, -1:]
        curr_X_test = curr_test[:, :-1]
        curr_y_test = curr_test[:, -1:]
        svm = SVC(kernel=kernel, C=C_min).fit(curr_X_train, curr_y_train)
        train_error = 1 - svm.score(curr_X_train, curr_y_train)
        test_error = 1 - svm.score(curr_X_test, curr_y_test)
        df.loc[len(df)] = [C_min, train_error, test_error]
        C_min = C_min * alpha

    plt.figure()
    df.plot(x='C', logx=True)

    plt.ylabel('Error')
    plt.xlabel('Regularization (log Scale)')
    plt.title('Training and Test Error vs regularization')
    plt.show()

    return df


def train_multiple_Gaussian_SVM(X_train, y_train, X_test, y_test, C_min, C_max, gamma_min, gamma_max, alpha=10,
                                kernel='rbf'):
    test_min = 1
    C_iter = 0
    columns = ['C', 'Gamma', 'Training Error', 'Test Error']
    df = pd.DataFrame(columns=columns)

    k = math.log(C_max * alpha, alpha) - math.log(C_min, alpha)
    k = int(round(k))
    folds = k_fold_cross_validation(X_train, y_train, k)

    while gamma_min <= gamma_max:
        C_curr = C_min
        test_min = 1
        for curr_train, curr_test in folds:
            curr_X_train = curr_train[:, :-1]
            curr_y_train = curr_train[:, -1:]
            curr_X_test = curr_test[:, :-1]
            curr_y_test = curr_test[:, -1:]
            svm = SVC(kernel=kernel, C=C_curr, gamma=gamma_min).fit(curr_X_train, curr_y_train)
            train_error = 1 - svm.score(curr_X_train, curr_y_train)
            test_error = 1 - svm.score(curr_X_test, curr_y_test)
            if test_error < test_min:
                test_min = test_error
                C_iter = C_curr
            C_curr = C_curr * alpha

        svm = SVC(kernel=kernel, C=C_iter, gamma=gamma_min).fit(X_train, y_train)
        train_error = 1 - svm.score(X_train, y_train)
        test_error = 1 - svm.score(X_test, y_test)
        df.loc[len(df)] = [C_iter, gamma_min, train_error, test_error]
        gamma_min = gamma_min * alpha

    plt.figure()
    df.plot(x='Gamma', y=['Training Error', 'Test Error'], logx=True)

    plt.ylabel('Error')
    plt.xlabel('Regularization (log Scale)')
    plt.title('Training and Test Errors vs Scale')
    plt.show()

    return df