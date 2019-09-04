import numpy as np
from sklearn import datasets


def accuracy(preds, labels):
    """returns the accuray of the prediction

    Arguments:
        preds {numpy array} -- predictions
        labels {numpy array} -- ground truth

    Returns:
        float -- accuracy in percentage
    """
    return np.mean(preds == labels)*100


def confusion_matirx(preds, labels, total_classes):
    """returns the confusion matrix of size total_class X total_class

    Arguments:
        preds {numpy array} -- predictions
        labels {numpy array} -- ground truth
        total_classes {int} -- total number of classes

    Returns:
        numpy 2 dimensional array -- confusion matrix
    """
    matrix = np.zeros((total_classes, total_classes))
    for i in range(len(preds)):
        matrix[preds[i], labels[i]] += 1
    return matrix


def precision_and_recall(preds, labels, total_classes):
    """claculate precision and recall scores for different classes

    Arguments:
        preds {numpy array} -- predictions
        labels {numpy array} -- ground truth
        total_classes {int} -- total number of classes

    Returns:
        (numpy array, numpy array) -- precision and recall scores per class
    """
    c_mat = confusion_matirx(preds, labels, total_classes)
    recall = []
    precision = []

    for i in range(total_classes):
        TP = float(c_mat[i][i])
        TP_FP = np.sum(c_mat[i, :])
        TP_FN = np.sum(c_mat[:, i])

        r = "NA" if TP_FN == 0 else TP / TP_FN
        p = "NA" if TP_FN == 0 else TP / TP_FP

        recall.append(r)
        precision.append(p)

    return precision, recall


def f1score(preds, labels, total_classes):
    """f1-score of the predictions and the ground truth

    Arguments:
        preds {numpy array} -- predictions
        labels {numpy array} -- ground truth
        total_classes {int} -- total number of classes

    Returns:
        float -- f1 score percentage
    """
    p, r = precision_and_recall(preds, labels, total_classes)

    precision = 0
    recall = 0

    count = 0.0
    for x in p:
        if x != "NA":
            count += 1
            precision += x
    precision /= count

    count = 0.0
    for x in r:
        if x != "NA":
            count += 1
            recall += x
    recall /= count

    return (2*precision*recall/(precision+recall))*100


def get_iris_data(
        ratio_train=0.8
        ):
    """returns the iris data splitted into train and test

    Keyword Arguments:
        ratio_train {float} -- ratio of the train data from the total IRIS
            Data, 1- ratio_train is the size of test data (default: {0.8})
    """
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.reshape(iris.target.shape[0], 1)
    data = np.append(data, target, axis=1)
    np.random.shuffle(data)

    n = int(iris.data.shape[0] * ratio_train)

    train = data[:n]
    test = data[n:]

    feature_names = iris.feature_names
    return feature_names, train, test


def get_wine_data(
        ratio_train=0.8
        ):
    """returns the wine data splitted into train and test

    Keyword Arguments:
        ratio_train {float} -- ratio of the train data from the total wine
            Data, 1- ratio_train is the size of test data (default: {0.8})
    """
    wine = datasets.load_wine()
    data = wine.data
    target = wine.target.reshape(wine.target.shape[0], 1)
    data = np.append(data, target, axis=1)
    np.random.shuffle(data)

    n = int(wine.data.shape[0] * ratio_train)

    train = data[:n]
    test = data[n:]

    feature_names = wine.feature_names
    return feature_names, train, test


def unique_values(df, col):
    return np.unique(df[:, col])


def class_counts(df):
    """frequencies accross different classes in the numpy array

    Arguments:
        df {numpy array} -- data with last column as the target variable

    Returns:
        class labels(numpy array) , class frequencies(numpy array) -- class
        labels and the corresponding class frequencies
    """
    return np.unique(df[:, -1], return_counts=True)


def is_numeric(value):
    """checks is the valus is numeric

    Arguments:
        value {int/float/char/bool} -- any value

    Returns:
        True/False -- true is the value is int or float else false
    """
    return isinstance(value, int) or isinstance(value, float)


def entropy(df):
    """calculates the entropy of the numpy array considering that the last
    column represents the gorund truth labels

    Arguments:
        df {numpy array 2 dimensional} -- numpy array of numpy array

    Returns:
        flaot -- entropy of the data
    """
    _, counts = class_counts(df)
    probs = counts / float(df.shape[0])
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def info_gain(left, right, current_entrophy):
    """calculates the information gain based on the current entrophy and the
    split that has given

    Arguments:
        left {numpy array of numpy array} -- data on left split
        right {numpy array of numpy array} -- data on the right split
        current_entrophy {float} -- entropy of the current node

    Returns:
        float -- information gain due to the split
    """
    p = float(left.shape[0]) / float(left.shape[0] + right.shape[0])
    return current_entrophy - (p*entropy(left)) - ((1-p)*entropy(right))
