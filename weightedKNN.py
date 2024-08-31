import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier


def weightedKNN(X_train, y_train, X_test, sigma):
    # compute weights
    dist = scipy.spatial.distance.cdist(X_test, X_train, metric='euclidean')
    weight = np.exp(-dist**2/(2*sigma**2))

    # create prediction vector
    pred = np.zeros(X_test.shape[0])

    for i in range(len(X_test)):
        value = {}
        for j in range(len(X_train)):
            # determine label based on y
            label = y_train[j]

            # determine weight at that value
            vote = weight[i][j]

            # if that label, add weight of vote to value
            if label in value:
                value[label] += vote
            else:
                value[label] = vote

        # determine prediction based on max value
        pred[i] = max(value, key=value.get)

    # return
    return pred
