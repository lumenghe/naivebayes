"""
Implementation of Naive Bayes and test on IRIS dataset.
Comparison with sklearn implementation and Random Forest

run: python ml_test.py
"""
from __future__ import print_function
from six.moves import zip, range
import numpy as np
import random
from collections import Counter
from sklearn import metrics, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def gaussian(x, mu, sig):
    return -np.power(x - mu, 2.) / (2 * np.power(sig, 2.))

class NaiveBayes(object):
    def __init__(self, size):
        self.size = size

    def fit(self, X, Y):
        if len(X)!=len(Y):
            raise ValueError("X should have same size as Y")
        training_X = {}
        self.X_mean = {}
        self.X_std = {}
        self.P = {}
        for i in range(len(Y)):
            training_X.setdefault(Y[i], []).append(X[i])
        for i in range(self.size):
            self.X_mean[i] = np.array(training_X[i]).mean(axis=0)
            self.X_std[i] = np.array(training_X[i]).std(axis=0)
            self.P[i] = 1.0 * len(training_X[i]) / len(Y)
