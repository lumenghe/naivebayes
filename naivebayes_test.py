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
