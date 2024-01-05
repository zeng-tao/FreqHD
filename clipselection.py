# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import random

random.seed(123)
np.random.seed(123)

def softmax(x):
    expfact = 10    # used to set the level of suppression for outliers
    x_exp = np.exp(expfact*x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s

def clip_select(X, y):
    X = np.array(X) / max(X)
    y = np.array(y) / max(y)
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)

    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    rawresidual = y - ransac.predict(X)
    absresidual = abs(rawresidual)
    softmaxresidual = softmax(absresidual)

    return softmaxresidual
