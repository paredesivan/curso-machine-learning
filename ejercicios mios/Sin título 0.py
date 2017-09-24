# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:26:57 2017

@author: ivan
"""

from sklearn import datasets
boston = datasets.load_boston()
boston.DESCR
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(boston.data, boston.target)
for (feature, coef) in zip(boston.feature_names, lr.coef_):
    print('{:>7}: {: 9.5f}'.format(feature, coef))


import matplotlib.pyplot as plt
import numpy as np
def plot_feature(feature):
    f = (boston.feature_names == feature)
    plt.scatter(boston.data[:,f], boston.target, c='b', alpha=0.3)
    plt.plot(boston.data[:,f], boston.data[:,f]*lr.coef_[f] + lr.intercept_, 'k')
    plt.legend(['Predicted value', 'Actual value'])
    plt.xlabel(feature)
    plt.ylabel("Median value in $1000's")
plot_feature('AGE')

predictions = lr.predict(boston.data)
f, ax = plt.subplots(1)
ax.hist(boston.target - predictions, bins=50, alpha=0.7)
ax.set_title('Histograma de residuales')
ax.text(0.95, 0.90, 'Media de residuales: {:.3e}'.format(np.mean(boston.target - predictions)),
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')