# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:55:05 2017

@author: woshilk
"""

import math
import pandas as pd
import numpy as np

class LogisticRegressionClassifier(object):
    def __init__(self,alpha,n_iter):
        self.alpha = alpha
        self.iter = n_iter
    
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    
    def train(self,X,y):
        samples,features = np.shape(X)
        weights = np.ones((features,1))
        error = np.ones((features,1))
        for times in range(self.iter):
            output = self.sigmoid(X*weights)
            error = y-output
            weights = weights+self.alpha*X.transpose()*error
        
        return weights
    
    def predict(self,X,weights):
        results = self.sigmoid(X*weights)
        return results
        

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
train_X = iris.data[:80, :4]  # we only take the first two features.
train_y = iris.target[:80]

train_X = np.matrix(train_X)
train_y = np.matrix(train_y).transpose()

LR = LogisticRegressionClassifier(0.01,300)
weights = LR.train(train_X,train_y)
result = LR.predict(train_X,weights)
result = result.A


result[result>0.5] = 1
result[result<0.5] = 0
print(result)

