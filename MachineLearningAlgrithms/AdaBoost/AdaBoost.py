# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


class AdaBoost(object):
    def __init__(self,data_matrix,labels,iter_nums):
        self.X = np.matrix(data_matrix)
        self.y = np.matrix(labels)
        samples = np.shape(data_matrix)[0]    
        self.D = np.mat(np.ones((samples,1))/samples)
        self.iter = range(iter_nums)
    
    def stumpClassfily(self,data_matrix,dimen,thresh_val,thresh_ineq):
        rest_arr = np.ones((np.shape(data_matrix)[0],1))
        if thresh_ineq is 'lt':
            rest_arr[data_matrix[:,dimen]<=thresh_val] = -1.0
        else:
            rest_arr[data_matrix[:,dimen]>thresh_val] = -1.0
        return rest_arr
    
    def buildStump(self):
        samples,features = np.shape(self.X)
        num_steps = 10
        best_stump = {}
        best_class_estimate = np.mat(np.zeros((samples,1)))
        min_error = np.inf
        
        for i in range(features):
            range_min = self.X[:,i].min()
            range_max = self.X[:,i].max()
            step_size = (range_max-range_min)/num_steps
            for j in range(-1,num_steps+1):
                for inequal in ['lt','gt']:
                    thresh_val = (range_min+float(j)*step_size)
                    predict_val = self.stumpClassfily(self.X,i,thresh_val,inequal)
                    error_matrix = np.matrix(np.ones((samples,1)))
                    error_matrix[predict_val==self.y.T] = 0
                    weight_error = self.D.T * error_matrix
                    
                    if weight_error < min_error:
                        min_error = weight_error
                        best_class_estimate = predict_val.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        return best_stump,min_error,best_class_estimate
    
    
    def train(self):
        weak_class_arr = []
        samples = np.shape(self.X)[0]
        aggravete_class_est = np.mat(np.zeros((samples,1)))
        for rounds in self.iter:
            best_stump,error,class_est = self.buildStump()
            alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
            best_stump['alpha'] = alpha
            weak_class_arr.append(best_stump)
            expon_loss_func = np.multiply(-1*alpha*self.y.T,class_est)
            self.D = np.multiply(self.D,np.exp(expon_loss_func))
            self.D = self.D/self.D.sum()
            aggravete_class_est += alpha*class_est
            agg_errors_matrix = np.multiply(np.sign(aggravete_class_est)!=self.y.T,
                                            np.ones((samples,1)))
            error_rate = agg_errors_matrix.sum()/samples
            if error_rate is 0.0:break
        
        return weak_class_arr
    

class Predcitor(AdaBoost):
    def __init__(self,data_matrix,classifyer_set):
        self.X = data_matrix
        self.classifyers = classifyer_set
    
    
    def predict(self):
        samples = np.shape(self.X)[0]
        agg_class_est = np.mat(np.zeros((samples,1)))
        for classifyer in self.classifyers:
            class_est = self.stumpClassfily(self.X,classifyer['dim'],
                                            classifyer['thresh'],
                                            classifyer['ineq'])
            agg_class_est += classifyer['alpha']*class_est
        return np.sign(agg_class_est)
    
    
data_matrix,labels = loadSimpData()
AdaBoost = AdaBoost(data_matrix,labels,9)
classifyers_set = AdaBoost.train()

test_X = np.matrix([[5,5],[0,0]])
predicter = Predcitor(test_X,classifyers_set)
print (predicter.predict())
            
            
            
            
            
        
    
    
