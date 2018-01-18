# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:32:55 2018

@author: Stu.zhouyc
"""

import math
import pandas as pd
import operator

class DecisionTree(object):
    def __init__(self,data,lable):
        self.data_set = data
        self.lable = lable
    
    #input: one parameter
    #data type: pandas DataFrame; content: [fea1,fea2,...,fea,lable]
    #out put: entropy
    def calcutaleEnt(self,data_set):
        num_samples = len(data_set)
        lable_list = set(list(data_set[self.lable]))
        count_dic = data_set.groupby(by=self.lable).count().to_dict()
        main_key_list = list(count_dic.keys())
        
        cal_dic = {}
        for item in lable_list:
            count_dic[item] = 0
        for main_key in main_key_list:
            for lable in lable_list:
                cal_dic[lable] = count_dic[main_key][lable]
        
        ent = 0.0
        for key in cal_dic.keys():
            prob = float(cal_dic[key])/num_samples
            ent-=prob*math.log(prob,2)
        return ent
        
    
    def splitData(self,data_set,feature,value):
        columns = list(data_set.columns)
        columns.remove(feature)
        restData = data_set[data_set[feature]==value]
        restData = restData[columns]
        
        return restData
    
    def getFeatureProprity(self,data_set):
        num_smaples = data_set.shape[0]
        return num_smaples
        
    def inforGain(self,data_set):
        base_ent = self.calcutaleEnt(data_set)
        best_feature = ''
        base_inforgain = 0.0
        feature_list = list(data_set.columns)
        feature_list.remove(self.lable)
        
        for feature in feature_list:
            this_feature_samples = self.getFeatureProprity(data_set[feature])
            fea_val_list = set(list(data_set[feature]))
            feature_ent = 0.0
            for fea_val in fea_val_list:
                extract_data = self.splitData(data_set,feature,fea_val)
                cell_fea_sample = self.getFeatureProprity(extract_data)
                cell_feature_ent = (abs(cell_fea_sample)/abs(this_feature_samples))*self.calcutaleEnt(extract_data)
                
                feature_ent += cell_feature_ent
            tmp_infor_gain = base_ent-feature_ent
            if tmp_infor_gain > base_inforgain:
                base_inforgain = tmp_infor_gain
                best_feature = feature
        return best_feature
        
    def majorityCnt(self,classList):
        classCount={}
        print (classList)
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
        
    
        
    def createTree(self,data_set,feature):
        class_list = list(data_set[self.lable])
        
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if data_set.shape[0] == 1:
            return self.majorityCnt(class_list)
        
        best_feature = self.inforGain(data_set)
        tree = {best_feature:{}}
        feature.remove(best_feature)
        feature_values = set(list(data_set[best_feature]))
        for fea_val in feature_values:
            copy_features = feature[:]
            tree[best_feature][fea_val] = self.createTree(self.splitData(data_set,best_feature,fea_val),copy_features)
        return tree
        
    
class Predict(DecisionTree):
    def __init__(self):
       pass
    
    def predicter(self,tree,features,testX):
        first_key = list(tree.keys())[0]
        second_dict = tree[first_key]
        for key in second_dict.keys():
            if testX[first_key][0] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_lable = self.predicter(second_dict[key],features,testX)
                else:
                    class_lable = second_dict[key]
        return class_lable
                
        
    
        
        
    
    
        
        
def CreateDataSet1():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    data = pd.DataFrame(dataSet,columns=['no_surfing','flipper','lable'])
    return data

def CreateDataSet():
    data_set = pd.read_csv('melon.csv')[['色泽','根蒂','敲声','纹理','脐部','触感','好瓜']]
    return data_set

data_set = CreateDataSet()
Tree = DecisionTree(data_set)
#Tree.splitData(CreateDataSet(),'flipper',1)
#print (Tree.inforGain(data_set))
tree = Tree.createTree(data_set=CreateDataSet(),feature=['色泽','根蒂','敲声','纹理','脐部','触感','好瓜'])

test_data = data_set.head(1)
features = ['色泽','根蒂','敲声','纹理','脐部','触感','好瓜']
test_data = test_data[features]
print (test_data)
Predicter = Predict()
print (Predicter.predicter(tree,features,test_data))