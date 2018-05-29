
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import feature_extraction 


# In[2]:

from dgadetec import dataset
import pandas as pd
import numpy as np
from dgadetec.feature_extractor import  get_feature
from sklearn.externals import joblib


# In[3]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics


# ## pre-load data

# In[4]:

import os
def load_simple_data():
	files = os.listdir('./dgadetec/AlgorithmPowereddomains')
	
	domain_list = []
	for f in files:
		path = './dgadetec/AlgorithmPowereddomains/'+f
		domains = pd.read_csv(path,names=['domain'])
		domains = domains['domain'].tolist()
		for item in domains:
			domain_list.append(item)
	return domain_list


def load_data():
	if os.path.exists('./dgadetec/resource/train.npy'):
		train = np.load('./dgadetec/resource/train.npy')
		return train

	domains360 = pd.read_csv('./dgadetec/resource/360.txt',
							header=None)[[1]]
	domains360 = domains360.dropna()
	domains360['label'] = [0]*domains360.shape[0]

	#domains360 = domains360.drop_duplicates()

	domainsdga = pd.read_csv('./dgadetec/resource/dga-feed.txt', 
								names=['domain'], 
								header=None)
	domainsdga = domainsdga.dropna()
	domainsdga['label'] = [0]*domainsdga.shape[0]

	domain_normal = pd.read_csv('./dgadetec/resource/normal_domains.csv', 
							names=['domain'],
							header=None)
	domain_normal = domain_normal.dropna()
	domain_normal['label'] = [1]*domain_normal.shape[0]


	train = np.concatenate((domains360.values, domainsdga.values, domain_normal.values),axis=0)

	#train = train.drop_duplicates(subset=1)
	
	#train = np.array(train)
	np.random.shuffle(train)
	np.save('./dgadetec/resource/train.npy', train)

	return train


# In[5]:

data = load_data()
data = pd.DataFrame(data, columns=['domain', 'label'])
data = data.drop_duplicates(subset='domain')
data = np.array(data)
print("all samples= ",data.shape)
print("dataY contains:", np.unique(data[:,1]))


# In[6]:

trainX = data[:50000,0]
trainY = data[:50000,1].astype(int) 
testX = data[50000:51000, 0]
testY = data[50000:51000, 1].astype(int)


# In[9]:


trainX = get_feature(trainX[:500])
testX = get_feature(testX)


# ## various models

# In[ ]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:

def metric_me(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 =f1_score(y_true, y_pred)
    
    return accuracy, f1


# In[ ]:

simpleLR = LogisticRegression()
simpleLR.fit(trainX, trainY)
pred_y = simpleLR.predict(testX)
acc, f1 = metric_me(testY, pred_y)
print("simpleLR acc={} f1={}".format(acc, f1))
######################################################################
simpleSVM = SVC()
simpleSVM.fit(trainX,trainY)
pred_y = simpleSVM.predict(testX)
acc, f1 = metric_me(testY, pred_y)
print("simpleSVM acc={} f1={}".format(acc, f1))
###########################################################################3
simpleGBM = GradientBoostingClassifier()
simpleGBM.fit(trainX, trainY)
pred_y = simpleGBM.predict(testX)
acc, f1= metric_me(testY, pred_y)
print("simpleGBM acc={} f1={}".format(acc, f1))


# In[ ]:

from sklearn.externals import joblib
joblib.dump(simpleLR, './dgadetec/models/LR.pkl')
joblib.dump(simpleSVM, './dgadetec/models/SVM.pkl')
joblib.dump(simpleGBM, './dgadetec/models/GBM.pkl')


# In[ ]:

import time

start = time.clock()
X = get_feature(['www.deweuhydh.com'])
pred_result = simpleSVM.predict(X)
end = time.clock()

print(end-start)
print(pred_result)

