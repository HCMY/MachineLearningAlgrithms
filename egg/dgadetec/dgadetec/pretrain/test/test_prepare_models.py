
from .. import prepare_models as pm
from .. import cluster
from dgadetec import settings

import numpy as np
import pandas as pd
import os

def test_positive_train():
	positive_domains = np.load(settings._positive_domain_path)
	pm.positive_train(positive_domains)


'''# This function is not avaliable for Python2.6.6 on server(xxxx.0.191), that's great! 
def test_word_train():
	word_dataframe = pd.read_csv(settings._word_path, 
		names=['word'], 
		header=None, 
		dtype={'word': np.str}, encoding='utf-8')

	word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
	word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
	word_dataframe = word_dataframe.dropna()
	word_dataframe = word_dataframe.drop_duplicates()
	
	word_dataframe = word_dataframe['word'].tolist()
	
	pm.word_train(word_dataframe)
'''

def test_cluster():
	positive_domains = np.load(settings._positive_domain_path)
	cluster.extract_point_domain(positive_domains[:1000])


def test_maintain_table():
	"""geberate sort table 
	"""
	positive_domains = np.load(settings._positive_domain_path)
	pm.maintain_length_rank_table(positive_domains)
	pm.maintain_aeiou_rank_table(positive_domains)



def metric_me(y_true, y_pred):
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn import metrics
	
	accuracy = accuracy_score(y_true, y_pred)
	f1 =f1_score(y_true, y_pred)
    
	return accuracy, f1



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

def test_train_model():
	from dgadetec.feature_extractor import  get_feature

	from sklearn.externals import joblib
	
	data = load_data()

	trainX = data[:50000,0]
	trainY = data[:50000,1].astype(int) 
	testX = data[50000:51000, 0]
	testY = data[50000:51000, 1].astype(int)


	trainX = get_feature(trainX)
	testX = get_feature(testX)

	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC
	from sklearn.ensemble import GradientBoostingClassifier


	simpleLR = LogisticRegression()
	simpleLR.fit(trainX, trainY)
	pred_y_LR = simpleLR.predict(testX)
	acc, f1 = metric_me(testY, pred_y_LR)
	print("simpleLR acc={0} f1={1}".format(acc, f1))
	######################################################################
	simpleSVM = SVC()
	simpleSVM.fit(trainX,trainY)
	pred_y_SVM = simpleSVM.predict(testX)
	acc, f1 = metric_me(testY, pred_y_SVM)
	print("simpleSVM acc={0} f1={1}".format(acc, f1))
	###########################################################################3
	simpleGBM = GradientBoostingClassifier()
	simpleGBM.fit(trainX, trainY)
	pred_y_GBM = simpleGBM.predict(testX)
	acc, f1= metric_me(testY, pred_y_GBM)
	print("simpleGBM acc={0} f1={1}".format(acc, f1))


	from sklearn.externals import joblib
	joblib.dump(simpleLR, './dgadetec/models/LR.pkl')
	joblib.dump(simpleSVM, './dgadetec/models/SVM.pkl')
	joblib.dump(simpleGBM, './dgadetec/models/GBM.pkl')

	pred_y = (pred_y_LR+pred_y_GBM+pred_y_SVM)/3.0
	
	y = []
	for item in pred_y:
		if item>0.5:
			y.append(1)
		else:
			y.append(0)

	acc, f1 = metric_me(testY, y)
	print("simpleMixture acc={0} f1={1}".format(acc, f1))


