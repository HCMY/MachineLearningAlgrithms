
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
import os
from sklearn import feature_extraction 
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from dgadetec import settings
from dgadetec.pretrain.cluster import extract_point_domain



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

def metric_me(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 =f1_score(y_true, y_pred)
    
    return accuracy, f1


def train_all_models():
	from dgadetec.feature_extractor import  get_feature
	data = load_data()
	data = pd.DataFrame(data, columns=['domain', 'label'])
	data = data.drop_duplicates('domain')
	data = np.array(data)
	print("all samples= ",data.shape)
	print("dataY contains:", np.unique(data[:,1]))

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

	from sklearn.externals import joblib
	joblib.dump(simpleLR, './dgadetec/models/LR.pkl')
	joblib.dump(simpleSVM, './dgadetec/models/SVM.pkl')
	joblib.dump(simpleGBM, './dgadetec/models/GBM.pkl')


# generate n_grame from positive domain
def positive_train(big_domain):
	"""parameters:
	big domain: large scale posotive domains, type:list
	"""
	vec = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
	grame_model = vec.fit(big_domain)
	joblib.dump(grame_model, settings._positive_grame_model_path)
	counts_matrix = grame_model.transform(big_domain)
	positive_counts = np.log10(counts_matrix.sum(axis=0).getA1())
	np.save(settings._positive_count_matrix , positive_counts)

def word_train(word):
	vec = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
	grame_model = vec.fit(word)
	
	joblib.dump(grame_model, settings._word_grame_model_path)
	counts_matrix = grame_model.transform(word)
	word_counts =  np.log10(counts_matrix.sum(axis=0).getA1())
	
	np.save(settings._word_count_matrix, word_counts)


def hmm_train(big_domain, big_y):
	X = [[0]]
	X_lens = [1]
	for domain in self._domain_list:
		vec = self._domain2vec(domain)
		np_vec = np.array(vec)
		X = np.concatenate([X, np_vec])
		x_lens.append(len(np_vec))
	remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
	remodel.fit(X,X_lens)
	pickle.dump(remodel, './models/hmm_gmm.pkl')

# generate std_pos_domain 
def cluster(positive_doamins):
	extract_point_domain(positive_doamins)

#make a domain length table which is used to calculate domain's length rank
def maintain_length_rank_table(big_positive_doamins):
	length_list = []
	for domain in big_positive_doamins:
		length_list.append(len(domain))

	length_list = list(set(length_list))
	length_list.sort()
	np.save(settings._length_rank_table, length_list)

# same as length rank table
def maintain_aeiou_rank_table(big_positive_doamins):
	aeiou_length_list = []
	for domain in big_positive_doamins:
		len_aeiou = len(re.findall(r'[aeiou]',domain.lower()))
		aeiou_length_list.append(len_aeiou)
	aeiou_length_list = list(set(aeiou_length_list))

	aeiou_length_list.sort()
	np.save(settings._aeiou_rank_table, aeiou_length_list)


if __name__ == '__main__':
	
	positive_domains_vec = np.load(settings._positive_domain_path)
	
	positive_train(positive_domains_vec)
	print("positive_train done\n")

	word_dataframe = pd.read_csv(settings._word_path, names=['word'], 
								 header=None, dtype={'word': np.str}, 
								 encoding='utf-8')
	word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
	word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
	word_dataframe = word_dataframe.dropna()
	word_dataframe = word_dataframe.drop_duplicates()
	word_train(word_dataframe['word'])
	print('word_train done\n')
	
	cluster(positive_domains_vec[:500])
	print('cluster done\n')
	maintain_aeiou_rank_table(positive_domains_vec)
	print('maintain_aeiou_rank_table done\n')
	maintain_length_rank_table(positive_domains_vec)
	print('maintain_length_rank_table done\n')
	
	print('start train models\n')
	train_all_models()
	
	print("test domains: www.hao123.com and www.dssd2sddjs.com\n")
	
	from dgadetec import detector
	print(detector.predict(['www.hao123.com','www.dssd2sddjs.com']))
	print("1 represent legit domain, 0 represent dga domain\n")
