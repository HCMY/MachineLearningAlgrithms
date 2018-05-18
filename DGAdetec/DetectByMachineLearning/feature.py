
import re
import os
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from collections import Counter
from functools import reduce


import dataset

class FeatureExtractor(object):
	"""docstring for ClassName"""
	def __init__(self, domain_list):
		self._domain_list = domain_list
		self._positive_domain_list = None
		self._big_grame_model_path = './models/big_grame.pkl'
		self._triple_grame_model_path = './models/tripple_gram.pkl'
		self._positive_grame_model_path = './models/positive_grame.pkl'
		self._word_grame_model_path = './models/word_grame.pkl'
		self._positive_count_matrix = './models/positive_count_matrix.npy'
		self._word_count_matrix = './models/word_count_matrix.npy' 
		self._positive_domain_list = self._load_positive_domain()

	# check wether required files exis 
	def _check_files(self, **args):
		pass


	def _load_positive_domain(self):
		positive = pd.read_csv('../datas/aleax100k.csv', names=['domain'], header=None, dtype={'word': np.str}, encoding='utf-8')
		positive = positive.dropna()
		positive = positive.drop_duplicates()
		return positive['domain'].tolist()

	def count_aeiou(self):
		count_result = []
		for domain in self._domain_list:
			len_aeiou = len(re.findall(r'[aeiou]',domain.lower()))
			aeiou_rate = (0.0+len_aeiou)/len(domain)
			tmp = [domain, len(domain), len_aeiou, aeiou_rate]
			count_result.append(tmp)

		count_result = pd.DataFrame(count_result, 
									columns=['domain','domain_len',
											 'aeiou_len','aeiou_rate'])
		return count_result


	#the rate between original domain length and seted domain length
	
	def unique_char_rate(self):
		unique_rate_list = []
		for domain in self._domain_list:
			unique_len = len(set(domain))
			unique_rate = (unique_len+0.0)/len(domain)
			tmp = [domain, unique_len, unique_rate]
			unique_rate_list.append(tmp)

		unique_rate_df = pd.DataFrame(unique_rate_list, 
										columns=['domain','unique_len','unique_rate'])
		return unique_rate_df


	# calculate double domain's jarccard index
	def _jarccard2domain(self, domain_aplha, domain_beta):
		"""parameters:
		domain_alpha/beta: string-like domain
		returns: this couples jarccard index
		"""
		listit_domain_alpha = list(domain_aplha)
		listit_domain_beta = list(domain_beta)

		abs_intersection = np.intersect1d(listit_domain_alpha, listit_domain_beta).shape[0]
		abs_union = np.union1d(listit_domain_alpha, listit_domain_beta).shape[0]
		
		return abs_intersection/abs_union*1.0


	# calculate each fake domain's average corresponding jarccard index 
	# with positive domain collection
	def jarccard_index(self):
		"""parameters:
		positive_domain_list: positve samples list, 1Darray like
		return: a pandas DataFrame, 
				contains domian col and average jarccard index col
		"""
		positive_domain_list=self._positive_domain_list

		jarccard_index_list = []
		for fake_domain in self._domain_list:
			total_jarccard_index = 0.0
			for std_domain in positive_domain_list:
				total_jarccard_index += self._jarccard2domain(fake_domain, std_domain)
			
			avg_jarccard_index = total_jarccard_index/len(positive_domain_list)
			tmp = [fake_domain, avg_jarccard_index]
			jarccard_index_list.append(tmp)

		jarccard_index_df = pd.DataFrame(jarccard_index_list, 
											columns=['domain','avg_jarccard_index'])

		return jarccard_index_df

	# TODO
	def hmm_index(self):
		pass

	'''
	# calculate n_grame of each domains
	# notes: you should update this model frequency
	# 		decrease the dimension of the transformed data set
	# 		and rank features
	def big_grame(self):
		if not os.path.exists(self._n_grame_model_path):
			raise("n_grame model dosen't exists, try to training this model\n\
				train scripts at same level folder called prepare_model.py\n\
				notes: training n_grame model by domains as much as you have")

		grame_model = joblib.load(self._n_grame_model_path)
		vec = grame_model.transform(np.array(self._domain_list))

		df = pd.DataFrame(vec.toarray(), columns=grame_model.get_feature_names())
		domains = pd.DataFrame(self._domain_list, columns=['domain'])
		df = pd.concat([domains, df], axis=1)
		
		return df

	def tripple_gram(self):
		pass
	
	'''


	#calculate entropy of domains entropy
	def entropy(self):
		"""parameters

		return: entropy DataFrame [doamin, entropy]
		"""
		entropy_list = []
		for domain in self._domain_list:
			p, lns = Counter(domain), float(len(domain))
			entropy = (-sum(count/lns * math.log(count/lns, 2) for count in p.values()))
			tmp = [domain, entropy]
			entropy_list.append(tmp)

		entropy_df = pd.DataFrame(entropy_list, columns=['domain','entropy'])
		return entropy_df



	#calculate grame(3,4,5) and its differ
	def n_grame(self):
		"""
		return local grame differ with positive domains and word list
		"""
		'''self._check_files(self._positive_count_matrix,
						  self._positive_grame_model_path,
						  self._word_grame_model_path,
						  self._word_count_matrix)
		'''
		positive_count_matrix = np.load(self._positive_count_matrix)
		positive_vectorizer = joblib.load(self._positive_grame_model_path)
		word_count_matrix = np.load(self._word_count_matrix)
		word_vectorizer = joblib.load(self._word_grame_model_path)

		positive_grames = positive_count_matrix * positive_vectorizer.transform(self._domain_list).T
		word_grames = word_count_matrix * word_vectorizer.transform(self._domain_list).T
		diff = positive_grames - word_grames
		domains = np.asarray(self._domain_list)


		n_grame_nd = np.c_[domains, positive_grames, word_grames, diff]
		n_grame_df = pd.DataFrame(n_grame_nd, columns=['domain','positive_grames','word_grames','diff'])
		
		return n_grame_df




def get_feature(domain_list):
	extractor = FeatureExtractor(domain_list)

	print("extracting count_aeiou....")
	aeiou_df = extractor.count_aeiou()
	print("extracted count_aeiou, shape is %d\n" % aeiou_df.shape[0])

	print("extracting unique_rate....")
	unique_rate_df = extractor.unique_char_rate()
	print("extracted unique_rate, shape is %d\n" % unique_rate_df.shape[0])

	#print("extracting jarccard_index....")
	#jarccard_index_df = extractor.jarccard_index()
	#print("extracted jarccard_index.....\n")

	print("extracting entropy....")
	entropy_df = extractor.entropy()
	print("extracted entropy, shape is %d\n"%entropy_df.shape[0])
	
	print("extracting n_grame....")
	n_grame_df = extractor.n_grame()
	print("extracted n_grame, shape is %d\n"%n_grame_df.shape[0])

	print("merge all features on domains...")
	multiple_df = [aeiou_df, unique_rate_df, 
				  entropy_df,
				  n_grame_df]

	df_final = reduce(lambda left,right: pd.merge(left,right,on='domain',how='left'), multiple_df)

	print("merged all features, shape is %d\n"%df_final.shape[0])
	# check df
	std_rows = aeiou_df.shape[0]
	df_final_rows = df_final.shape[0]

	if std_rows != df_final_rows:
		raise("row dosen't match after merged multiple_df")

	df_final = df_final.drop(['domain'],axis=1)
	df_final = df_final.round(3)
	return np.array(df_final)

	

		










		