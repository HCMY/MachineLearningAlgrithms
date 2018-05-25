import re
import os
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from collections import Counter
from functools import reduce

from . import settings

root_path = os.path.dirname(__file__)

####belong to function ngrame, 
positive_count_matrix = np.load(settings._positive_count_matrix)
positive_vectorizer = joblib.load(settings._positive_grame_model_path)
word_count_matrix = np.load(settings._word_count_matrix)
word_vectorizer = joblib.load(settings._word_grame_model_path)
std_positive_domain_center = np.load(settings.__std_postive_domain_path)
####consistent stay in memory so that whole paorgame could be speeded

class FeatureExtractor(object):
	"""docstring for ClassName"""
	def __init__(self, domain_list):
		self._domain_list = domain_list
		self._positive_domain_list = None
		self._positive_domain_list = self._load_positive_domain()

	# check wether required files exis 
	def _check_files(self, *args):
		for val in args:
			if not os.path.exists(val):
				raise ValueError("file{} doesn't exis, check scripts \
					dataset and prepare_model ".format(val))	

	def _load_positive_domain(self):
		"""
		"""
		positive = np.load(settings._positive_domain_path)
		return positive

	# Calculate AEIOU rate in full domain
	def count_aeiou(self):
		"""parameters: None
		returns: np.ndarray like [[domain_len, len_aeiou, aeiou_rate]]
		"""
		count_result = []
		for domain in self._domain_list:
			len_aeiou = len(re.findall(r'[aeiou]',domain.lower()))
			aeiou_rate = (0.0+len_aeiou)/len(domain)
			tmp = [len(domain), len_aeiou, aeiou_rate]
			count_result.append(tmp)

		count_result = np.asarray(count_result) 
		return count_result

	def unique_char_rate(self):
		"""parameters: None
		return: np.adarray like [["unique char len","unique char rate"]]
		"""
		unique_rate_list = []
		for domain in self._domain_list:
			unique_len = len(set(domain))
			unique_rate = (unique_len+0.0)/len(domain)
			tmp = [unique_len, unique_rate]
			unique_rate_list.append(tmp)

		unique_rate_list = np.asarray(unique_rate_list)
		return unique_rate_list

	def entropy(self):
		"""parameters
		return: entropy ndarray like [doamin, entropy]
		"""
		entropy_list = []
		for domain in self._domain_list:
			p, lns = Counter(domain), float(len(domain))
			entropy = (-sum(count/lns * math.log(count/lns, 2) \
						for count in p.values()))
			entropy_list.append(entropy)

		entropy_list = np.asarray(entropy_list)
		return entropy_list

	def n_grame(self):
		"""
		return local grame differ with positive domains and word list
		"""
		positive_grames = positive_count_matrix * positive_vectorizer.transform(self._domain_list).T
		word_grames = word_count_matrix * word_vectorizer.transform(self._domain_list).T
		diff = positive_grames - word_grames
		domains = np.asarray(self._domain_list)


		n_grame_nd = np.c_[positive_grames, word_grames, diff]
		
		return n_grame_nd


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


	def jarccard_index(self):
		"""parameters:
		positive_domain_list: positve samples list, 1Darray like
		return: a pandas DataFrame, 
				contains domian col and average jarccard index col
		"""
		positive_domain_list=std_positive_domain_center

		jarccard_index_list = []
		for fake_domain in self._domain_list:
			total_jarccard_index = 0.0
			for std_domain in positive_domain_list:
				total_jarccard_index += self._jarccard2domain(fake_domain, std_domain)
			
			avg_jarccard_index = total_jarccard_index/len(positive_domain_list)
			tmp = [fake_domain, avg_jarccard_index]
			jarccard_index_list.append(tmp)

		jarccard_index_list = np.asarray(jarccard_index_list)
		return jarccard_index_list



def get_feature(domain_list):
	extractor = FeatureExtractor(domain_list)

	aeiou_df = extractor.count_aeiou()
	unique_rate_df = extractor.unique_char_rate()
	entropy_df = extractor.entropy()
	n_grame_df = extractor.n_grame()
	jarccard_index_df = extractor().jarccard_index()

	df_final = np.c_[aeiou_df, unique_rate_df, entropy_df, n_grame_df, jarccard_index_df]


	std_rows = aeiou_df.shape[0]
	df_final_rows = df_final.shape[0]
	if std_rows != df_final_rows:
		raise("row dosen't match after merged multiple_df")

	np.around(df_final, decimals=3, out=df_final)
	
	return df_final