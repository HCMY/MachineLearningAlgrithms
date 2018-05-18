
import re
import os
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from collections import Counter



import dataset

class FeatureExtractor(object):
	"""docstring for ClassName"""
	def __init__(self, domain_list):
		self._domain_list = list(set(domain_list))
		self._positive_domain_list = None
		self._n_grame_model_path = './models/n_grame.pkl'


	
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
	def jarccard_index(self, positive_domain_list):
		"""parameters:
		positive_domain_list: positve samples list, 1Darray like
		return: a pandas DataFrame, 
				contains domian col and average jarccard index col
		"""
		positive_domain_list = set(positive_domain_list)

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



	#calculate entropy of domains entropy
	def entropy(self):
		entropy_list = []
		for domain in self._domain_list:
			p, lns = Counter(domain), float(len(domain))
			entropy = (-sum(count/lns * math.log(count/lns, 2) for count in p.values()))
			tmp = [domain, entropy]
			entropy_list.append(tmp)

		entropy_df = pd.DataFrame(entropy_list, columns=['domain','entropy'])
		return entropy_df


	
	

		










		