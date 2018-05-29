from sklearn.externals import joblib
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from . import settings
from . import cluster

def n_grame_train(big_domain, grame_range, model_name):
	"""parameters:
	big dimain: domains waiting to bed fitted
	grame_range: a tuple
	model_name: string end with '.pkl'

	"""
	vec = CountVectorizer(min_df=1, ngram_range=grame_range, token_pattern=r"\w")
	grame_model = vec.fit(big_domain)
	joblib.dump(grame_model, './models/'+model_name)


# generate n_grame from positive domain
def positive_train(big_domain):
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
    joblib.dump(remodel, './models/hmm_gmm.pkl')

# generate std_pos_domain 
def cluster(positive_doamins):
	cluster.extract_point_domain(positive_doamins)


 def train_model(train_martix):
 	pass
 	

