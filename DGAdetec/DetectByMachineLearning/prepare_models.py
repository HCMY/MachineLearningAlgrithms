from sklearn.externals import joblib
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

def n_grame_train(big_domain, grame_range, model_name):
	"""parameters:
	big dimain: domains waiting to bed fitted
	grame_range: a tuple
	model_name: string end with '.pkl'

	"""
	vec = CountVectorizer(min_df=1, ngram_range=grame_range, token_pattern=r"\w")
	grame_model = vec.fit(big_domain)
	joblib.dump(grame_model, './models/'+model_name)


