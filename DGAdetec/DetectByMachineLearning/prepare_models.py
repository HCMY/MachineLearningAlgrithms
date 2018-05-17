from sklearn.externals import joblib
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

def n_grame_train(big_domain):
	vec = CountVectorizer(min_df=1, ngram_range=(1,2), token_pattern=r"\w")
	grame_model = vec.fit(big_domain)
	joblib.dump(grame_model, './models/n_grame.pkl')


