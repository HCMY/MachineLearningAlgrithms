
from feature import FeatureExtractor, get_feature
from sklearn.externals import joblib
import numpy as np

#joblib.dump(lr, 'lr.model')
model = joblib.load('./models/LR.pkl')

def _check(pred_y, stdsamples):
	if not instance(pred_y, list):
		raise("type error. predict result is not list but%s"%type(pred_y))
	if len(pred_y) is not stdsamples:
		raise("predict error, expexted sampls is %d"%stdsamples,\
			"but real sampls is %d"%len(pred_y))

def detector(domain):
	feature_table = get_feature(domain)
	pred_y = model.predict(feature_table)
	_check(pred_y, len(domain))

	return pred_y

	


