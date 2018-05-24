from . import feature_extractor as  feature

from sklearn.externals import joblib
import numpy as np
import os

root_path = os.path.dirname(__file__) 
model = joblib.load(root_path+'/models/LR.pkl')

class Detector(object):
	'''
	'''
	def _check(self, pred_y, stdsamples):
		if not isinstance(pred_y, list):
			raise ValueError("type error. predict result is not list but%s"%type(pred_y))
		if len(pred_y) is not stdsamples:
			raise ValueError("predict error, expexted sampls is {1},\
				but real sampls is {2}".format(stdsamples, len(pred_y)))

	def _pre_check(self, domain):
		if not isinstance(domain, list):
			raise ValueError("input should be a list, not {}".format(type(domain)))

		if not isinstance(domain[0], str):
			raise ValueError("input inner element should be str, not{}".format(type(domain[0])))

def predict(domain):
	ckeck = Detector()
	ckeck._pre_check(domain)

	feature_table = feature.get_feature(domain)
	pred_y = model.predict(feature_table)
	pred_y = pred_y.tolist()
		
	ckeck._check(pred_y, len(domain))

	return pred_y



