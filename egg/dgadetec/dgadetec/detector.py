#-*- coding:utf-8 -*-
from . import feature_extractor as  feature

from sklearn.externals import joblib
import numpy as np
import os
import pickle

root_path = os.path.dirname(__file__) 

GBM = joblib.load(root_path+'/models/GBM.pkl')
LR = joblib.load(root_path+'/models/LR.pkl')
SVM = joblib.load(root_path+'/models/SVM.pkl') 


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
			raise ValueError("input should be a list, not {0}".format(type(domain)))

		if not isinstance(domain[0], str):
			raise ValueError("input inner element should be str, not{0}".format(type(domain[0])))

def predict(domain):
	"""parameters
	@domain: It must be a vector or your programe will crash.

	I used thress models, Gradiant Bosting Decision Tree、Logistic Regression and SVM for classify mission
	at the end, I synthesize these models' predicted result, and select their mean value as final result
	"""
	ckeck = Detector()
	ckeck._pre_check(domain)

	feature_table = feature.get_feature(domain)
	pred_by_GBM = GBM.predict(feature_table)
	pred_by_LR = LR.predict(feature_table)
	pred_by_SVM = SVM.predict(feature_table)

	pred_y = (pred_by_GBM+pred_by_SVM+pred_by_LR)/3.0

	pred_y = pred_y.tolist()

	y = []
	for item in pred_y:
		if item>0.5:
			y.append(1)
		else:
			y.append(0)
		
	ckeck._check(y, len(domain))

	return y



