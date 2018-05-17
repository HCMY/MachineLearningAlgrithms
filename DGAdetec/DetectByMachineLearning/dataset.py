
import os
import numpy as np
import pandas as pd

def load_simple_data():
	files = os.listdir('../AlgorithmPowereddomains')
	
	domain_list = []
	for f in files:
		path = '../AlgorithmPowereddomains/'+f
		domains = pd.read_csv(path,names=['domain'])
		domains = domains['domain'].tolist()
		for item in domains:
			domain_list.append(item)
	return domain_list


