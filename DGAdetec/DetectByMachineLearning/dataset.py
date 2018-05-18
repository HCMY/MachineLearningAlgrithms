
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


def load_data():
	domains360 = pd.read_csv('../datas/360.txt',
							names=['domain'],
							header=None)
	domains360 = domains360.dropna()
	domains360['label'] = [0]*domains360.shape[0]
	#domains360 = domains360.drop_duplicates()

	domainsdga = pd.read_csv('../datas/dga.txt', 
								names=['domain'], 
								header=None)
	domainsdga = domainsdga.dropna()
	domainsdga['label'] = [0]*domainsdga.shape[0]

	domain_normal = pd.read_csv('../datas/normal_domains.csv', 
							names=['domain'],
							header=None)
	domain_normal = domain_normal.dropna()
	domain_normal['label'] = [1]*domain_normal.shape[0]

	train = pd.concat([domains360, domainsdga, normaldga], 
					   ignore_index=True,
					   key=['domain'])

	train = np.array(train)
	train = np.random.shuffle(train)

	return train




