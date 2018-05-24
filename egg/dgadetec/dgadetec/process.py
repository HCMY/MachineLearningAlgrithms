
import pandas as pd
import numpy as np
import settings
import os

root_path = os.path.abspath('.')

def csv2npy():
	positive = pd.read_csv(root_path+'/resource/tmp/aleax100k.csv', names=['domain'], 
								header=None, dtype={'word': np.str}, 
								encoding='utf-8')
	positive = positive.dropna()
	positive = positive.drop_duplicates()
	positive = positive['domain'].tolist()
	positive = np.asarray(positive)
	print(positive)
	np.save(settings._positive_domain_path,positive)


if __name__ == '__main__':
	csv2npy()
