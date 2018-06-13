
from sklearn.externals import joblib
import numpy as np

from . import settings

'''
Here the library load all the required models, files used to extract features
Notes: all the required files shoud be manufacturaled befor you used it
       btw, as how to generate these files, please check relevant documentation 
       or look up the floder  './dgadetec/pretrain'
'''

positive_count_matrix = np.load(settings._positive_count_matrix)
positive_vectorizer = joblib.load(settings._positive_grame_model_path)

# model 'word_vectorizer' couldn't work on pythton 2.6.6, but 2.7 is work well
# or it's scikit-learn's unfixed bug, probobaly
#---------------------------------------------------------------
#word_vectorizer = joblib.load(settings._word_grame_model_path)
#word_count_matrix = np.load(settings._word_count_matrix)
#with open(settings._word_grame_model_path,'r') as f:
#	word_vectorizer = pickle.load(f)
#-----------------------------------------------------------

std_positive_domain_center = np.load(settings._std_postive_domain_path)

length_stdrank_table = np.load(settings._length_rank_table)
aeiou_stdrank_table = np.load(settings._aeiou_rank_table)

####consistent stay in memory so that whole paorgame could be speeded

