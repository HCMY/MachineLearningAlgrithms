import os

root_path = os.path.dirname(__file__)

"""
this file as a global setting files, mainly used to store where the required files we need is
, and where prevalant files store to 
"""

#_big_grame_model_path = root_path+'/models/big_grame.pkl'
#_triple_grame_model_path = root_path+'/models/tripple_gram.pkl'
_positive_grame_model_path = root_path+'/models/positive_grame.pkl'
_word_grame_model_path = root_path+'/models/word_grame.pkl'
_positive_count_matrix = root_path+'/models/positive_count_matrix.npy'
_word_count_matrix = root_path+'/models/word_count_matrix.npy' 
_word_path = root_path+'/resource/words.csv'
_train_data_path = root_path+'/resource/train.npy'

_positive_domain_path = root_path+'/resource/aleax100k.npy'
_std_postive_domain_path = root_path+'/resource/std_pos_domain.npy'

_length_rank_table = root_path+'/resource/length_rank_table.npy'
_aeiou_rank_table = root_path+'/resource/aeiou_rank_table.npy'


_model_LR_path = root_path+'/models/LR.pkl'
_model_SVM_path = root_path+'/models/SVM.pkl'
_model_GBM_path = root_path+'/models/GBM.pkl'
