import os

root_path = os.path.dirname(__file__)

#_big_grame_model_path = root_path+'/models/big_grame.pkl'
#_triple_grame_model_path = root_path+'/models/tripple_gram.pkl'
_positive_grame_model_path = root_path+'/models/positive_grame.pkl'
_word_grame_model_path = root_path+'/models/word_grame.pkl'
_positive_count_matrix = root_path+'/models/positive_count_matrix.npy'
_word_count_matrix = root_path+'/models/word_count_matrix.npy' 


_positive_domain_path = root_path+'/resource/aleax100k.npy'
_std_postive_domain_path = root_path+'/resource/std_pos_domain.npy'