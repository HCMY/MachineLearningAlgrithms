import dataset
from feature import FeatureExtractor
##############test load data##########
data = dataset.load_simple_data()
#data = data[1000:1010]
#################################

######################TEST Feature extractor###################
my_extractor = FeatureExtractor(data[6:10])
########################### AEIOU corresponding#####################
#aeiou_corr_arr = my_extractor.count_aeiou()
#print(aeiou_corr_arr)
###############################AEIOU EBD#############################


###################### 字母数字所占域名长度的比例###############
#unique_corr_arr = my_extractor.unique_char_rate()
#print(unique_corr_arr)
#######################字母数字所占域名长度的比例###############

##########################jarccard index##############
#jarccard_index_arr = my_extractor.jarccard_index(data[1:10])
#print(jarccard_index_arr)
##########################jarccard index##############

######################## n-grame#########################
#from prepare_models import n_grame_train
#n_grame_train(data[1000:])
#n_grame_corr = my_extractor.big_grame()
#print(n_grame_corr)
#########################n grame end but have to decrease its dimension#################


########################hmm leran############################

########################entropy######################
#entropy_corr = my_extractor.entropy()
#print(entropy_corr)
########################entropy end###################