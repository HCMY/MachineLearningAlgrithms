import dataset
import pandas as pd
import numpy as np
from feature import FeatureExtractor, get_feature
from sklearn.externals import joblib
##############test load data##########
data = dataset.load_simple_data()
#################################

######################TEST Feature extractor###################
my_extractor = FeatureExtractor(data[0:10])

######################################################


#TRAIN PREMODEL
#from prepare_models import n_grame_train, positive_train, word_train
#words = pd.read_csv('../datas/words.csv', names=['word'], header=None, dtype={'word': np.str}, encoding='utf-8')
#words = words.applymap(lambda x: str(x).strip().lower())
#words = words.dropna()
#words = words.drop_duplicates()
#word_train(words['word'].tolist())
#
#positive = pd.read_csv('../datas/aleax100k.csv', names=['domain'], header=None, dtype={'word': np.str}, encoding='utf-8')
#positive = positive.dropna()
#positive = positive.drop_duplicates()
#positive_train(positive['domain'].tolist())

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

#n_grame_corr = my_extractor.big_grame()
#print(n_grame_corr)
#########################n grame end but have to decrease its dimension#################


########################hmm leran############################

########################entropy######################
#entropy_corr = my_extractor.entropy()
#print(entropy_corr)
########################entropy end###################


################## n-grame #############
#n_grame_corr = my_extractor.n_grame()
#print(n_grame_corr)


################# n-grame ###############

#################### GET ALL Features#############

###################### digit ratio ######################
digit_ration_corr = my_extractor.digit_ratio()
print(digit_ration_corr)
#################### digit ratio end######################



def model():
	"""TODO MLmodel
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score


	data = dataset.load_data()
	print("all samples= ",data.shape)

	print("dataY contains:", np.unique(data[:,1]))

	data = pd.DataFrame(data, columns=['domain', 'label'])
	data = data.drop_duplicates(subset='domain')
	data = np.array(data)

	trainX = data[:30000,0]
	trainY = data[:30000,1].astype(int) 
	testX = data[30000:30500, 0]
	testY = data[30000:30500,1].astype(int)

	#print(trainX)
	print("trainY contains: ", np.unique(trainY))
	#print(testX)
	print("testY contains: ", np.unique(testY))

	feature_table = get_feature(trainX)

	LR = LogisticRegression()
	LR = LR.fit(feature_table,trainY)
	pred = LR.predict(get_feature(testX))
	acc = accuracy_score(testY, pred)
	print("acc stage 1: ", acc)

	joblib.dump(LR, './models/LR.pkl')
	algorithm_domains = dataset.load_simple_data()
	algorithm_domains = list(set(algorithm_domains))
	algorithm_y = [0]*len(algorithm_domains)

	pred_feature = get_feature(algorithm_domains)
	pred = LR.predict(pred_feature)

	acc = accuracy_score(algorithm_y, pred)
	print("acc stage 2: ", acc)



#if __name__ == '__main__':
#	model()