

from feature_extractor import FeatureExtractor, get_feature
import dataset
import time

data = dataset.load_simple_data()
FE = FeatureExtractor(data)

def test_load_positive_domain():
	start = time.clock()
	data = FE._load_positive_domain()
	end = time.clock()
	print(end-start)

def test_count_aeiou1():
	start = time.clock()
	data = FE.count_aeiou1()
	end = time.clock()
	print(end-start)


def test_count_aeiou2():
	start = time.clock()
	data = FE.count_aeiou2()
	end = time.clock()
	print(end-start)

def test_entropy():
	start = time.clock()
	data = FE.entropy()
	end = time.clock()
	print(end-start)
	print(data)

def test_n_grame():
	start = time.clock()
	data = FE.n_grame()
	end = time.clock()
	print(end-start)
	#print(data)

def test_jarcard():
	start = time.clock()
	data = FE.jarccard_index()
	end = time.clock()
	print(end-start)


def test_get_feature():
	start = time.clock()
	feature = get_feature(data)
	end = time.clock()
	print(end-start)
	print(feature[0])


if __name__ == '__main__':
	#test_load_positive_domain()
	#test_count_aeiou1()
	#test_count_aeiou2()
	#test_entropy()
	test_get_feature()