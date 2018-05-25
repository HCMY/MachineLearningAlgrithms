
import numpy as np
import math
from sklearn.cluster import KMeans

from . import settings

class Kcluster(object):
	"""docstring for Kneighbor"""
	def __init__(self, neighbors=10):
		self.k = neighbors

	def _maintain_dict(self, domainA, domainB):
		domainA = list(domainA)
		domainB = list(domainB)
		
		std_dict = {}
		for domain in domainA:
			if  domain in std_dict:continue
			std_dict[domain] = 1
		for domain in domainB:
			if domain in std_dict:continue
			std_dict[domain] = 1

		return std_dict



	def _mapper(self, domainA, domainB):
		std_dict = self._maintain_dict(domainA, domainB)
		domainA = list(domainA)
		domainB = list(domainB)

		maxlen = max(len(domainA),len(domainB))

		for idx in range(len(domainA)):
			character = domainA[idx]
			if character in std_dict:
				domainA[idx] = 1
			else:
				domainA[idx] = 0
		for out_idx in range(maxlen-len(domainA)):
			domainA.append(0)

		for idx in range(len(domainB)):
			character = domainB[idx]
			if character in std_dict:
				domainB[idx] = 1
			else:
				domainB[idx] = 0

		for out_idx in range(maxlen-len(domainB)):
			domainB.append(0)
		
		return np.array(domainA), np.array(domainB)

	def _simility(self, domainA, domainB):
		mapped_A, mapped_B = self._mapper(domainA, domainB)

		numerrator = np.dot(mapped_A, mapped_B)
		abs_A = math.sqrt(np.dot(mapped_A, mapped_A))
		abs_B = math.sqrt(np.dot(mapped_B, mapped_B))
		diver = abs_A*abs_B+0.0

		if diver == 0:
			diver = 1.0
		simility = numerrator/diver

		return simility


	def _match_sim(self, complete_domains, target_domain):
		"""parameters:
		
		"""
		simility_list = []
		for domain in complete_domains:
			simility = self._simility(domain, target_domain)
			simility_list.append(simility)
		
		return simility_list

	def _init_feature_table(self, complete_domains):
		featube_mat = []
		for domain in complete_domains:
			featube_mat.append(self._match_sim(complete_domains, domain))

		featube_mat = np.array(featube_mat)

		return featube_mat


	def _get_clustures_list(self, positive_domains):
		features = self._init_feature_table(positive_domains)
		model = KMeans(n_clusters=self.k, 
						random_state=0).fit(features)
		centers = model.cluster_centers_
		lables = model.labels_
		idx_list = []

		for point in centers:
			idx = 0; target_idx = 0; diff = np.inf
			for feature in features:
				tmp_diff = abs((point-feature).sum())
				if tmp_diff<diff:
					diff = tmp_diff
					target_idx = idx
				idx += 1
			idx_list.append(target_idx)

		return idx_list



def extract_point_domain(domain_list):
	idx_list = Kcluster()._get_clustures_list(domain_list)
	point_domain = np.array(domain_list)[idx_list]
	
	#save like [domian1, domain2....]
	print(point_domain)
	np.save(settings._std_postive_domain_path, point_domain)






		
