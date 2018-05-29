
import numpy as np

from dgadetec import cluster
from dgadetec import settings

aleax100k = np.load(settings._positive_domain_path)

def test_cluster():
	cluster.extract_point_domain(aleax100k[:500])


if __name__ == '__main__':
	test_cluster()
