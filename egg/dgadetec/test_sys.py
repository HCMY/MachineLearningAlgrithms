from dgadetec import detector


def test1():
	domain1 = ['www.baijdfjddu.com']
	predy = detector.predict(domain1)
	print(predy)


def test2():
	domain1 = ['www.baidu.com','www.sssllllld.com']
	predy = detector.predict(domain1)
	print(predy)

def test3():
	domain1 = [['www.baidu.com','www.sssllllld.com'],
			   ['www.baidu.com','www.sssllllld.com']]
	predy = detector.predict(domain1)
	print(predy)


if __name__ == '__main__':
	test2()