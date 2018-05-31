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
	domain1 = ['www.baidu.com','www.sssllllld.com',
			   'www.google.com','www.sssllllld.com']
	predy = detector.predict(domain1)
	print(predy)


import threading
def multiple_thread():
	a = threading.Thread(target=test1,)
	b = threading.Thread(target=test2,)
	c = threading.Thread(target=test3,)

	a.start()
	b.start()
	c.start()



