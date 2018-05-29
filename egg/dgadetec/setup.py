from setuptools import setup

setup(
    name="dgadetec", 
    version="1.0",
    author="yaunchen zhou",
    author_email="yaunchenzhouhcmy@gmail.com",
    description=("This is a service of dga detection subscripe"),
    license="GPLv3",
    keywords="redis subscripe",
    packages=['dgadetec'],

    include_package_data = True,
    package_dir = {'dgadetec':'dgadetec'},
    package_data = {'dgadetec':['models/*','resource/*']},

    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
    ],
    
)