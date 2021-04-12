from setuptools import setup

setup(name='edm',
      version='1.0',
      packages=['env'],
      install_requires=['numpy',
                        'numba',
                        'gym'])
