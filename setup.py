"""
Created on Sun Nov 27 12:35:52 2022
@author: Murat Cem Köse
"""

from setuptools import setup, find_packages

setup(name='91AutoImageClsuter',
      version='0.2',
      url='https://github.com/muratcemkose/91AutoImageClsuter',
      author='Murat Cem Köse',
      author_email='muratcem.kose@gmail.com',
      description='A python library to cluster your images easily based on geolocation',
      packages=find_packages(),
      long_description=open('README.md').read(),
      zip_safe=False)
