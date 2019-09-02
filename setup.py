__authors__ = ['Anselm W. Hahn']
__license__ = 'MIT'
__date__ = '02/09/2019'

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
	name='EdrixsRun',
	version='0.5',
	packages=['tests','EdrixsRun'],
	url='',
	license='MIT',
	author='hahn',
	author_email='Anselm.Hahn@gmail.com',
	description='Command Line tool for the NSLS-II edrixs libaries'
)
