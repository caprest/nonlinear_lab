# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nonlinearlab',
    version='0.0.94',
    description='Nonlinear Library for graduate thesis',
    long_description=readme,
    author='Fumiya Shimada',
    author_email='caprest.f@gmail.com',
    url='',
    license=license,
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'matplotlib',
        'tqdm',
        'scipy',
        'statsmodels'
      ],
    packages=find_packages(exclude=('tests', 'docs'))
)
