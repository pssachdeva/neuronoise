from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nd',
    description='',
    long_description=long_description,
    author='Pratik Sachdeva',
    author_email='pratik.sachdeva@berkeley.edu',
    install_requires=[
      'numpy',
      'scipy'
    ]
)
