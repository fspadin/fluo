#!/usr/bin/env python3

from setuptools import setup, find_packages
from os.path import join, dirname
import re

version = re.search("__version__ = '([^']+)'",
                    open('./fluo/__init__.py').read()).group(1)

setup(
    name='fluo',
    version=version,
    author='Anna Chmielinska',
    author_email='anka.chmielinska@gmail.com',
    url = 'https://github.com/AnnaChmielinska/fluo',
    download_url = 'https://github.com/AnnaChmielinska/fluo/archive/1.0.tar.gz',
    license='General Public License 3',
    description='Fluorescence in time domain toolkit.',
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    packages = find_packages(),
    install_requires=[
        'numpy >= 1.13.1',
        'matplotlib >= 2.0.2',
        'scipy >= 0.19.1',
        'lmfit >= 0.9.7',
        'tqdm >= 4.15.0'],
)
