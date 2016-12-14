#!/usr/bin/env python
"""Setup file for the popeye package."""

import urllib
import zipfile 
import os
import sys

popeye_config = os.path.join(os.path.expanduser('~'), '.popeye')

# # If the data is not already there: 
# if not os.path.exists(popeye_config):
#     os.mkdir(popeye_config)
#     # You might need to get the data:
#     print ("Downloading test-data ...")
#     # Get the test data and put it in the right place
#     f=urllib.urlretrieve("http://arokem.org/data/popeye.zip")[0]
#     zf = zipfile.ZipFile(f)
#     zf.extractall(path=popeye_config)    

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from distutils.core import setup
from setuptools import setup, Extension, Command

# Get version and release info, which is all stored in popeye/version.py
ver_file = os.path.join('popeye', 'version.py')
with open(ver_file) as f:
    exec(f.read())

install_requires=['scipy','numpy','matplotlib','nibabel','statsmodels','sharedmem','cython'],

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=['scipy','numpy','matplotlib','nibabel','statsmodels','sharedmem','cython'],
            )

try:
    from distutils.extension import Extension
    from Cython.Distutils import build_ext as build_pyx_ext
    from numpy import get_include
    # add Cython extensions to the setup options
    exts = [Extension('popeye.spinach',
                      ['popeye/spinach.pyx'],
                       include_dirs=[get_include()]),
                      ]
    opts['cmdclass'] = dict(build_ext=build_pyx_ext)
    opts['ext_modules'] = exts
except ImportError:
    # no loop for you!
    pass

# For some commands, use setuptools.  Note that we do NOT list install here!
# If you want a setuptools-enhanced install, just run 'setupegg.py install'
needs_setuptools = set(('develop', ))
if len(needs_setuptools.intersection(sys.argv)) > 0:
    import setuptools

# Only add setuptools-specific flags if the user called for setuptools, but
# otherwise leave it alone
if 'setuptools' in sys.modules:
    opts['zip_safe'] = False

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

# Now call the actual setup function  
if __name__ == '__main__':
    setup(**opts)
    
    # this is a total hack!
    # python setup.py install build_ext does not create
    # spinach.so in the ./popeye/ dir.  this is necessary
    # for importing the Cython methods.  don't know why it
    # doesn't build file in the propery location.
    # import os
    # import shutil
    # srcfile = find('spinach.so','.')
    # dstfile = './popeye/spinach.so'
    # shutil.copy(srcfile, dstfile)
    
    