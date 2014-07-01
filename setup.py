#!/usr/bin/env python
"""Setup file for the popeye package."""

import urllib
import zipfile 
import os
import sys

popeye_config = os.path.join(os.path.expanduser('~'), '.popeye')
# If the data is not already there: 
if not os.path.exists(popeye_config):
    os.mkdir(popeye_config)
    # You might need to get the data:
    print ("Downloading test-data ...")
    # Get the test data and put it in the right place
    f=urllib.urlretrieve("http://arokem.org/data/popeye.zip")[0]
    zf = zipfile.ZipFile(f)
    zf.extractall(path=popeye_config)    

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from distutils.core import setup

# Get version and release info, which is all stored in popeye/version.py
ver_file = os.path.join('popeye', 'version.py')
with open(ver_file) as f:
    exec(f.read())

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
            requires=REQUIRES,
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

# Now call the actual setup function  
if __name__ == '__main__':
    setup(**opts)
