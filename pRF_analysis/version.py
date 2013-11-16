"""pRF_analysis version/release information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "pRF_analysis : voxel population receptive fields from fMRI data"

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
===============================================================
pRF_analysis : voxel population receptive fields from fMRI data
===============================================================

XXX Write a longer description here.


License information
===================

pRF_analysis is licensed under the terms of the new BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2013, Kevin Desimone
All rights reserved.
"""

NAME = "pRF_analysis"
MAINTAINER = "Kevin Desimone"
MAINTAINER_EMAIL = "kevindesimone@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/kdesimone/pRF_analysis/"
DOWNLOAD_URL = "http://github.com/kdesimone/pRF_analysis/downloads"
LICENSE = "Simplified BSD"
AUTHOR = "Kevin Desimone"
AUTHOR_EMAIL = "kevindesimone@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['pRF_analysis']
PACKAGE_DATA = {"pRF_analysis": ["LICENSE"]}
REQUIRES = []
