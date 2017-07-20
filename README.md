[![TravisCI](https://api.travis-ci.org/kdesimone/popeye.svg?branch=master)](https://travis-ci.org/kdesimone/popeye/)
[![Coverage](https://codecov.io/gh/kdesimone/popeye/branch/master/graph/badge.svg)](https://codecov.io/gh/kdesimone/popeye)
[![JOSS](http://joss.theoj.org/papers/053a64ce9fda79e99fe8a703e30e4786/status.svg)](http://joss.theoj.org/papers/053a64ce9fda79e99fe8a703e30e4786)
[![DOI](https://zenodo.org/badge/11797525.svg)](https://zenodo.org/badge/latestdoi/11797525)
[![PyPI version](https://badge.fury.io/py/popeye.svg)](https://badge.fury.io/py/popeye)

popeye
======

popeye is a Python module for estimating population receptive fields
from from fMRI data built on top of SciPy and distributed under the
3-Clause BSD license.

Documentation for popeye and contact information for the authors are available at https://kdesimone.github.io/popeye

popeye is currently under development.

31 July 2013 kevin desimone <kevindesimone@gmail.com>

Dependencies
============

Popeye is tested to work under Python 2.7 and Python 3.5

The required dependencies to build the software are NumPy &gt;= 1.6.2, SciPy &gt;= 0.9, Nibabel &gt;= 1.3.0, Cython &gt;= 0.18, sharedmem &gt;= 0.3, and statsmodels &gt;= 0.6.

For running the tests you need nose &gt;= 1.1.2.

Install
=======

You can install popeye and its dependencies through PyPi:

    pip install popeye

Or you can install from the source. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install
