[![Build Status](https://api.travis-ci.org/kdesimone/popeye.svg?branch=master)](https://travis-ci.org/kdesimone/popeye/)

[![image](https://circleci.com/gh/kdesimone/popeye.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/kdesimone/popeye)

[![image](https://coveralls.io/repos/github/kdesimone/popeye/badge.svg?branch=master)](https://coveralls.io/github/kdesimone/popeye?branch=master)

[![status](http://joss.theoj.org/papers/053a64ce9fda79e99fe8a703e30e4786/status.svg)](http://joss.theoj.org/papers/053a64ce9fda79e99fe8a703e30e4786)


popeye
======

popeye is a Python module for estimating population receptive fields
from from fMRI data built on top of SciPy and distributed under the
3-Clause BSD license.

popeye is currently under development.

31 July 2013 kevin desimone <kevindesimone@gmail.com>

Dependencies
============

popeye is tested to work under Python 2.7.

The required dependencies to build the software are NumPy &gt;= 1.6.2,
SciPy &gt;= 0.9, Nibabel &gt;= 1.3.0, and Cython &gt;= 0.18.

For running the tests you need nose &gt;= 1.1.2.

Install
=======

To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install
