.. -*- mode: rst -*-

|Travis|_

.. |Travis| image:: https://api.travis-ci.org/kdesimone/popeye.png?branch=master
.. _Travis: https://travis-ci.org/kdesimone/popeye/

popeye
============

popeye is a Python module for estimating population receptive fields from from fMRI data built on top of SciPy and distributed under the 3-Clause BSD license.

popeye is currently under development.

31 July 2013
kevin desimone
kevindesimone@gmail.com

Dependencies
============

popeye is tested to work under Python 2.7.

The required dependencies to build the software are NumPy >= 1.6.2,
SciPy >= 0.9, Nibabel >= 1.3.0, and Cython >= 0.18.

For running the tests you need nose >= 1.1.2.

Install
=======

To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install