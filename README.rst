====
COWS
====

.. start-badges

.. image:: https://img.shields.io/pypi/v/pycows.svg
    :alt: PyPi version
    :target: https://pypi.com/v/pycows


.. image:: https://img.shields.io/github/commits-since/SimonPfeifer/python-cows/v/v/v0.0.2...master.svg.svg
    :alt: Commits since latest release
    :target: https://github.com/SimonPfeifer/python-cows/compare/v/v/v0.0.2...master.svg...master


.. image:: https://img.shields.io/pypi/pyversions/pycows.svg
    :alt: Python versions
    :target: https://pypi.python.org/pypi/pycows/


.. image:: https://readthedocs.org/projects/python-cows/badge/?version=latest
    :alt: Documentation Status
    :target: https://python-cows.readthedocs.io/en/latest/?badge=latest

.. end-badges


The *cows* python package is an implementation of the cosmic filament finder COsmic Web Skeleton (COWS). The cosmic filament finder works on Hessian-based cosmic web idetifier (such as the V-web) and returns a catalogue of filament spines.

It works by identifying the medial axis, or skeleton, of cosmic web filaments and then separating this skeleton into individual filaments. For more information of the specifics of the method, see `here <https://arxiv.org/pdf/2201.04624.pdf>`_.


Requirements
------------
::

    numpy>=1.19.3

Quick installation
------------------

To install the *cows* python package using PyPI::

    pip install pycows

To install the package from source::

    python setup.py install

For more inforamtion on installation, see `INSTALL.rst <https://github.com/SimonPfeifer/cows/blob/master/INSTALLATION.rst>`_.

Documentation
==============

The full documentation can be accessed at `readthedocs <https://python-cows.readthedocs.io/en/latest/index.html>`_ or generated as a set of local files by running::

    sphinx-build ./docs ./docs/_build


Citing
======

When using COWS in a publication, please cite the following paper:

`arXiv:2201.04624 <https://arxiv.org/abs/2201.04624>`_ : "COWS: A filament finder for Hessian cosmic web identifiers"
