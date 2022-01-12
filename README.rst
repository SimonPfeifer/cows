
.. start-badges

.. image:: https://img.shields.io/github/commits-since/SimonPfeifer/python-cows/v0.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/SimonPfeifer/python-cows/compare/v0.0.1...master

.. image:: https://readthedocs.org/projects/python-cows/badge/?version=latest
    :target: https://python-cows.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. end-badges

====
COWS
====

The *cows* python package is an implementation of the cosmic filament finder COsmic Web Skeleton (COWS). The cosmic filament finder works on Hessian-based cosmic web idetifier (such as the V-web) and returns a catalogue of filament spines.

It works by identifying the medial axis, or skeleton, of cosmic web filaments and then separating this skeleton into individual filaments. For more information of the specifics of the method, see `here <https://arxiv.org/search/astro-ph?searchtype=author&query=Pfeifer%2C+S>`_.


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

`arXiv:XXXX.XXXXX <https://arxiv.org/search/astro-ph?searchtype=author&query=Pfeifer%2C+S>`_ : "COWS: A filament finder for Hessian cosmic web identifiers"
