========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-cows/badge/?style=flat
    :target: https://python-cows.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/SimonPfeifer/python-cows/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/SimonPfeifer/python-cows/actions

.. |requires| image:: https://requires.io/github/SimonPfeifer/python-cows/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/SimonPfeifer/python-cows/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/SimonPfeifer/python-cows/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/SimonPfeifer/python-cows

.. |commits-since| image:: https://img.shields.io/github/commits-since/SimonPfeifer/python-cows/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/SimonPfeifer/python-cows/compare/v0.0.0...main



.. end-badges

Cosmic filament finder

* Free software: BSD 3-Clause License

Installation
============

::

    pip install cows

You can also install the in-development version with::

    pip install https://github.com/SimonPfeifer/python-cows/archive/main.zip


Documentation
=============


https://python-cows.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
