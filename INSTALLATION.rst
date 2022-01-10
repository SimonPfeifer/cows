=============
Installation
=============

Rquirements
-----------

The *cows* package integrates some C/C++ code which are generated using `cython`. These source files are included and are used by default. Therefore it is not necessary to have an installation of `cython`. The requirements for *cows* are::

    numpy>=1.19.3

    [optional]
    cython>=0.29.1

Quick installation
------------------

To install the *cows* python package using PyPI::

    pip install pycows

To install the package from source::

    python setup.py install


Rebuilding C/C++ source files
-----------------------------

The package uses `cython` to generate some source files. These source files are included and are used by default. To generate new source files during the installation, set ``USE_CYTHON=True`` in *setup.py* before installing from source. This requires `cython>=0.29.1`.


Clean
-----

When ``python setup.py`` is run, temporary build products are placed in the
``build`` directory. To clean and remove the ``build`` directory,
then run::

    python setup.py clean --all
