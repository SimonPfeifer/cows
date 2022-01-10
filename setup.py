#!/usr/bin/env python3

import io
import os
import platform
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

from numpy import get_include
from multiprocessing import cpu_count
from pkg_resources import parse_version

USE_CYTHON = False
CYTHON_VERSION = '0.23.4'

if __name__ == '__main__':

    try:
        from Cython import __version__
        if parse_version(__version__) < parse_version(CYTHON_VERSION):
            raise RuntimeError('Cython >= %s is needed to build' % CYTHON_VERSION)
        from Cython.Build import cythonize
    except ImportError:
        print('WARNING: Cython not found! Using pre-existing files to build.')
        USE_CYTHON = False


    if USE_CYTHON:
        extensions = [Extension("cows._filament", 
                                ["src/cows/_filament.pyx"],
                                language="c"),
                      Extension('cows._skeletonize_3d_cy',
                                sources=['src/cows/_skeletonize_3d_cy.pyx'],
                                include_dirs=[get_include()],
                                language='c++')]
        # Cython doesn't automatically choose a number of threads > 1
        # https://github.com/cython/cython/blob/a0bbb940c847dfe92cac446c8784c34c28c92836/Cython/Build/Dependencies.py#L923-L925
        extensions = cythonize(extensions, nthreads=cpu_count(),
                  compiler_directives={'language_level': 3})
    else:
        extensions = [Extension("cows._filament", 
                                ["src/cows/_filament.c"],
                                language="c"),
                      Extension('cows._skeletonize_3d_cy',
                                sources=['src/cows/_skeletonize_3d_cy.cpp'],
                                include_dirs=[get_include()],
                                language='c++')]


    # Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
    # dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
    # deps have been safely installed).
    if 'TOX_ENV_NAME' in os.environ and os.environ.get('SETUPPY_EXT_COVERAGE') == 'yes' and platform.system() == 'Linux':
        CFLAGS = os.environ['CFLAGS'] = '-fprofile-arcs -ftest-coverage'
        LFLAGS = os.environ['LFLAGS'] = '-lgcov'
    else:
        CFLAGS = ''
        LFLAGS = ''


    class BinaryDistribution(Distribution):
        """Distribution which almost always forces a binary package with platform name"""
        def has_ext_modules(self):
            return super().has_ext_modules() or not os.environ.get('SETUPPY_ALLOW_PURE')


    def read(*names, **kwargs):
        with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
        ) as fh:
            return fh.read()


    setup(
        name='pycows',
        version='0.0.0',
        license='BSD-3-Clause',
        description='Cosmic filament finder',
        long_description='%s\n' % (
            re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst'))),
        long_description_content_type = 'text/x-rst',
        author='Simon Pfeifer',
        author_email='spfeifer@aip.de',
        url='https://github.com/SimonPfeifer/python-cows',
        packages=find_packages('src'),
        package_dir={'': 'src'},
        py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
        include_package_data=True,
        zip_safe=False,
        classifiers=[
            # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: BSD License',
            'Operating System :: Unix',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            # uncomment if you test on these interpreters:
            # 'Programming Language :: Python :: Implementation :: IronPython',
            # 'Programming Language :: Python :: Implementation :: Jython',
            # 'Programming Language :: Python :: Implementation :: Stackless',
            'Topic :: Utilities',
        ],
        project_urls={
            'Documentation': 'https://python-cows.readthedocs.io/',
            'Issue Tracker': 'https://github.com/SimonPfeifer/cows/issues',
        },
        keywords=[
            # eg: 'keyword1', 'keyword2', 'keyword3',
        ],
        python_requires='>=3.6',
        install_requires=[
            # eg: 'aspectlib==1.1.1', 'six>=1.7',
        ],
        extras_require={
            # eg:
            #   'rst': ['docutils>=0.11'],
            #   ':python_version=="2.6"': ['argparse'],
        },
        ext_modules=extensions,
        distclass=BinaryDistribution,
    )
