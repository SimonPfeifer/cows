'''
Contains code from scikit-image v0.18.3
'''

import pytest

import numpy as np
from numpy.testing import assert_equal

from cows import skeletonize
from cows.dtype import img_as_ubyte


def test_skeletonize_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(ValueError):
        skeletonize(im)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        skeletonize(im)


def test_skeletonize_wrong_dim_periodic():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(ValueError):
        skeletonize(im, periodic=True)

    im = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        skeletonize(im, periodic=True)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        skeletonize(im, periodic=True)


def test_skeletonize_periodic_wrong_type():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    with pytest.raises(TypeError):
            skeletonize(im, periodic='True')

    with pytest.raises(TypeError):
            skeletonize(im, periodic=5)


def test_skeletonize_no_foreground():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    assert_equal(skeletonize(im), im)
    assert_equal(skeletonize(im, periodic=True), im)


def test_skeletonize_all_foreground():
    im = np.ones((3, 3, 4), dtype=np.uint8)
    assert_equal(skeletonize(im),
                 np.array([[[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0]],
 
                           [[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]], dtype=np.uint8))

    assert_equal(skeletonize(im, periodic=True), im)


def test_skeletonize_single_point():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    im[3, 3, 3] = 1
    assert_equal(skeletonize(im), im)
    assert_equal(skeletonize(im, periodic=True), im)


def test_skeletonize_already_thinned():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    im[3, 1:-1, 3] = 1
    im[2, -1, 3] = 1
    im[4, 0, 3] = 1
    assert_equal(skeletonize(im), im)
    assert_equal(skeletonize(im, periodic=True), im)


def test_dtype_conv():
    # check that the operation does the right thing with floats etc
    # also check non-contiguous input
    img = np.random.random((16, 16))[::2, ::2]
    img[img < 0.5] = 0

    orig = img.copy()
    res = skeletonize(img)
    img_max = img_as_ubyte(img).max()

    assert_equal(res.dtype, np.uint8)
    assert_equal(img, orig)  # operation does not clobber the original