import pytest

import numpy as np
from numpy.testing import assert_equal

from cows import gen_catalogue


def test_gen_catalogue_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=False)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=True)

    im = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=False)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=True)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=False)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=True)


def test_gen_catalogue_wrong_shape():
    im = np.zeros((1,2,3), dtype=np.int8)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=False)
    with pytest.raises(AssertionError):
        gen_catalogue(im, periodic=True)


def test_gen_catalogue_no_foreground():
    im = np.zeros((3, 3, 3))
    assert_equal(gen_catalogue(im, periodic=False, sort=True),
                 np.zeros((0, 8)))
    assert_equal(gen_catalogue(im, periodic=True, sort=True),
                 np.zeros((0, 8)))
    assert_equal(gen_catalogue(im, periodic=False, sort=False), 
                 np.zeros((0, 8)))
    assert_equal(gen_catalogue(im, periodic=True, sort=False),
                 np.zeros((0, 8)))


def test_gen_catalogue_single_line():
    im = np.zeros((5, 5, 5))
    im[2, 2, 2:4] = 2
    im[2, 2, 1] = 1
    im[2, 2, 4] = 1

    fil_correct = im = np.zeros((5, 5, 5))
    fil_correct[2, 2, 1:] = 1
    assert_equal(gen_catalogue(im, periodic=False, sort=True), 
                 np.array([[1, 4, 4, 2, 2, -1, 0, 0],
                           [1, 4, 3, 2, 2, -1, 0, 0],
                           [1, 4, 2, 2, 2, -1, 0, 0],
                           [1, 4, 1, 2, 2, -1, 0, 0]], dtype=np.float64))
    assert_equal(gen_catalogue(im, periodic=False, sort=False), 
                 np.array([[1, 4, 1, 2, 2, -1, 0, 0],
                           [1, 4, 2, 2, 2, -1, 0, 0],
                           [1, 4, 3, 2, 2, -1, 0, 0],
                           [1, 4, 4, 2, 2, -1, 0, 0]], dtype=np.float64))

    im = np.zeros((5, 5, 5))
    im[2, 2, 3:] = 2
    im[2, 2, 2] = 1
    im[2, 2, 0] = 1

    fil_correct = im = np.zeros((5, 5, 5))
    fil_correct[2, 2, 2:] = 1
    fil_correct[2, 2, 0] = 1
    assert_equal(gen_catalogue(im, periodic=True, sort=True), 
                 np.array([[1, 4, 2, 2, 2, 1, 0, 0],
                           [1, 4, 3, 2, 2, 1, 0, 0],
                           [1, 4, 4, 2, 2, 1, 0, 0],
                           [1, 4, 0, 2, 2, 1, 0, 0]], dtype=np.float64))
    assert_equal(gen_catalogue(im, periodic=True, sort=False), 
                 np.array([[1, 4, 0, 2, 2, 1, 0, 0],
                           [1, 4, 4, 2, 2, 1, 0, 0],
                           [1, 4, 3, 2, 2, 1, 0, 0],
                           [1, 4, 2, 2, 2, 1, 0, 0]], dtype=np.float64))


def test_gen_catalogue_double_line():
    im = np.zeros((5, 5, 5))
    # First line
    im[2, 0, :3] = 1 
    im[2, 0, 1] = 2
    # Second line
    im[2, 2, 2:4] = 2 
    im[2, 2, 1] = 1
    im[2, 2, 4] = 1
    assert_equal(gen_catalogue(im, periodic=False, sort=True), 
                 np.array([[1, 4, 4, 2, 2, -1, 0, 0],
                           [1, 4, 3, 2, 2, -1, 0, 0],
                           [1, 4, 2, 2, 2, -1, 0, 0],
                           [1, 4, 1, 2, 2, -1, 0, 0],
                           [2, 3, 2, 0, 2, -1, 0, 0],
                           [2, 3, 1, 0, 2, -1, 0, 0],
                           [2, 3, 0, 0, 2, -1, 0, 0]], dtype=np.float64))
    assert_equal(gen_catalogue(im, periodic=False, sort=False), 
                 np.array([[1, 3, 0, 0, 2, -1, 0, 0],
                           [1, 3, 1, 0, 2, -1, 0, 0],
                           [1, 3, 2, 0, 2, -1, 0, 0],
                           [2, 4, 1, 2, 2, -1, 0, 0],
                           [2, 4, 2, 2, 2, -1, 0, 0],
                           [2, 4, 3, 2, 2, -1, 0, 0],
                           [2, 4, 4, 2, 2, -1, 0, 0]], dtype=np.float64))

    im = np.zeros((5, 5, 5))
    # First line
    im[2, 0, :3] = 1
    im[2, 0, 1] = 2 
    # Second line
    im[2, 2, 3:] = 2 
    im[2, 2, 2] = 1
    im[2, 2, 0] = 1
    print(gen_catalogue(im, periodic=True, sort=True))
    print(gen_catalogue(im, periodic=True, sort=False))
    assert_equal(gen_catalogue(im, periodic=True, sort=True), 
                 np.array([[1, 4, 2, 2, 2,  1, 0, 0],
                           [1, 4, 3, 2, 2,  1, 0, 0],
                           [1, 4, 4, 2, 2,  1, 0, 0],
                           [1, 4, 0, 2, 2,  1, 0, 0],
                           [2, 3, 2, 0, 2, -1, 0, 0],
                           [2, 3, 1, 0, 2, -1, 0, 0],
                           [2, 3, 0, 0, 2, -1, 0, 0]], dtype=np.float64))
    assert_equal(gen_catalogue(im, periodic=True, sort=False), 
                 np.array([[1, 3, 0, 0, 2, -1, 0, 0],
                           [1, 3, 1, 0, 2, -1, 0, 0],
                           [1, 3, 2, 0, 2, -1, 0, 0],
                           [2, 4, 0, 2, 2,  1, 0, 0],
                           [2, 4, 4, 2, 2,  1, 0, 0],
                           [2, 4, 3, 2, 2,  1, 0, 0],
                           [2, 4, 2, 2, 2,  1, 0, 0]], dtype=np.float64))