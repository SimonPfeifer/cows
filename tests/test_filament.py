import pytest

import numpy as np
from numpy.testing import assert_equal

from cows import label_skeleton, separate_skeleton, find_filaments


def test_label_skeleton_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=True)

    im = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=True)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        label_skeleton(im, periodic=True)


def test_label_skeleton_no_foreground():
    im = np.zeros((3, 3, 3))
    assert_equal(label_skeleton(im, periodic=False), im)
    assert_equal(label_skeleton(im, periodic=True), im)


def test_label_skeleton_all_foreground():
    im = np.ones((3, 3, 3))
    assert_equal(label_skeleton(im, periodic=False),
                 np.array([[[7 , 11,  7],
                            [11, 17, 11],
                            [7 , 11,  7]],
                           
                           [[11, 17, 11],
                            [17, 26, 17],
                            [11, 17, 11]],
 
                           [[7 , 11,  7],
                            [11, 17, 11],
                            [7 , 11,  7]]]))
    assert_equal(label_skeleton(im, periodic=True), 26 * im)


def test_label_skeleton_single_line():
    im = np.zeros((3, 3, 5))
    im[1, 1, :] = 1
    assert_equal(label_skeleton(im, periodic=False),
                 np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [1, 2, 2, 2, 1],
                            [0, 0, 0, 0, 0]],
 
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))
    assert_equal(label_skeleton(im, periodic=True),
                 np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0]],
 
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))


def test_separate_skeleton_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=True)

    im = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=True)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=False)
    with pytest.raises(AssertionError):
        separate_skeleton(im, periodic=True)


def test_separate_skeleton_no_foreground():
    im = np.zeros((3, 3, 3))
    assert_equal(separate_skeleton(im, periodic=False), im)
    assert_equal(separate_skeleton(im, periodic=True), im)


def test_separate_skeleton_all_foreground():
    im = np.ones((3, 3, 3))
    arr_zero = np.zeros(im.shape)
    assert_equal(separate_skeleton(im, periodic=False), arr_zero)
    assert_equal(separate_skeleton(im, periodic=True), arr_zero)


def test_separate_skeleton_single_line():
    im = np.zeros((3, 3, 3))
    im[1, 1, :] = 1
    assert_equal(separate_skeleton(im, periodic=False),
                 np.array([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           
                           [[0, 0, 0],
                            [1, 2, 1],
                            [0, 0, 0]],
 
                           [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]]))
    assert_equal(separate_skeleton(im, periodic=True),
                 np.array([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           
                           [[0, 0, 0],
                            [2, 2, 2],
                            [0, 0, 0]],
 
                           [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]]))

def test_separate_skeleton_double_line():
    im = np.zeros((5, 5, 5))
    im[2, 3, :] = 1
    im[2, :, 3] = 1
    assert_equal(separate_skeleton(im, periodic=False),
                 np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))

    im = np.zeros((5, 5, 5))
    im[2, 2, :] = 1
    im[2, :, 2] = 1
    assert_equal(separate_skeleton(im, periodic=True),
                 np.array([[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0]],
                           
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],

                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))


def test_find_filaments_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=False)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=True)

    im = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=False)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=True)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=False)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=True)


def test_find_filaments_wrong_shape():
    im = np.zeros((1,2,3), dtype=np.int8)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=False)
    with pytest.raises(AssertionError):
        find_filaments(im, periodic=True)


def test_find_filaments_no_foreground():
    im = np.zeros((3, 3, 3))
    fil, cat = find_filaments(im, periodic=False)
    assert_equal(fil, im)
    assert_equal(cat, np.zeros((0, 4)))

    fil, cat = find_filaments(im, periodic=True)
    assert_equal(fil, im)
    assert_equal(cat, np.zeros((0, 4)))


def test_find_filaments_single_line():
    im = np.zeros((5, 5, 5))
    im[2, 2, 2:4] = 2
    im[2, 2, 1] = 1
    im[2, 2, 4] = 1

    fil_correct = im = np.zeros((5, 5, 5))
    fil_correct[2, 2, 1:] = 1
    fil, cat = find_filaments(im, periodic=False)
    assert_equal(fil, fil_correct)
    assert_equal(cat, np.array([[1, 1, 2, 2],
                                [1, 2, 2, 2],
                                [1, 3, 2, 2],
                                [1, 4, 2, 2]]))

    im = np.zeros((5, 5, 5))
    im[2, 2, 3:] = 2
    im[2, 2, 2] = 1
    im[2, 2, 0] = 1

    fil_correct = im = np.zeros((5, 5, 5))
    fil_correct[2, 2, 2:] = 1
    fil_correct[2, 2, 0] = 1
    fil, cat = find_filaments(im, periodic=True)
    assert_equal(fil, fil_correct)
    assert_equal(cat, np.array([[1, 0, 2, 2],
                                [1, 4, 2, 2],
                                [1, 3, 2, 2],
                                [1, 2, 2, 2]]))

