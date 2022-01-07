import numpy as np
from ._filament import _label_skeleton, _find_filaments


def label_skeleton(skel):
    ''' Label the skeleton.

        Label all skeleton cells with their respective number of neighbour.
        Also removes cells with zero or more than four neighbours by setting
        them to the background value of zero.

    Parameters
    ----------
    skel : ndarray, 3D
        A binary image containing the skeletonized objects. Zeros
        represent background, nonzero values are foreground.

    Returns
    -------
    result : ndarray
        The labeled skeleton.
    '''
    assert skel.ndim == 3

    return _label_skeleton(skel)

def separate_skeleton(skel):
    ''' Separate the skeleton.

        Set all the skeleton cells with more than 2 neighbours to the
        background value of zero. This results in a set of individual
        objects of arbitrary length and 2 endpoints.

    Parameters
    ----------
    skel : ndarray, 3D
        A binary image containing the skeletonized objects. Zeros
        represent background, nonzero values are foreground.

    Returns
    -------
    result : ndarray
        The separated skeleton.
    '''
    assert skel.ndim == 3

    # Label the skeleton
    skel = _label_skeleton(skel)
    
    # Remove all cells with more than two neighbours
    data_shape = skel.shape
    skel[skel>2] = 0
    
    return skel.reshape(data_shape)

def find_filaments(skel):
    ''' Find individual filament.

        Connects all cells that are neighbours within a 3x3x3 neihbourhood.
        The set of connected cells are labled with a unique ID.

        Parameters
        ----------
        skel : ndarray, 3D
            An array containing the classified and separated skeleton. Zeros
            represent background, ones are endpoints and twos are regular
            cells.

        Returns
        -------
        result : ndarray, 3D
            An array with skel.shape containing the sets of connected cells
            (filaments) with their respective ID.
        catalogue : ndarray, 2D
            A catalogue containing, for each cell, a row of ID, X-, Y- and Z-
            position.
    '''
    assert skel.ndim == 3
    assert skel.shape[0] == skel.shape[1]
    assert skel.shape[0] == skel.shape[2]

    return _find_filaments(skel)
