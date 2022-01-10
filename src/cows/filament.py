import numpy as np
from ._filament import _label_skeleton, _find_filaments


def label_skeleton(skel, periodic=False):
    ''' Label the skeleton.

        Label all skeleton cells with their respective number of neighbour
        that they share a face, edge or vertex with (N_26).

    Parameters
    ----------
    skel : ndarray, 3D
        A binary image containing the skeletonized objects. Zeros
        represent background, nonzero values are foreground.
    periodic: bool
        If True, the skeletonization uses periodic boundary conditions 
        for the input array. Input array must be 3D.

    Returns
    -------
    result : ndarray
        The labeled skeleton.
    '''
    assert skel.ndim == 3

    return _label_skeleton(skel, periodic)


def separate_skeleton(skel, periodic=False):
    ''' Separate the skeleton.

        Set all the skeleton cells with more than 2 neighbours to the
        background value of zero. This results in a set of individual
        objects of arbitrary length and 2 endpoints.

    Parameters
    ----------
    skel : ndarray, 3D
        A binary image containing the skeletonized objects. Zeros
        represent background, nonzero values are foreground.
    periodic: bool
        If True, the skeletonization uses periodic boundary conditions 
        for the input array. Input array must be 3D.

    Returns
    -------
    result : ndarray
        The separated skeleton.
    '''
    assert skel.ndim == 3

    # Label the skeleton
    skel = _label_skeleton(skel, periodic)
    
    # Remove all cells with more than two neighbours
    data_shape = skel.shape
    skel[skel>2] = 0

    # Label the separated skeleton
    skel = _label_skeleton(skel, periodic)
    
    return skel


def find_filaments(skel, periodic=False):
    ''' Find individual filament.

        Connects all cells that are neighbours within a 3x3x3 neihbourhood.
        The set of connected cells are labled with a unique ID.

        Parameters
        ----------
        skel : ndarray, 3D
            An array containing the classified and separated skeleton. Zeros
            represent background, ones are endpoints and twos are regular
            cells.
        periodic: bool
            If True, the skeletonization uses periodic boundary conditions 
            for the input array. Input array must be 3D.

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

    return _find_filaments(skel, periodic)
