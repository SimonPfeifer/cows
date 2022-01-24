# distutils: language = c++

'''
Contains code from scikit-image v0.18.3
'''

"""
This is an implementation of the 2D/3D thinning algorithm
of [Lee94]_ of binary images, based on [IAC15]_.

The original Java code [IAC15]_ carries the following message:

 * This work is an implementation by Ignacio Arganda-Carreras of the
 * 3D thinning algorithm from Lee et al. "Building skeleton models via 3-D
 * medial surface/axis thinning algorithms. Computer Vision, Graphics, and
 * Image Processing, 56(6):462-478, 1994." Based on the ITK version from
 * Hanno Homann <a href="http://hdl.handle.net/1926/1292"> http://hdl.handle.net/1926/1292</a>
 * <p>
 *  More information at Skeletonize3D homepage:
 *  https://imagej.net/Skeletonize3D
 *
 * @version 1.0 11/13/2015 (unique BSD licensed version for scikit-image)
 * @author Ignacio Arganda-Carreras (iargandacarreras at gmail.com)

References
----------
.. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
       via 3-D medial surface/axis thinning algorithms.
       Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

.. [IAC15] Ignacio Arganda-Carreras, 2015. Skeletonize3D plugin for ImageJ(C).
           https://imagej.net/Skeletonize3D

"""

from libc.string cimport memcpy
from libcpp.vector cimport vector

import numpy as np
from numpy cimport npy_intp, npy_uint8, ndarray
cimport cython

ctypedef npy_uint8 pixel_type

# struct to hold 3D coordinates
cdef struct coordinate:
    npy_intp p
    npy_intp r
    npy_intp c


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_thin_image(pixel_type[:, :, ::1] img not None,
                        surface= False,
                        periodic=False):
    """Compute a thin image.

    Loop through the image multiple times, removing "simple" points, i.e.
    those point which can be removed without changing local connectivity in the
    3x3x3 neighborhood of a point.

    This routine implements the two-pass algorithm of [Lee94]_. Namely,
    for each of the six border types (positive and negative x-, y- and z-),
    the algorithm first collects all possibly deletable points, and then
    performs a sequential rechecking.

    The input, `img`, is assumed to be a 3D binary image in the
    (p, r, c) format [i.e., C ordered array], filled by zeros (background) and
    ones. 

    If periodic=False (default), `img` is assumed to be padded by zeros from all
    directions --- this way the zero boundary conditions are automatic
    and there is need to guard against out-of-bounds access. Else, `img` is
    padded with values from the opposite side of the 3D array creating periodic
    boundary conditions. Padding is updated every cycle.

    """
    cdef:
        int unchanged_borders = 0, curr_border, num_borders
        int borders[6]
        npy_intp p, r, c
        npy_intp imax, jmax, kmax
        bint no_change

        bint surface_flag, periodic_flag

        # list simple_border_points
        vector[coordinate] simple_border_points
        coordinate point

        Py_ssize_t num_border_points, i, j

        pixel_type neighb[27]

    # define the flag to be passed on
    surface_flag = int(surface)
    periodic_flag = int(periodic)

    # loop over the six directions in this order (for consistency with ImageJ)
    # borders[:] = [4, 3, 2, 1, 5, 6]
    borders[:] = [5, 6, 1, 2, 4, 3] # 1=N, 2=S, 3=E, 4=W, 5=U, 6=B

    # no need to worry about the z direction if the original image is 2D.
    if img.shape[0] == 3:
        num_borders = 4
    else:
        num_borders = 6
        kmax = img.shape[0]
        jmax = img.shape[1]
        imax = img.shape[2]

    # with nogil:
    # loop through the image several times until there is no change for all
    # the six border types
    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):

            # Periodic boundary conditions were added by SP
            if periodic_flag == 1:

                # with gil:
                    img = np.pad(np.asarray(img[1:kmax-1,1:jmax-1,1:imax-1]),
                                 1, 
                                 mode='wrap')

            curr_border = borders[j]

            find_simple_point_candidates(img,
                                         curr_border,
                                         simple_border_points,
                                         surface_flag)

            # sequential re-checking to preserve connectivity when deleting
            # in a parallel way
            no_change = True
            num_border_points = simple_border_points.size()
            for i in range(num_border_points):
                point = simple_border_points[i]
                p = point.p
                r = point.r
                c = point.c
                get_neighborhood(img, p, r, c, neighb)
                if is_simple_point(neighb):
                    img[p, r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1

    return np.asarray(img)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_simple_point_candidates(pixel_type[:, :, ::1] img,
                                       int curr_border,
                                       vector[coordinate] & simple_border_points,
                                       bint surface_flag):# nogil:
    """Inner loop of compute_thin_image.

    The algorithm of [Lee94]_ proceeds in two steps: (1) six directions are
    checked for simple border points to remove, and (2) these candidates are
    sequentially rechecked, see Sec 3 of [Lee94]_ for rationale and discussion.

    This routine implements the first step above: it loops over the image
    for a given direction and assembles candidates for removal.

    """
    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp p, r, c
        bint is_border_pt

        # rebind a global name to avoid lookup. The table is filled in
        # at import time.
        int[::1] Euler_LUT = LUT
        int[:, ::1] NO_idx = NOI
        int[:, ::1] NO_idx_old = NOIOLD

    # clear the output vector
    simple_border_points.clear();

    # loop through the image
    # NB: each loop is from 1 to size-1: img is padded from all sides
    for p in range(1, img.shape[0] - 1):
        for r in range(1, img.shape[1] - 1):
            for c in range(1, img.shape[2] - 1):

                # check if pixel is foreground
                if img[p, r, c] != 1:
                    continue

                is_border_pt = (curr_border == 1 and img[p, r, c-1] == 0 or  #N
                                curr_border == 2 and img[p, r, c+1] == 0 or  #S
                                curr_border == 3 and img[p, r+1, c] == 0 or  #E
                                curr_border == 4 and img[p, r-1, c] == 0 or  #W
                                curr_border == 5 and img[p+1, r, c] == 0 or  #U
                                curr_border == 6 and img[p-1, r, c] == 0)    #B
                if not is_border_pt:
                    # current point is not deletable
                    continue

                get_neighborhood(img, p, r, c, neighborhood)

                # check if (p, r, c) can be deleted:
                # * it must be Euler invariant (condition 1 in [Lee94]_); and
                # * it must be simple (i.e., its deletion does not change
                #   connectivity in the 3x3x3 neighborhood; conditions 2 and 
                #   3 in [Lee94]_); and
                # * it must not be an endpoint for medial axis thinning or
                #   a surface endpoint for medial surface thinning
                #   (condition 4 in [Lee94]_)
                if (not is_Euler_invariant(neighborhood, NO_idx, NO_idx_old, Euler_LUT) or
                    not is_simple_point(neighborhood)):
                    continue
                if (surface_flag == 0 and
                    is_endpoint(neighborhood)):
                    print('A hello')
                    continue
                elif (surface_flag == 1 and 
                      is_surface_endpoint(neighborhood, NO_idx, NO_idx_old)):
                    print("S hello")
                    continue

                # ok, add (p, r, c) to the list of simple border points
                point.p = p
                point.r = r
                point.c = c
                simple_border_points.push_back(point)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_neighborhood(pixel_type[:, :, ::1] img,
                           npy_intp p, npy_intp r, npy_intp c,
                           pixel_type neighborhood[]) nogil:
    """Get the neighborhood of a pixel.

    Assume zero boundary conditions.
    Image is already padded, so no out-of-bounds checking.

    For the numbering of points see Fig. 1a. of [Lee94]_, where the numbers
    do *not* include the center point itself. OTOH, this numbering below
    includes it as number 13. The latter is consistent with [IAC15]_.
    """
    neighborhood[0] = img[p-1, r-1, c-1]
    neighborhood[1] = img[p-1, r,   c-1]
    neighborhood[2] = img[p-1, r+1, c-1]

    neighborhood[ 3] = img[p-1, r-1, c]
    neighborhood[ 4] = img[p-1, r,   c]
    neighborhood[ 5] = img[p-1, r+1, c]

    neighborhood[ 6] = img[p-1, r-1, c+1]
    neighborhood[ 7] = img[p-1, r,   c+1]
    neighborhood[ 8] = img[p-1, r+1, c+1]

    neighborhood[ 9] = img[p, r-1, c-1]
    neighborhood[10] = img[p, r,   c-1]
    neighborhood[11] = img[p, r+1, c-1]

    neighborhood[12] = img[p, r-1, c]
    neighborhood[13] = img[p, r,   c]
    neighborhood[14] = img[p, r+1, c]

    neighborhood[15] = img[p, r-1, c+1]
    neighborhood[16] = img[p, r,   c+1]
    neighborhood[17] = img[p, r+1, c+1]

    neighborhood[18] = img[p+1, r-1, c-1]
    neighborhood[19] = img[p+1, r,   c-1]
    neighborhood[20] = img[p+1, r+1, c-1]

    neighborhood[21] = img[p+1, r-1, c]
    neighborhood[22] = img[p+1, r,   c]
    neighborhood[23] = img[p+1, r+1, c]

    neighborhood[24] = img[p+1, r-1, c+1]
    neighborhood[25] = img[p+1, r,   c+1]
    neighborhood[26] = img[p+1, r+1, c+1]


# Fill the look-up table for indexing octants for computing the Euler
# characteristic. See is_Euler_invariant routine below.
def fill_Euler_LUT():
    """ Look-up table for preserving Euler characteristic.

    This is column $\delta G_{26}$ of Table 2 of [Lee94]_.
    """
    cdef int arr[128]
    arr[:] = [1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1,
                 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,
                 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1,
                 -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, -1, -1, 1, -1,
                 1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1,
                 -1, 3, 1, 1, -1, -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3,
                 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1]
    cdef ndarray LUT = np.zeros(256, dtype=np.intc)
    LUT[1::2] = arr
    return LUT
cdef int[::1] LUT = fill_Euler_LUT()


def gen_neighborhood_octant_indices(old=False):
    """ The indices of points in a N_26 neighborhood for all 8 octants.

    It assumes that the centre point in the neighborhood (index 13) is
    included and is always in the final position.
    """
    cdef ndarray arr = np.zeros((8,7), dtype=np.intc)
    if old:
        arr[:] = [[ 2,  1, 11, 10,  5,  4, 14], # octant 0
                  [ 0,  9,  3, 12,  1, 10,  4], # octant 1
                  [ 8,  7, 17, 16,  5,  4, 14], # octant 2
                  [ 6, 15,  7, 16,  3, 12,  4], # octant 3
                  [20, 23, 19, 22, 11, 14, 10], # octant 4
                  [18, 21,  9, 12, 19, 22, 10], # octant 5
                  [26, 23, 17, 14, 25, 22, 16], # octant 6
                  [24, 25, 15, 16, 21, 22, 12]] # octant 7
    else:    
        arr[:] = [[ 2,  1, 11, 10,  5,  4, 14], # octant 0
                  [ 0,  3,  9, 12,  1,  4, 10], # octant 1
                  [ 8,  5, 17, 14,  7,  4, 16], # octant 2
                  [ 6,  7, 15, 16,  3,  4, 12], # octant 3
                  [20, 23, 11, 14, 19, 22, 10], # octant 4
                  [18, 21, 19, 22,  9, 12, 10], # octant 5
                  [26, 17, 23, 14, 25, 16, 22], # octant 6
                  [24, 25, 21, 22, 15, 16, 12]] # octant 7
    return arr
cdef int[:, ::1] NOI = gen_neighborhood_octant_indices()
cdef int[:, ::1] NOIOLD = gen_neighborhood_octant_indices(True)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int octant_decimal(pixel_type octant[]) nogil:
    """Calculate the decimal index of an octant configuration.
    
    Parameters
    ----------
    octant
        List of points in an octant. List must be ordered by
        appropriate binary octant order.

    Returns
    -------
    octant_index (int)
        The decimal index of the octant

    Notes
    -----
    This function assumes that the last point of the octant,
    in binary order, is always part of the foreground. This point 
    corresponds to the centre point in the N_26 neighbourhood.

    """

    cdef int octant_index = 1
    cdef int increment[7]
    increment[0] = 128
    increment[1] = 64
    increment[2] = 32
    increment[3] = 16
    increment[4] = 8
    increment[5] = 4
    increment[6] = 2

    for i in range(7): # assume last point is foreground
        if octant[i] == 1:
            octant_index |= increment[i]

    return octant_index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_Euler_invariant(pixel_type neighbors[],
                             int[:, ::1] noi,
                             int[:, ::1] noi_old,
                             int[::1] lut): #nogil:
    """Check if a point is Euler invariant.

    Calculate Euler characteristic for each octant and sum up.

    Parameters
    ----------
    neighbors
        neighbors of a point
    lut
        The look-up table for preserving the Euler characteristic.

    Returns
    -------
    bool (C bool, that is)

    """
    cdef int octant_index
    cdef int euler_char = 0
    cdef int euler_char2 = 0
    cdef pixel_type octant[7]
    cdef pixel_type octant2[7]
    cdef npy_intp i_oct, j_pnt

    for i_oct in range(8): # loop over octants
        for j_pnt in range(7): # loop over points
            octant[j_pnt] = neighbors[noi[i_oct,j_pnt]]
            octant2[j_pnt] = neighbors[noi_old[i_oct,j_pnt]]
        if lut[octant_decimal(octant)] != lut[octant_decimal(octant2)]:
            print('Euler octant {i_ict} mismatch:', lut[octant_decimal(octant)],
                                                    lut[octant_decimal(octant2)])
        euler_char += lut[octant_decimal(octant)]
        euler_char2 += lut[octant_decimal(octant2)]

    if euler_char != euler_char2:
        print('Euler:', euler_char, euler_char2, euler_char==euler_char2)

    return euler_char2 == 0


cdef inline bint is_endpoint(pixel_type neighbors[]) nogil:
    """An endpoint has exactly one neighbor in the 26-neighborhood.
    """
    # The center pixel is counted, thus r.h.s. is 2
    cdef int s = 0, j
    for j in range(27):
        s += neighbors[j]
    return s == 2

cdef bint is_surface_endpoint(pixel_type neighbors[],
                              int[:, ::1] noi,
                              int[:, ::1] noi_old): #nogil:
    """Check if a point is a surface endpoint.

    Check if each octant index is part of a set [240, 165, 170,
    204] and each octant contains less than 3 foreground points.
    This is Definition 1 in [Lee94]_.

    Parameters
    ----------
    neighbors
        Neighbors of a point.

    Returns
    -------
    bool
        Whether the point is a surface endpoint.

    """
    
    cdef int n_octant, n_octant2
    cdef int octant_index, octant_index2
    cdef pixel_type octant[7], octant2[7]

    cdef int euler_char = 0, euler_char2 = 0
    cdef int[::1] lut = fill_Euler_LUT()

    cdef int is_sep = 1, is_sep2 = 1

    cdef npy_intp i_oct, j_pnt

    for i_oct in range(8): # loop over octants
        n_octant = 1
        n_octant2 = 1
        for j_pnt in range(7): # loop over points
            # fill octant
            octant[j_pnt] = neighbors[noi[i_oct,j_pnt]]
            octant2[j_pnt] = neighbors[noi_old[i_oct,j_pnt]]
            # count point in octant
            n_octant += octant[j_pnt]
            n_octant2 += octant2[j_pnt]


        # get decimal index
        octant_index = octant_decimal(octant)
        octant_index2 = octant_decimal(octant2)
        if lut[octant_index] != lut[octant_index2]: # check Euler mismatch
            print('Euler octant {i_ict} mismatch:', lut[octant_index],
                                                    lut[octant_index2])
        # sum up Euler
        euler_char += lut[octant_index]
        euler_char2 += lut[octant_index2]

        
        if (octant_index != 15 and octant_index != 165 and
            octant_index != 85 and octant_index != 51 and
            n_octant >= 4):
            is_sep = 0
            return False
        if (octant_index2 != 15 and octant_index2 != 165 and
            octant_index2 != 85 and octant_index2 != 51 and
            n_octant2 >= 4):
            is_sep2 = 0

    if euler_char != euler_char2:
        print('Euler:', euler_char, euler_char2, euler_char==euler_char2)

    if is_sep != is_sep2:
        print('Surface endpoint mismatch:', is_sep, is_sep2)

    if is_sep:
        return True
    else:
        return False


cdef bint is_simple_point(pixel_type neighbors[]) nogil:
    """Check is a point is a Simple Point.

    A point is simple if its deletion does not change connectivity in
    the 3x3x3 neighborhood. (cf conditions 2 and 3 in [Lee94]_).

    This method is named "N(v)_labeling" in [Lee94]_.

    Parameters
    ----------
    neighbors : uint8 C array, shape(27,)
        neighbors of the point

    Returns
    -------
    bool
        Whether the point is simple or not.

    """
    # copy neighbors for labeling
    # ignore center pixel (i=13) when counting (see [Lee94]_)
    cdef pixel_type cube[26]
    memcpy(cube, neighbors, 13*sizeof(pixel_type))
    memcpy(cube+13, neighbors+14, 13*sizeof(pixel_type))

    # set initial label
    cdef int label = 2, i

    # for all point in the neighborhood
    for i in range(26):
        if cube[i] == 1:
            # voxel has not been labeled yet
            # start recursion with any octant that contains the point i
            if i in (0, 1, 3, 4, 9, 10, 12):
                octree_labeling(1, label, cube)
            elif i in (2, 5, 11, 13):
                octree_labeling(2, label, cube)
            elif i in (6, 7, 14, 15):
                octree_labeling(3, label, cube)
            elif i in (8, 16):
                octree_labeling(4, label, cube)
            elif i in (17, 18, 20, 21):
                octree_labeling(5, label, cube)
            elif i in (19, 22):
                octree_labeling(6, label, cube)
            elif i in (23, 24):
                octree_labeling(7, label, cube)
            elif i == 25:
                octree_labeling(8, label, cube)
            label += 1
            if label - 2 >= 2:
                return False
    return True


# Octree structure for labeling in `octree_labeling` routine below.
# NB: this is only available at build time, and is used by Tempita templating.

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void octree_labeling(int octant, int label, pixel_type cube[]) nogil:
    """This is a recursive method that calculates the number of connected
    components in the 3D neighborhood after the center pixel would
    have been removed.

    See Figs. 6 and 7 of [Lee94]_ for the values of indices.

    Parameters
    ----------
    octant : int
        octant index
    label : int
        the current label of the center point
    cube : uint8 C array, shape(26,)
        local neighborhood of the point

    """
    # This routine checks if there are points in the octant with value 1
    # Then sets points in this octant to current label
    # and recursive labeling of adjacent octants.
    #
    # Below, leading underscore means build-time variables.

    if octant == 1:
        if cube[0] == 1:
            cube[0] = label
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(2, label, cube)
        if cube[3] == 1:
            cube[3] = label
            octree_labeling(3, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(2, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
        if cube[9] == 1:
            cube[9] = label
            octree_labeling(5, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(2, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(3, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)

    if octant == 2:
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(1, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
        if cube[2] == 1:
            cube[2] = label
        if cube[5] == 1:
            cube[5] = label
            octree_labeling(4, label, cube)
        if cube[11] == 1:
            cube[11] = label
            octree_labeling(6, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(4, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)

    if octant == 3:
        if cube[3] == 1:
            cube[3] = label
            octree_labeling(1, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)
        if cube[6] == 1:
            cube[6] = label
        if cube[7] == 1:
            cube[7] = label
            octree_labeling(4, label, cube)
        if cube[14] == 1:
            cube[14] = label
            octree_labeling(7, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(4, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)

    if octant == 4:
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(3, label, cube)
        if cube[5] == 1:
            cube[5] = label
            octree_labeling(2, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)
        if cube[7] == 1:
            cube[7] = label
            octree_labeling(3, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)
        if cube[8] == 1:
            cube[8] = label
        if cube[16] == 1:
            cube[16] = label
            octree_labeling(8, label, cube)

    if octant == 5:
        if cube[9] == 1:
            cube[9] = label
            octree_labeling(1, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(6, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(7, label, cube)
        if cube[17] == 1:
            cube[17] = label
        if cube[18] == 1:
            cube[18] = label
            octree_labeling(6, label, cube)
        if cube[20] == 1:
            cube[20] = label
            octree_labeling(7, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(6, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)

    if octant == 6:
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(5, label, cube)
        if cube[11] == 1:
            cube[11] = label
            octree_labeling(2, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(8, label, cube)
        if cube[18] == 1:
            cube[18] = label
            octree_labeling(5, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)
        if cube[19] == 1:
            cube[19] = label
        if cube[22] == 1:
            cube[22] = label
            octree_labeling(8, label, cube)

    if octant == 7:
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(5, label, cube)
        if cube[14] == 1:
            cube[14] = label
            octree_labeling(3, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(8, label, cube)
        if cube[20] == 1:
            cube[20] = label
            octree_labeling(5, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)
        if cube[23] == 1:
            cube[23] = label
        if cube[24] == 1:
            cube[24] = label
            octree_labeling(8, label, cube)

    if octant == 8:
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(6, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(7, label, cube)
        if cube[16] == 1:
            cube[16] = label
            octree_labeling(4, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(7, label, cube)
        if cube[22] == 1:
            cube[22] = label
            octree_labeling(6, label, cube)
        if cube[24] == 1:
            cube[24] = label
            octree_labeling(7, label, cube)
        if cube[25] == 1:
            cube[25] = label
