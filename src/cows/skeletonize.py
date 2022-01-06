'''
Contains code from scikit-image v0.18.3
'''

import numpy as np
from .dtype import img_as_ubyte
from .arraycrop import crop

from ._skeletonize_3d_cy import _compute_thin_image


def skeletonize(image, periodic=False):
    """Compute the skeleton of a binary image.

    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.

    Parameters
    ----------
    image : ndarray, 2D or 3D
        A binary image containing the objects to be skeletonized. Zeros
        represent background, nonzero values are foreground.
    periodic: bool
        If True, the skeletonization uses periodic boundary conditions 
        for the input array. Input array must be 3D.

    Returns
    -------
    skeleton : ndarray
        The thinned image.

    Notes
    -----
    The method of [Lee94]_ uses an octree data structure to examine a 3x3x3
    neighborhood of a pixel. The algorithm proceeds by iteratively sweeping
    over the image, and removing pixels at each iteration until the image
    stops changing. Each iteration consists of two steps: first, a list of
    candidates for removal is assembled; then pixels from this list are
    rechecked sequentially, to better preserve connectivity of the image.

    References
    ----------
    .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

    """
    # make sure the image is 3D or 2D
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError("skeletonize can only handle 2D or 3D images; "
                         "got image.ndim = %s instead." % image.ndim)
    image = np.ascontiguousarray(image)
    image = img_as_ubyte(image, force_copy=False)

    if type(periodic) != bool:
        raise TypeError("keyword argument periodic must of of type bool; "
                        "got type %s instead." % type(periodic))
    if periodic and image.ndim != 3:
        raise ValueError("periodic boundaries currently only work for 3D "
                         "data. image.ndim = %s." % image.ndim)

    # make an in image 3D and pad it w/ zeros to simplify dealing w/ boundaries
    # NB: careful here to not clobber the original *and* minimize copying
    image_o = image
    if image.ndim == 2:
        image_o = image[np.newaxis, ...]
    image_o = np.pad(image_o, pad_width=1, mode='constant')

    # normalize to binary
    # maxval = image_o.max()
    image_o[image_o != 0] = 1

    # do the computation
    image_o = np.asarray(_compute_thin_image(image_o, periodic=periodic))

    # crop it back and restore the original intensity range
    image_o = crop(image_o, crop_width=1)
    if image.ndim == 2:
        image_o = image_o[0]
    # image_o *= maxval

    return image_o