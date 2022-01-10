import numpy as np

from .filament import label_skeleton, find_filaments


def gen_catalogue(data, periodic=False, sort=True):
    ''' Generate a catalogue of filaments

        Generates a catalogue of filaments given an array containing a
        cleaned and separated skeleton. 

        Parameters
        ----------
        data : ndarray, 3D
            An array containing the separated skeleton. Zeros represent 
            background, ones are endpoints and, twos are regular cells.
        periodic: bool
            If True, the skeletonization uses periodic boundary conditions 
            for the input array. Input array must be 3D.
        sort : boolean
            If sort=True, the filaments are sorted by filament length in
            descending order and reassigned IDs such that the longest filament
            has an ID of zero.

        Returns
        -------
        result : ndarray, 3D
            An array with data.shape containing the sets of connected cells
            with their respective ID.
        catalogue : ndarray, 2D
            A catalogue containing, for each cell, a row of filament ID, 
            filament length, X-, Y-, Z-position, X-, Y-, Z-direction.

        Notes
        -----
        The function assumes that values larger than zero are part of the 
        skeleton.

        Filament length is defined here as the number of member cells.
    '''

    ncells = data.shape[0]

    # Classify the skeleton to identify endpoints and regular cells
    data = label_skeleton(data, periodic=periodic)

    # Connect cells within a 3x3x3 neighbourhood from endpoint to endpoint
    # and store in the first column 
    _, cat = find_filaments(data, periodic=periodic)
    catalogue = np.zeros([cat.shape[0], 8], order='c')
    catalogue[:,0] = cat[:,0]

    # Return the empty catalogue if no filaments were found
    if catalogue.shape[0] == 0:
        return catalogue

    # Store the filament cell positions
    catalogue[:,2:5] = cat[:,1:]
    
    # Calculate the filament lengths and store in the second column
    group_lengths = np.diff(np.hstack([0,np.where(np.diff(cat[:,0])!=0)[0]+1,
                                       len(cat[:,0])]))
    catalogue[:,1] = np.repeat(group_lengths, group_lengths)

    # Calculate and store the filament cell directions
    catalogue[:,5:8] = _get_direction(cat[:,0], cat[:,1:], ncells)

    # Sort the catalogue by filament length in descending order
    if sort:
        sort_idx = np.lexsort([catalogue[:,0],catalogue[:,1]])[::-1]
        catalogue = catalogue[sort_idx]
        group_lengths = np.sort(group_lengths)[::-1]
        catalogue[:,0] = np.repeat(np.arange(np.max(catalogue[:,0]))+1, 
                                   group_lengths)

    return catalogue

def _get_direction(index, pos, box_size):
    '''
        Calculates the direction of a filament cell based on the location
        of that cells' neighbours. The direction vector is normalised.
    '''

    assert pos.ndim == 2
    assert pos.shape[1] == 3

    dxyz = np.zeros(pos.shape)
    
    # Find the beginning and end of filaments in the catalogue
    idx_diff = 1 - np.diff(index) 

    # Calculate position vector between 2 neighbours and account for periodic
    dxyz_tmp = np.mod((pos[:-1]-pos[1:])+1, box_size) - 1

    # Add direction to appropriate indices
    dxyz_tmp = dxyz_tmp * idx_diff[:,None] # set to 0 between filaments
    dxyz[:-1] += dxyz_tmp 
    dxyz[1:] += dxyz_tmp

    # Noramlise direction vector
    r = np.sqrt(np.sum(dxyz**2, axis=1))
    return dxyz/r[:,None]