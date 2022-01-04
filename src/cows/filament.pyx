import numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _count_neighbours(int[:, :, :] data, int i, int j, int k):
    '''
        Count the neighbours in a 3x3x3 cube around a cell.
    '''
    cdef Py_ssize_t di, dj, dk
    cdef int counter

    counter = -1    # -1 because we count the centre
    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                if data[k+dk, j+dj, i+di] != 0:
                    counter += 1
    return counter

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _get_neighbours(int[:, :, :] data, int i, int j, int k, int neighbours[]):
    '''
        Find the neighbours in a 3x3x3 cube around a cell.
    '''
    cdef Py_ssize_t di, dj, dk
    cdef int counter
    counter = 0

    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                neighbours[counter] = data[k+dk, j+dj, i+di]
                counter += 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _classify_skeleton(int[:, :, :] data, int[:, :, :] result):
    '''
        Loop through the data and assign a value equal to the number
        of neighbours each cell has for that cell.
        Cells with only 1 neighbour are endpoints.
        Cells with 2 neighbours are regular cells.
        Cells with 3 neighbours are T-junctions.
        Cells with 4 neighbours are X-junctions.
        Cells with no neighbours are set to background.
        Cells with more than 4 neighbours are not classified and
        are set to the background. This can happen if there are
        cavities in the data.
    '''
    cdef Py_ssize_t i_max, j_max, k_max
    cdef Py_ssize_t i, j, k, ii

    i_max = data.shape[2]
    j_max = data.shape[1]
    k_max = data.shape[0]

    # Loop from 1 to end-1 because data is padded
    for k in range(1, k_max-1):
        for j in range(1, j_max-1):
            for i in range(1, i_max-1):
                if data[k, j, i] == 0:
                    continue

                n_neighbours = _count_neighbours(data, i, j, k)
                if n_neighbours < 5 and n_neighbours > 0:
                    # -1 on indices because of padding
                    result[k-1, j-1, i-1] = n_neighbours

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def classify_skeleton(data):
    '''
        Loop through the data and assign a value equal to the number
        of neighbours each cell has for that cell.
    '''
    assert data.ndim == 3

    # Define output array
    result = np.zeros(data.shape, dtype=np.int32)
    cdef int[:, :, :] result_view = result

    # Pad data to deal with periodic boundaries
    data = np.pad(data, 1, mode='wrap').astype(np.int32)

    _classify_skeleton(data, result)
    return result

@cython.cdivision(True)     # Enable C modulo
cdef int modulo_int(int a, int b):
    return b+(a%b) if a<0 else a%b

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _check_has_neighbour(int[:,:,::1] data, int[:,:,::1] visit_map, 
                              int* i, int* j, int* k, int ncells):
    '''
        Check if cell with index (i,j,k) has a neighbour that has not been 
        visited before given a visitation map. Returns 1 if it has aneighbour 
        and 0 otherwise. Also sets the input index pointers to the index of 
        the neighbour. 
        Note: This algorithm is lazy an will return the first neighour that
        meets the conditons. Therefore it is assumed that the input cell has
        at most one unvisited neighbour to return a complete set.
    '''

    cdef int di, dj, dk
    cdef int i_tmp, j_tmp, k_tmp

    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                i_tmp = modulo_int(i[0] + di, ncells)
                j_tmp = modulo_int(j[0] + dj, ncells)
                k_tmp = modulo_int(k[0] + dk, ncells)
                if (data[k_tmp, j_tmp, i_tmp] > 0 and 
                    visit_map[k_tmp, j_tmp, i_tmp] == 0):
                    i[0] = i_tmp    # set the value of the pointers
                    j[0] = j_tmp    # to the neighbour indices
                    k[0] = k_tmp
                    return 1
    return 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _connect_neighbours(int[:, :, ::1] data, int[:, :, ::1] visit_map, 
                         int[:, :, ::1] result, int[:, ::1] cat):
    '''
        Find connected cells and mark each set with a unique ID starting
        at 1 and going to total sets of connected cells (filaments). Also
        stores the ID and index of each cell in an empty input catalogue.
        Note: This method assumes that each filament consists of two
        endpoints with any number of regular points connecting them. An
        endpoint has exactly one neighbour and a regular point exactly two
        neighbours in a 3x3x3 neighbourhood. Endpoints need to be identified
        by a value of 1 and regular point by any other positive integer.
    '''
    cdef int i, j, k
    cdef int idx_i, idx_j, idx_k
    cdef int ncells
    cdef int n_filaments
    cdef int has_neighbour
    cdef int count
    cdef int zero_count, one_count, two_count

    ncells = data.shape[0]

    count = 0
    n_filaments = 0
    for k in range(ncells):
        for j in range(ncells):
            for i in range(ncells):
                if (data[k, j, i] != 1 or 
                    visit_map[k, j, i] == 1):
                    continue

                idx_i = i
                idx_j = j
                idx_k = k
                n_filaments += 1
                has_neighbour = 1
                while has_neighbour == 1:
                    cat[count, 0] = n_filaments
                    cat[count, 1] = idx_i
                    cat[count, 2] = idx_j
                    cat[count, 3] = idx_k

                    result[idx_k ,idx_j, idx_i] = n_filaments
                    visit_map[idx_k, idx_j, idx_i] = 1

                    has_neighbour = _check_has_neighbour(data, visit_map, 
                                                         &idx_i, &idx_j, 
                                                         &idx_k, ncells)
                    count += 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def connect_neighbour(data):

    assert data.ndim == 3
    assert data.shape[0] == data.shape[1]
    assert data.shape[0] == data.shape[2]

    # Make sure input data is the correct type and ordering
    data = np.array(data, dtype=np.int32, order='c')

    # Define output array, visitation map and catalogue
    result = np.zeros(data.shape, dtype=np.int32, order='c')
    visit_map = np.zeros(data.shape, dtype=np.int32, order='c')
    cat = np.zeros([np.sum(data!=0), 4], dtype=np.int32, order='c')
    
    cdef int[:, :, ::1] data_view = data
    cdef int[:, :, ::1] result_view = result
    cdef int[:, :, ::1] visit_map_view = visit_map
    cdef int[:, ::1] cat_view = cat

    _connect_neighbours(data_view, visit_map_view, result_view, cat_view)
    if np.min(cat[:,0]) == 0:
        cat = cat[:np.argmin(cat[:,0])]

    return result, cat

def get_direction(index, pos, box_size, norm=True):

    assert pos.ndim == 2
    assert pos.shape[1] == 3

    idx_dif = 1 - np.diff(index) 

    dxyz = np.zeros(pos.shape)
    dxyz_tmp = np.mod((pos[:-1]-pos[1:])+1, box_size) - 1
    dxyz_tmp = dxyz_tmp * idx_dif[:,None]
    dxyz[:-1] += dxyz_tmp 
    dxyz[1:] += dxyz_tmp

    if norm:
        r = np.sqrt(np.sum(dxyz**2,axis=1))
        return dxyz/r[:,None]
    return dxyz

def gen_catalogue(data, sort=True):

    ncells = data.shape[0]

    _, cat = connect_neighbour(data)
    catalogue = np.zeros([cat.shape[0], 8], order='c')
    catalogue[:,0] = cat[:,0]
    group_lengths = np.diff(np.hstack([0,np.where(np.diff(cat[:,0])!=0)[0]+1,
                                       len(cat[:,0])]))
    catalogue[:,1] = np.repeat(group_lengths, group_lengths)
    catalogue[:,2:5] = cat[:,1:]

    catalogue[:,5:8] = get_direction(cat[:,0], cat[:,1:], ncells)

    if sort:
        catalogue = catalogue[np.lexsort([catalogue[:,0],catalogue[:,1]])[::-1]]
        group_lengths = np.sort(group_lengths)[::-1]
        catalogue[:,0] = np.repeat(np.arange(np.max(catalogue[:,0]))+1, group_lengths)

    return catalogue

