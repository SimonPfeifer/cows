import numpy as np
cimport cython

def _label_skeleton(data, periodic=False):
    ''' Label the skeleton. 

        Python wrapper for the cdef function ``label_skeleton``.

        Label all skeleton cells with their respective number of neighbour.
        Also removes cells with zero or more than four neighbours by setting
        them to the background value of zero.
    '''

    if periodic:
        # Pad/wrap data to deal with periodic boundaries
        data = np.pad(data, 1, mode='wrap')
    else:
        # Pad with zeros to make boundary conditions easier to handle
        data = np.pad(data, 1, mode='constant')
    data = np.array(data, dtype=np.int32, order='c')
    
    # Define output array
    result = np.zeros(data.shape, dtype=np.int32, order='c')

    label_skeleton(data, result) # calls the cdef function
    result = np.asarray(result)

    # Remove the padding
    rs = result.shape
    result = result[1:rs[0]-1, 1:rs[1]-1, 1:rs[2]-1]

    return result

def _find_filaments(data, periodic=False):
    ''' Find individual filament.

        Python wrapper for the cdef function ``connect_neighbours``.

        Connects all cells that are neighbours within a 3x3x3 neihbourhood.
        The set of connected cells are labled with a unique ID.
    '''

    # Make sure input data is the correct type and ordering
    data = np.array(data, dtype=np.int32, order='c')

    # Define output array, visitation map and catalogue
    result = np.zeros(data.shape, dtype=np.int32, order='c')
    visit_map = np.zeros(data.shape, dtype=np.int32, order='c')
    catalogue = np.zeros([np.sum(data!=0), 4], dtype=np.int32, order='c')
    
    # Define the C memory view of the variables
    cdef int[:, :, ::1] data_view = data
    cdef int[:, :, ::1] result_view = result
    cdef int[:, :, ::1] visit_map_view = visit_map
    cdef int[:, ::1] cat_view = catalogue

    connect_neighbours(data_view, visit_map_view, result_view, cat_view, 
                       np.int32(periodic))
    return result, catalogue


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int count_neighbours(int[:,:,:] data, int i, int j, int k):
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
cdef label_skeleton(int[:,:,:] data, int[:,:,:] result):
    '''
        Loop through the data and assign a value equal to the number
        of neighbours each cell has for that cell.
        Cells with no neighbours are set to background.
        Cells with only 1 neighbour are endpoints.
        Cells with 2 neighbours are regular cells.
        Cells with 3 neighbours are T-junctions.
        Cells with 4 neighbours are X-junctions.
        Cells with more than 4 neighbours are most likely artefact from
        cavities in the input data.
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

                n_neighbours = count_neighbours(data, i, j, k)
                if n_neighbours > 0:
                    result[k, j, i] = n_neighbours


@cython.cdivision(True)     # Enable C modulo
cdef int modulo_int(int a, int b):
    return b+(a%b) if a<0 else a%b

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int check_has_neighbour(int[:,:,::1] data, int[:,:,::1] visit_map,
                             int* i, int* j, int* k, int ncells):
    '''
        Check if cell with index (i,j,k) has a neighbour that has not been 
        visited before given a visitation map. Returns 1 if it has a neighbour
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
                i_tmp = i[0] + di
                j_tmp = j[0] + dj
                k_tmp = k[0] + dk
                if (i_tmp>=0 and i_tmp<ncells and
                    j_tmp>=0 and j_tmp<ncells and
                    k_tmp>=0 and k_tmp<ncells):
                    if (data[k_tmp, j_tmp, i_tmp] > 0 and 
                        visit_map[k_tmp, j_tmp, i_tmp] == 0):
                        i[0] = i_tmp    # set the value of the pointers
                        j[0] = j_tmp    # to the neighbour indices
                        k[0] = k_tmp
                        return 1
    return 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int check_has_neighbour_wrap(int[:,:,::1] data, int[:,:,::1] visit_map,
                                  int* i, int* j, int* k, int ncells):
    '''
        Check if cell with index (i,j,k) has a neighbour that has not been 
        visited before given a visitation map. Returns 1 if it has a neighbour
        and 0 otherwise. Also sets the input index pointers to the index of 
        the neighbour. Uses periodic wrapping.
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
cdef connect_neighbours(int[:,:,::1] data, int[:,:,::1] visit_map, 
                        int[:,:,::1] result, int[:, ::1] cat,
                        int periodic):
    '''
        Find connected cells and mark each set with a unique ID starting
        at 1 and going to total sets of connected cells. 

        Requires a labeled skeleton as input. Finds an endpoint and loops 
        through neighbouring cells until all connected cells have been 
        visited. Stores the ID and index of each cell in an empty input
        catalogue.
        
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

                    if periodic == 0:
                        has_neighbour = check_has_neighbour(data, visit_map,
                                                            &idx_i, &idx_j, 
                                                            &idx_k, ncells)
                    else:
                        has_neighbour = check_has_neighbour_wrap(data, 
                                                                 visit_map,
                                                                 &idx_i, 
                                                                 &idx_j, 
                                                                 &idx_k,
                                                                 ncells)

                    count += 1