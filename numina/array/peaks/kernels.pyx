
#cython: language_level=3

from libc.stdlib cimport malloc, free

from cpython.pycapsule cimport PyCapsule_New
from cpython.pycapsule cimport PyCapsule_GetContext
from cpython.pycapsule cimport PyCapsule_SetContext


# Kernel function is the same
cdef int _kernel_function(double* buffer, int filter_size,
                          double* return_value, void* cb):
    cdef double* data = <double*> cb
    cdef int nmed = filter_size // 2
    cdef int i = 0
    cdef int start = 0
    cdef int mcount = 0
    cdef double th = data[0]
    cdef int limit = <int>data[1]

    if buffer[nmed] < th:
        return_value[0] = 0.0
        return 1

    # Count contiguous equal values to the right
    for i in range(nmed, filter_size-1):
        # print '0-',i, i+1
        # print '0-',buffer[i], buffer[i+1]
        if buffer[i] == buffer[i+1]:
            mcount += 1
            start = i + 1
        else:
            start = i
            break

    for i in range(start, filter_size-1):
        if buffer[i] <= buffer[i+1]:
            return_value[0] = 0.0
            return 1

    # Count contiguous equal values to the left
    for i in range(nmed, 0, -1):
        if buffer[i] == buffer[i-1]:
            mcount += 1
            start = i - 1
        else:
            start = i
            break

    for i in range(0, start):
        if buffer[i] >= buffer[i+1]:
            return_value[0] = 0.0
            return 1

    # Reject peak if it has too much
    # flat values
    if mcount > limit:
        return_value[0] = 0.0
        return 1

    return_value[0] = 1.0
    return 1


cdef void _destructor_cobj(void* cobject, void *kernel_data):
    free(kernel_data)


cdef void _destructor_cap(object cap):
    cdef void *cdata
    cdata = PyCapsule_GetContext(cap)
    free(cdata)


def kernel_peak_function(double threshold=0.0, int fpeak=1):

    cdef object result
    cdef double *data

    data = <double*>malloc(2 * sizeof(double))
    if data is NULL:
        raise MemoryError()

    data[0] = threshold
    # A value of 1 allows a peak with 2 equal pixels
    data[1] = fpeak

    result = PyCapsule_New(&_kernel_function,
                           NULL, # if we set a name here, generic_f doesn't work
                           _destructor_cap)

    PyCapsule_SetContext(result, data)

    return result
