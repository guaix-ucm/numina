
from libc.stdlib cimport malloc, free
from cpython.cobject cimport PyCObject_FromVoidPtrAndDesc

cdef int _kernel_function(double* buffer, int filter_size,
                          double* return_value, void* cb):
    cdef double* th_data = <double*> cb
    cdef int nmed = filter_size / 2
    cdef int i = 0

    if buffer[nmed] < th_data[0]:
        return_value[0] = 0.0
        return 1

    for i in range(nmed, filter_size):
        if buffer[i] < buffer[i+1]:
            return_value[0] = 0.0
            return 1

    for i in range(0, nmed):
        if buffer[i] > buffer[i+1]:
            return_value[0] = 0.0
            return 1

    return_value[0] = 1.0
    return 1


cdef void _kernel_destructor(void* cobject, void *kernel_data):
    free(kernel_data)

def kernel_peak_function(double threshold=0.0):

    cdef double* th_data = <double*>malloc(sizeof(double))
    th_data[0] = threshold

    return PyCObject_FromVoidPtrAndDesc(&_kernel_function,
                                        th_data,
                                        &_kernel_destructor)