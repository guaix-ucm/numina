
from cpython.version cimport PY_MAJOR_VERSION, PY_VERSION_HEX

from libc.stdlib cimport malloc, free


if PY_MAJOR_VERSION >= 3:
    # PyCapsule exits in Py3 and Py2.7
    # But scipy use PyCObject in Py2.7
    # and PyCapsule in Py3

    from cpython.pycapsule cimport PyCapsule_New
    from cpython.pycapsule cimport PyCapsule_GetContext
    from cpython.pycapsule cimport PyCapsule_SetContext

if PY_MAJOR_VERSION < 3:
    from cpython.cobject cimport PyCObject_FromVoidPtrAndDesc

# Kernel function is the same
cdef int _kernel_function(double* buffer, int filter_size,
                          double* return_value, void* cb):
    cdef double* th_data = <double*> cb
    cdef int nmed = filter_size / 2
    cdef int i = 0

    if buffer[nmed] < th_data[0]:
        return_value[0] = 0.0
        return 1

    for i in range(nmed, filter_size-1):
        if buffer[i] <= buffer[i+1]:
            return_value[0] = 0.0
            return 1

    for i in range(0, nmed):
        if buffer[i] >= buffer[i+1]:
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

if PY_MAJOR_VERSION < 3:
    def kernel_peak_function(double threshold=0.0):

        cdef object result
        cdef double *data

        data = <double*>malloc(sizeof(double))
        if data is NULL:
            raise MemoryError()

        data[0] = threshold

        result = PyCObject_FromVoidPtrAndDesc(&_kernel_function,
                                              data,
                                              &_destructor_cobj)

        return result

if PY_MAJOR_VERSION >= 3:

    def kernel_peak_function(double threshold=0.0):

        cdef object result
        cdef double *data

        data = <double*>malloc(sizeof(double))
        if data is NULL:
            raise MemoryError()

        data[0] = threshold

        result = PyCapsule_New(&_kernel_function,
                               NULL, # if we set a name here, generic_f doesn't work
                               _destructor_cap)

        PyCapsule_SetContext(result, data)

        return result

