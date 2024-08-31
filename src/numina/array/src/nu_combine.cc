/*
 * Copyright 2008-2024 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL combine_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <vector>
#include <algorithm>
#include <memory>
// Define and/not in WIN32
#include <ciso646>

#include "nu_combine_defs.h"

// Convenience function to avoid the Py_DECREF macro
static inline void My_Py_Decref(PyObject* obj)
{
  Py_DECREF(obj);
}

static inline void My_PyArray_Iter_Decref(PyArrayIterObject* it)
{
  Py_DECREF(it);
}

// Convenience function to avoid the PyArray_ITER_NEXT macro
static inline void My_PyArray_Iter_Next(PyArrayIterObject* it)
{
  PyArray_ITER_NEXT(it);
}

// Convenience PyArrayIterObject* creator
static inline PyArrayIterObject* My_PyArray_IterNew(PyObject* obj)
{
  return (PyArrayIterObject*) PyArray_IterNew(obj);
}

typedef std::vector<PyArrayIterObject*> VectorPyArrayIter;



// Checking for images
bool NU_combine_image_check(PyObject* exception, PyObject* image,
    PyObject* ref, PyObject* typeref, const char* name, size_t index) {
    if (not PyArray_Check(image)) {
      PyErr_Format(exception,
              "item %zd in %s list is not a ndarray or subclass", index, name);
      return false;
    }

    int image_ndim = PyArray_NDIM((PyArrayObject*)image);

    if (PyArray_NDIM((PyArrayObject*)ref) != image_ndim) {
      PyErr_Format(exception,
          "item %zd in %s list has inconsistent number of axes", index, name);
      return false;
    }

    for(int i = 0; i < image_ndim; ++i) {
      int image_dim_i = PyArray_DIM((PyArrayObject*)image, i);
      if (PyArray_DIM((PyArrayObject*)ref, i) != image_dim_i) {
        PyErr_Format(exception,
            "item %zd in %s list has inconsistent dimension (%i) in axis %i", index, name, image_dim_i, i);
        return false;
      }
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes((PyArrayObject*)typeref, (PyArrayObject*)image)) {
      PyErr_Format(exception,
          "item %zd in %s list has inconsistent dtype", index, name);
      return false;
    }
    return true;
  }

int NU_generic_combine(PyObject** images, PyObject** masks, size_t size,
    PyObject* out[NU_COMBINE_OUTDIM],
    CombineFunc function,
    void* vdata,
    const double* zeros,
    const double* scales,
    const double* weights)
{
}
