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
#include <numpy/arrayobject.h>

#include "nu_combine_methods.h"
#include "nu_combine.h"

PyDoc_STRVAR(combine__doc__, "Internal combine module, not to be used directly.");

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

// An exception in this module
static PyObject* CombineError;

// Convenience check function
static inline int check_1d_array(PyObject* array, size_t nimages, const char* name) {
  if (PyArray_NDIM((PyArrayObject*)array) != 1)  //  error: cannot convert ‘PyObject*’ {aka ‘_object*’} to ‘const PyArrayObject*’
  {
    PyErr_Format(CombineError, "%s dimension %i != 1", name, PyArray_NDIM((PyArrayObject*)array));
    return 0;
  }
  if (PyArray_SIZE((PyArrayObject*)array) != (npy_intp)nimages) //  cannot convert ‘PyObject*’ {aka ‘_object*’} to ‘const PyArrayObject*’
  {
    PyErr_Format(CombineError, "%s size %zd != number of images", name, PyArray_SIZE((PyArrayObject*)array));
    return 0;
  }
  return 1;
}

static PyObject* py_generic_combine1(PyObject *self, PyObject *args)
{
  /* Arguments */
  PyObject *images = NULL;
  PyObject *masks = NULL;
  // Output has one dimension more than the inputs, of size
  // OUTDIM
  const Py_ssize_t OUTDIM = NU_COMBINE_OUTDIM;
  PyObject *out[NU_COMBINE_OUTDIM] = {NULL, NULL, NULL};
  PyObject* fnc = NULL;

  PyObject* scales = NULL;
  PyObject* zeros = NULL;
  PyObject* weights = NULL;


  PyObject *images_seq = NULL;
  PyObject *masks_seq = NULL;
  PyObject* zeros_arr = NULL;
  PyObject* scales_arr = NULL;
  PyObject* weights_arr = NULL;

  void *func = (void*)NU_mean_function;
  void *data = NULL;

  Py_ssize_t nimages = 0;
  Py_ssize_t nmasks = 0;
  Py_ssize_t ui = 0;

  PyObject** allimages = NULL;
  PyObject** allmasks = NULL;

  double* zbuffer = NULL;
  double* sbuffer = NULL;
  double* wbuffer = NULL;

  int ok = PyArg_ParseTuple(args,
      "OOO!O!O!|OOOO:generic_combine",
      &fnc,
      &images,
      &PyArray_Type, &out[0],
      &PyArray_Type, &out[1],
      &PyArray_Type, &out[2],
      &masks,
      &zeros,
      &scales,
      &weights);

  if (!ok)
  {
    goto exit;
  }

  images_seq = PySequence_Fast(images, "expected a sequence");
  nimages = PySequence_Size(images_seq);

  if (nimages == 0) {
    PyErr_Format(CombineError, "data list is empty");
    goto exit;
  }

  // Converted to an array of pointers
  allimages = PySequence_Fast_ITEMS(images_seq);

  // Checking for images
  for(ui = 0; ui < nimages; ++ui) {
    if (!NU_combine_image_check(CombineError, allimages[ui], allimages[0], allimages[0], "data", ui))
      goto exit;
  }

  // Checking for outputs
  for(ui = 0; ui < OUTDIM; ++ui) {
    if (!NU_combine_image_check(CombineError, out[ui], allimages[0], out[0], "output", ui))
      goto exit;
  }

  if (!PyCapsule_IsValid(fnc, "numina.cmethod")) {
    PyErr_SetString(PyExc_TypeError, "parameter is not a valid capsule");
    goto exit;
  }

  func = PyCapsule_GetPointer(fnc, "numina.cmethod");
  data = PyCapsule_GetContext(fnc);

  // Checking zeros, scales and weights
  if (zeros == Py_None) {
    zbuffer = (double*)PyMem_Malloc(nimages * sizeof(double));
    if (zbuffer == NULL) {
      PyErr_NoMemory();
      goto exit;
    }
    for(ui = 0; ui < nimages; ++ui)
      zbuffer[ui] = 0.0;
  }
  else {
    zeros_arr = PyArray_FROM_OTF(zeros, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!check_1d_array(zeros, nimages, "zeros"))
      goto exit;

    zbuffer = (double*)PyArray_DATA((PyArrayObject*)zeros_arr);
  }

  if (scales == Py_None) {
    sbuffer = (double*)PyMem_Malloc(nimages * sizeof(double));
    if (sbuffer == NULL) {
      PyErr_NoMemory();
      goto exit;
    }
    for(ui = 0; ui < nimages; ++ui)
      sbuffer[ui] = 1.0;
  }
  else {
    scales_arr = PyArray_FROM_OTF(scales, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!check_1d_array(scales_arr, nimages, "scales"))
      goto exit;

    sbuffer = (double*)PyArray_DATA((PyArrayObject*)scales_arr);
  }

  if (weights == Py_None) {
    wbuffer = (double*)PyMem_Malloc(nimages * sizeof(double));
    if (wbuffer == NULL) {
      PyErr_NoMemory();
      goto exit;
    }
    for(ui = 0; ui < nimages; ++ui)
      wbuffer[ui] = 1.0;
  }
  else {
    weights_arr = PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!check_1d_array(weights, nimages, "weights"))
      goto exit;

    wbuffer = (double*)PyArray_DATA((PyArrayObject*)weights_arr);
  }

  if (masks == Py_None) {
    allmasks = NULL;
  }
  else {
    // Checking the masks
    masks_seq = PySequence_Fast(masks, "expected a sequence");
    nmasks = PySequence_Size(masks_seq);

    if (nimages != nmasks) {
      PyErr_Format(CombineError, "number of images (%zd) and masks (%zd) is different", nimages, nmasks);
      goto exit;
    }

    allmasks = PySequence_Fast_ITEMS(masks_seq);

    for(ui = 0; ui < nimages; ++ui) {
      if (!NU_combine_image_check(CombineError, allmasks[ui], allimages[0], allmasks[0], "masks", ui))
        goto exit;
    }
  }

  if(!NU_generic_combine(allimages, allmasks, nimages, out,
      (CombineFunc)func, data, zbuffer, sbuffer, wbuffer)
    )
    goto exit;

exit:
  Py_XDECREF(images_seq);

  if (masks != Py_None)
    Py_XDECREF(masks_seq);

  if (zeros == Py_None)
    PyMem_Free(zbuffer);

  Py_XDECREF(zeros_arr);

  if (scales == Py_None)
    PyMem_Free(sbuffer);

  Py_XDECREF(scales_arr);

  if (weights == Py_None)
    PyMem_Free(wbuffer);

  Py_XDECREF(weights_arr);

  return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyMethodDef module_functions[] = {
    {"generic_combine", py_generic_combine1, METH_VARARGS, ""},
    { NULL, NULL, 0, NULL } /* sentinel */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_combine1",     /* m_name */
    combine__doc__,  /* m_doc */
    -1,                  /* m_size */
    module_functions,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit__combine1(void)
{
   PyObject *m;
   m = PyModule_Create(&moduledef);
   if (m == NULL)
     return NULL;

   import_array();
   if (CombineError == NULL)
   {
    /*
     * A different base class can be used as base of the exception
     * passing something instead of NULL
     */
     CombineError = PyErr_NewException("_combine1.CombineError", NULL, NULL);
   }
   Py_INCREF(CombineError);
   PyModule_AddObject(m, "CombineError", CombineError);
   return m;
}
