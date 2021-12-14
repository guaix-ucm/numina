/*
 * Copyright 2008-2018 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * Numina is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Numina is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Numina.  If not, see <http://www.gnu.org/licenses/>.
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
  if (PyArray_NDIM(array) != 1)
  {
    PyErr_Format(CombineError, "%s dimension %i != 1", name, PyArray_NDIM(array));
    return 0;
  }
  if (PyArray_SIZE(array) != (npy_intp)nimages)
  {
    PyErr_Format(CombineError, "%s size %zd != number of images", name, PyArray_SIZE(array));
    return 0;
  }
  return 1;
}

static PyObject* py_generic_combine(PyObject *self, PyObject *args)
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
    zeros_arr = PyArray_FROM_OTF(zeros, NPY_DOUBLE, NPY_IN_ARRAY);
    if (!check_1d_array(zeros, nimages, "zeros"))
      goto exit;

    zbuffer = (double*)PyArray_DATA(zeros_arr);
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
    scales_arr = PyArray_FROM_OTF(scales, NPY_DOUBLE, NPY_IN_ARRAY);
    if (!check_1d_array(scales_arr, nimages, "scales"))
      goto exit;

    sbuffer = (double*)PyArray_DATA(scales_arr);
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
    weights_arr = PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_IN_ARRAY);
    if (!check_1d_array(weights, nimages, "weights"))
      goto exit;

    wbuffer = (double*)PyArray_DATA(weights_arr);
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

void NU_destructor(PyObject *cap) {
  void* cdata;
  cdata = PyCapsule_GetContext(cap);
  PyMem_Free(cdata);
}

static PyObject *
py_method_mean(PyObject *obj, PyObject *args) {
  
  return PyCapsule_New((void*)NU_mean_function, "numina.cmethod", NULL);
}

static PyObject *
py_method_median(PyObject *obj, PyObject *args) {
  
  return PyCapsule_New((void*)NU_median_function, "numina.cmethod", NULL);
}

static PyObject *
py_method_sum(PyObject *obj, PyObject *args) {

  return PyCapsule_New((void*)NU_sum_function, "numina.cmethod", NULL);
}

static PyObject *
py_method_minmax(PyObject *obj, PyObject *args) {
  int nmin = 0;
  int nmax = 0;
  int* funcdata = NULL;
  PyObject * cap = NULL;

  if (!PyArg_ParseTuple(args, "ii", &nmin, &nmax))
    return NULL;

  if ((nmin < 0) || (nmax < 0)) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, nmin and nmax must be >= 0");
    return NULL;
  }

  cap = PyCapsule_New((void*) NU_minmax_function, "numina.cmethod", NU_destructor);
  if (cap == NULL)
    return NULL;

  funcdata = (int*)PyMem_Malloc(2 * sizeof(int));
  if (funcdata == NULL) {
    Py_DECREF(cap);
    return PyErr_NoMemory();
  }

  funcdata[0] = nmin;
  funcdata[1] = nmax;

  if (PyCapsule_SetContext(cap, funcdata))
  {
    PyMem_Free(funcdata);
    Py_DECREF(cap);
    return NULL;
  }

  return cap;
}

static PyObject *
py_method_sigmaclip(PyObject *obj, PyObject *args) {
  double low = 0.0;
  double high = 0.0;
  double *funcdata = NULL;
  PyObject *cap;

  if (!PyArg_ParseTuple(args, "dd", &low, &high)) 
    return NULL;

  if (low < 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, low < 0");
    return NULL;
  }

  if (high < 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, high < 0");
    return NULL;
  }

  cap = PyCapsule_New((void*)NU_sigmaclip_function, "numina.cmethod", NU_destructor);
  if (cap == NULL)
    return NULL;

  funcdata = (double*)PyMem_Malloc(2 * sizeof(double));
  if (funcdata == NULL) {
      Py_DECREF(cap);
      return PyErr_NoMemory();
  }

  funcdata[0] = low;
  funcdata[1] = high;

  if (PyCapsule_SetContext(cap, funcdata))
  {
    PyMem_Free(funcdata);
    Py_DECREF(cap);
    return NULL;
  }

  return cap;
}

static PyObject *
py_method_quantileclip(PyObject *obj, PyObject *args) {
  double fclip = 0.0;
  double *funcdata = NULL;
  PyObject *cap;

  if (!PyArg_ParseTuple(args, "d", &fclip)) 
    return NULL;

  if (fclip < 0 || fclip > 0.4) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter fclip, must be 0 <= fclip < 0.4");
    return NULL;
  }

  cap = PyCapsule_New((void*) NU_quantileclip_function, "numina.cmethod", NU_destructor);
  if (cap == NULL)
    return NULL;

  funcdata = (double*)PyMem_Malloc(sizeof(double));
  if (funcdata == NULL) {
    Py_DECREF(cap);
    return PyErr_NoMemory();
  }

  *funcdata = fclip;

  if (PyCapsule_SetContext(cap, funcdata))
  {
    PyMem_Free(funcdata);
    Py_DECREF(cap);
    return NULL;
  }

  return cap;
}

static PyMethodDef module_functions[] = {
    {"generic_combine", py_generic_combine, METH_VARARGS, ""},
    {"mean_method", py_method_mean, METH_NOARGS, ""},
    {"median_method", py_method_median, METH_NOARGS, ""},
    {"minmax_method", py_method_minmax, METH_VARARGS, ""},
    {"sigmaclip_method", py_method_sigmaclip, METH_VARARGS, ""},
    {"quantileclip_method", py_method_quantileclip, METH_VARARGS, ""},
    {"sum_method", py_method_sum, METH_NOARGS, ""},
    { NULL, NULL, 0, NULL } /* sentinel */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_combine",     /* m_name */
    combine__doc__,  /* m_doc */
    -1,                  /* m_size */
    module_functions,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit__combine(void)
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
     CombineError = PyErr_NewException("_combine.CombineError", NULL, NULL);
   }
   Py_INCREF(CombineError);
   PyModule_AddObject(m, "CombineError", CombineError);
   return m;
}
