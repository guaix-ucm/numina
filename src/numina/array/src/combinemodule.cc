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
#include <vector>
#include <algorithm>

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

static PyObject* py_generic_combine(PyObject *self, PyObject *args)
{
  /* Inputs */
  PyObject *images = NULL;
  PyObject *images_seq = NULL;
  Py_ssize_t nimages = 0;
  /* masks */
  PyObject *masks = NULL;
  PyObject *masks_seq = NULL;
  Py_ssize_t nmasks = 0;
  /* outputs */
  PyObject *out_res = NULL;
  PyObject *out_var = NULL;
  PyObject *out_pix = NULL;
  Py_ssize_t nout = 3;
  Py_ssize_t ui;
  /* scales, zeros and weights */
  PyObject* scales = NULL;
  PyObject* zeros = NULL;
  PyObject* weights = NULL;
  PyObject* zeros_arr = NULL;
  PyObject* scales_arr = NULL;
  PyObject* weights_arr = NULL;

  /* Capsule */
  PyObject* fnc = NULL;
  void *capsule_func = (void*)NU_mean_function;
  void *capsule_data = NULL;

  PyObject** tmp_list = NULL;

  double* zbuffer = NULL;
  double* sbuffer = NULL;
  double* wbuffer = NULL;

  /* Iterator */
  PyArray_Descr* dtype_res = NULL;
  PyArray_Descr* dtype_pix = NULL;
  npy_uint32 iflags;
  std::vector<npy_uint32> op_flags;
  std::vector<PyArray_Descr*> op_dtypes;
  std::vector<PyObject*> ops;
  NpyIter *iter = NULL;
  NpyIter_IterNextFunc *iternext;
  char** dataptr;

  /* NU_generic_combine */
  std::vector<double> data2;
  std::vector<double> wdata;

  int outval;

  outval = PyArg_ParseTuple(args,
      "OOO!O!O!|OOOO:generic_combine",
      &fnc, /* capsule */
      &images, /* sequence of images */
      &PyArray_Type, &out_res, /* result */
      &PyArray_Type, &out_var, /* variance */
      &PyArray_Type, &out_pix, /* npix */
      &masks, /* sequence of masks */
      &zeros, /* sequence of zeros */
      &scales,/* sequence of scales */
      &weights /* sequence of weights */
      );

  if (!outval) return NULL;

  images_seq = PySequence_Fast(images, "expected a sequence");
  nimages = PySequence_Size(images_seq);

  if (nimages == 0) {
    PyErr_Format(CombineError, "data list is empty");
    goto exit;
  }
  if (masks != NULL && masks != Py_None) {
    masks_seq = PySequence_Fast(masks, "expected a sequence");
    nmasks = PySequence_Size(masks_seq);
    if (nmasks != nimages) {
      PyErr_Format(CombineError,
          "number of images (%zd) and masks (%zd) is different", nimages, nmasks);
      goto exit;
    }
  }

  if (!PyCapsule_IsValid(fnc, "numina.cmethod")) {
    PyErr_SetString(PyExc_TypeError, "parameter is not a valid capsule");
    goto exit;
  }

  capsule_func = PyCapsule_GetPointer(fnc, "numina.cmethod");
  capsule_data = PyCapsule_GetContext(fnc);

  // Checking zeros, scales and weights
  if (zeros == Py_None  || zeros == NULL) {
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

  if (scales == Py_None || scales == NULL) {
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

  if (weights == Py_None || weights == NULL) {
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

  dtype_res = PyArray_DescrFromType(NPY_DOUBLE);
  dtype_pix = PyArray_DescrFromType(NPY_INT);
  /* Inputs */
  tmp_list = PySequence_Fast_ITEMS(images_seq);
  std::copy(tmp_list, tmp_list + nimages, std::back_inserter(ops));
  std::fill_n(std::back_inserter(op_flags), nimages, NPY_ITER_READONLY | NPY_ITER_NBO);
  std::fill_n(std::back_inserter(op_dtypes), nimages, dtype_res);
  /* outputs */
  ops.push_back(out_res);
  ops.push_back(out_var);
  ops.push_back(out_pix);
  op_dtypes.push_back(dtype_res);
  op_dtypes.push_back(dtype_res);
  op_dtypes.push_back(dtype_pix);
  std::fill_n(std::back_inserter(op_flags), nout, NPY_ITER_WRITEONLY | NPY_ITER_NBO| NPY_ITER_ALIGNED);
  /* masks */
  if (nmasks != 0) {
    tmp_list = PySequence_Fast_ITEMS(masks_seq);
    std::copy(tmp_list, tmp_list + nmasks, std::back_inserter(ops));
    std::fill_n(std::back_inserter(op_flags), nmasks, NPY_ITER_READONLY | NPY_ITER_NBO);
    std::fill_n(std::back_inserter(op_dtypes), nmasks, dtype_pix);
  }

  iter = NpyIter_MultiNew(nimages + nout + nmasks, (PyArrayObject**)(&ops[0]),
                  NPY_ITER_BUFFERED,
                  NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                  &op_flags[0], &op_dtypes[0]);
  Py_DECREF(dtype_res);
  Py_DECREF(dtype_pix);
  op_dtypes.resize(0); /* clean up */

  if (iter == NULL) {
      goto exit;
  }

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);

  data2.reserve(nimages);
  wdata.reserve(nimages);

  do {
      double* p_val;
      double converted;
      double* pvalues[3];
      int ncalc;

      for(int ii = 0; ii < nimages; ++ii) {
        npy_bool m_val = NPY_FALSE;

        if(wbuffer[ii] < 0)
          continue;

        if(nmasks == nimages) {
           m_val = (* (int*)dataptr[nimages + nout + ii]) > 0;
        }

        if (m_val == NPY_TRUE)
          continue;

        p_val = (double*)dataptr[ii];
        converted = (*p_val - zbuffer[ii]) / (sbuffer[ii]);
        data2.push_back(converted);
        wdata.push_back(wbuffer[ii]);
      }

      // And pass the data to the combine method
      if (data2.size() > 0) {
          outval = ((CombineFunc)capsule_func)(&data2[0], &wdata[0], data2.size(), pvalues, capsule_data);
          if(!outval) {
             PyErr_SetString(PyExc_RuntimeError, "unknown error in combine method");
             NpyIter_Deallocate(iter);
             goto exit;
          }
        ncalc = (*pvalues[2]);
        memcpy(dataptr[nimages], pvalues[0], sizeof(double));
        memcpy(dataptr[nimages + 1], pvalues[1], sizeof(double));
        memcpy(dataptr[nimages + 2], &ncalc, sizeof(int));
    }
    data2.clear();
    wdata.clear();


    } while(iternext(iter));

  NpyIter_Deallocate(iter);

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
