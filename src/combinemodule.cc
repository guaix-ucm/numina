/*
 * Copyright 2008 Sergio Pascual
 *
 * This file is part of PyEmir
 *
 * PyEmir is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PyEmir is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* $Id$ */

#include <Python.h>
#include <numpy/arrayobject.h>
#include "methods.h"

#include <vector>

PyDoc_STRVAR(combine__doc__, "Module doc");
PyDoc_STRVAR(test1__doc__, "Combines identically shaped images");
PyDoc_STRVAR(method1__doc__, "method_mean doc");
PyDoc_STRVAR(method2__doc__, "method_median doc");


static PyObject *CombineError;

static PyObject* py_method1(PyObject *self, PyObject *args)
{
  int ok;
  PyObject *pydata = NULL;
  PyObject *item = NULL;
  ok = PyArg_ParseTuple(args, "O!:method_mean", &PyList_Type, &pydata);
  if (!ok)
    return NULL;

  /* data is forced to be a list */
  int ndata = PyList_GET_SIZE(pydata);
  if (ndata == 0)
    return Py_BuildValue("(d,d,i)", 0., 0., 0);

  /* Computing when n >= 2 */
  std::vector<double> data(ndata);
  int i;
  for (i = 0; i < ndata; ++i)
  {
    item = PyList_GetItem(pydata, i);
    /*  if (!PyFloat_Check(item))
     {
     PyErr_SetString(PyExc_TypeError, "expected sequence of floats");
     return NULL;
     }*/
    data[i] = PyFloat_AsDouble(item);
  }

  double val, var;
  long number;
  void * params = NULL;
  method_mean(&data[0], ndata, &val, &var, &number, params);
  return Py_BuildValue("(d,d,i)", val, var, number);
}

namespace {

PyObject* py_method2(PyObject *self, PyObject *args)
{
  int ok;
  PyObject *pydata = NULL;
  PyObject *item = NULL;
  ok = PyArg_ParseTuple(args, "O!:method_median", &PyList_Type, &pydata);
  if (!ok)
    return NULL;

  /* data is forced to be a list */
  int ndata = PyList_GET_SIZE(pydata);
  if (ndata == 0)
    return Py_BuildValue("(d,d,i)", 0., 0., 0);

  /* Computing when n >= 2 */
  std::vector<double> data(ndata);
  int i;
  for (i = 0; i < ndata; ++i)
  {
    item = PyList_GetItem(pydata, i);
    /*  if (!PyFloat_Check(item))
     {
     PyErr_SetString(PyExc_TypeError, "expected sequence of floats");
     return NULL;
     }*/
    data[i] = PyFloat_AsDouble(item);
  }

  double val, var;
  long number;
  void * params = NULL;
  method_median(&data[0], ndata, &val, &var, &number, params);
  return Py_BuildValue("(d,d,i)", val, var, number);
}

PyObject* py_internal_combine(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *method = NULL;
  PyObject *images = NULL;
  PyObject *masks = NULL;
  PyObject *res = Py_None;
  PyObject *var = Py_None;
  PyObject *num = Py_None;
  PyObject *resarr = NULL;
  PyObject *vararr = NULL;
  PyObject *numarr = NULL;


  static char *kwlist[] = { "method", "nimages", "nmasks",
		  "result", "variance", "numbers", NULL };

  int ok = PyArg_ParseTupleAndKeywords(args, keywds, "OO!O!|OOO:test1", kwlist, &method,
      &PyList_Type, &images, &PyList_Type, &masks, &res, &var, &num);
  if (!ok)
    return NULL;

  /* images are forced to be a list */
  const int nimages = PyList_GET_SIZE(images);

  /* getting the contents */
  std::vector<PyObject*> iarr(nimages);

  for (int i = 0; i < nimages; i++)
  {
    PyObject *item = PyList_GetItem(images, i);
    /* To be sure is double */
    iarr[i] = PyArray_FROM_OT(item, NPY_DOUBLE);
    /* We don't need item anymore */
    Py_DECREF(item);
  }

  npy_intp* refdims = PyArray_DIMS(iarr[0]);

  /* getting the contents */
  std::vector<PyObject*> marr(nimages);

  for (int i = 0; i < nimages; i++)
  {
    PyObject *item = PyList_GetItem(masks, i);
    /* To be sure is bool */
    marr[i] = PyArray_FROM_OT(item, NPY_BOOL);
    /* We don't need item anymore */
    Py_DECREF(item);
  }

  /*
   * This is ok if we are passing the data to a C function
   * but, as we are creating here a PyList, perhaps it's better
   * to build the PyList with PyObjects and make the conversion to doubles
   * inside the final function only
   */
  std::vector<npy_double> data(nimages);
  npy_intp* dims = PyArray_DIMS(iarr[0]);

  for (npy_intp ii = 0; ii < dims[0]; ++ii)
    for (npy_intp jj = 0; jj < dims[1]; ++jj)
    {
      int used = 0;
      /* Collect the valid values */
      for (int i = 0; i < nimages; ++i)
      {
        npy_bool *pmask = (npy_bool*) PyArray_GETPTR2(marr[i], ii, jj);
        if (*pmask == NPY_TRUE) // <- True values are skipped
          break;

        npy_double *pdata = static_cast<double*>(PyArray_GETPTR2(iarr[i], ii, jj));
        data[i] = *pdata;
        ++used;
      }
      /* Create a PyList with the values */
      PyObject* pydata = PyList_New(used);

      /* Fill it */
      for (int i = 0; i < used; ++i)
      {
        PyObject* value = PyFloat_FromDouble(data[i]);
        PyList_SET_ITEM(pydata, i, value);
      }

      // Calling the function with the pylist
      PyObject* argl = Py_BuildValue("(O)", pydata);
      Py_DECREF(pydata);
      PyObject* result = NULL;
      result = PyEval_CallObject(method, argl);
      Py_DECREF(argl);

      if (!result)
      {
        /* Clean up */
        /* throw exception */
        return NULL;
      }

      void *r = PyArray_GETPTR2(resarr, ii, jj);
      void *v = PyArray_GETPTR2(vararr, ii, jj);
      void *n = PyArray_GETPTR2(numarr, ii, jj);

      /* store the values in the final arrays */
      PyArray_SETITEM(resarr, r, PyTuple_GetItem(result, 0));
      PyArray_SETITEM(vararr, v, PyTuple_GetItem(result, 1));
      PyArray_SETITEM(numarr, n, PyTuple_GetItem(result, 2));
      Py_DECREF(result);
    }

  return Py_None;
}

} // namesppace


static PyMethodDef combine_methods[] = {
    {"_internal_combine", (PyCFunction) py_internal_combine, METH_VARARGS | METH_KEYWORDS, test1__doc__ },
    {"method_mean", py_method1, METH_VARARGS, method1__doc__ },
    {"method_median", py_method2, METH_VARARGS, method2__doc__ },
    {NULL, NULL, 0, NULL} /* sentinel */
};

PyMODINIT_FUNC init_combine(void)
{
  PyObject *m;
  m = Py_InitModule3("ccombine", combine_methods, combine__doc__);
  import_array();

  if (m == NULL)
    return;

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
}

