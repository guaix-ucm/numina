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

PyDoc_STRVAR(combine__doc__, "Module doc");

PyDoc_STRVAR(test__doc__, "function doc");

PyDoc_STRVAR
(
combine_fun__doc__,
"_combine function doc\n\
method, result, variance, number, inputs, shapes, masks, noffsets\n\
End of docs.");

static PyObject *CombineError;

static PyObject* py_test(PyObject *self, PyObject *args)
{
  int i = 0;
  int ok;
  long iterations = 0;
  ok = PyArg_ParseTuple(args, "i", &i);
  if (!ok)
    return NULL;

  if (i < 0)
  {
    PyErr_SetString(CombineError, "invalid i parameter");
    return NULL;
  }

  return PyInt_FromLong((long) iterations);
}

static PyObject* py_combine(PyObject *self, PyObject *args)
{
  int ok;
  int i;
  PyObject *method, *result = NULL, *variance, *number;

  PyObject *inputs, *shapes, *masks, *noffsets;
  int ninputs = 0;

  PyObject *resultarr;
  Py_ssize_t ii;
  ok = PyArg_ParseTuple(args, "OOOOOOOO", &method, &result, &variance, &number,
      &inputs, &shapes, &masks, &noffsets);
  if (!ok)
    return NULL;

  ninputs = PySequence_Length(inputs);

  if (!PyList_Check(inputs))
  {
    PyErr_SetString(CombineError, "combine: inputs is not a sequence");
    return NULL;
  }

  /* check the rest of sequences...
   * we're sure that everything is all right
   */
  PyArrayObject **arr;
  arr = malloc(ninputs * sizeof(PyArrayObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *a = PySequence_GetItem(inputs, i);
    if (!a)
      return NULL;
    arr[i] = PyArray_FROM_OTF(a, NPY_NOTYPE, NPY_IN_ARRAY);
    if (!arr[i])
      return NULL;
    Py_DECREF(a);
  }

  // Using the arrays here...
  printf("We have %d images\n", ninputs);

  resultarr = PyArray_FROM_OTF(result, NPY_NOTYPE, NPY_INOUT_ARRAY);

  if (resultarr == NULL)
  {
    PyArray_XDECREF_ERR(resultarr);
    return NULL;
  }

  double** data;
  data = malloc(ninputs * sizeof(double*));

  npy_intp* dims = PyArray_DIMS(resultarr);

  int iindex, jindex;
  for (iindex = 0; iindex < dims[0]; ++iindex)
    for (jindex = 0; jindex < dims[1]; ++jindex)
    {
      /* Collect the values */
      for (i = 0; i < ninputs; ++i)
      {
        data[i] = (double*) PyArray_GETPTR2(arr[i], iindex, jindex);
      }

      /* Operate with them */
      double sum = 0.0;
      for (i = 0; i < ninputs; ++i)
      {
        sum += *(data[i]);
      }
      double result = sum / ninputs;

      /* Store in result */
      double* p = (double*) PyArray_GETPTR2(resultarr, iindex, jindex);
      *p = result;
    }

  free(data);

  // Cleaning up
  for (i = 0; i < ninputs; i++)
  {
    Py_DECREF(arr[i]);
  }

  free(arr);

  long sum = 0;
  Py_ssize_t len = PyList_Size(inputs);
  for (i = 0; i < len; ++i)
  {
    PyObject* val = PyList_GetItem(inputs, i);
    sum += PyInt_AS_LONG(val);
  }

  return Py_BuildValue("(i,i,i)", sum, 0, len);
}

static PyMethodDef combine_methods[] = { { "test", py_test, METH_VARARGS,
    test__doc__ },
    { "_combine", py_combine, METH_VARARGS, combine_fun__doc__ }, { NULL, NULL,
        0, NULL } /* sentinel */
};

PyMODINIT_FUNC init_combine(void)
{
  PyObject *m;
  m = Py_InitModule3("_combine", combine_methods, combine__doc__);
  import_array();

  if (m == NULL)
    return;

  if (CombineError == NULL)
  {
    /*
     * A different base class can be used as base of the exception
     * passing something instead of NULL
     */
    CombineError = PyErr_NewException("_combine.error", NULL, NULL);
  }
  Py_INCREF(CombineError);
  PyModule_AddObject(m, "error", CombineError);
}

