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

PyDoc_STRVAR(combine_fun__doc__,
"_combine function doc\n\
method, result, variance, number, inputs, shapes, masks, noffsets\n\
End of docs.");

static PyObject *CombineError;

static PyObject* py_test(PyObject *self, PyObject *args)
{
  int ok;

  PyObject *fun = NULL;
  PyObject *input = NULL;
  PyObject *inputarr;
  PyObject *mask = NULL;
  ok = PyArg_ParseTuple(args, "OO!O!:test", &fun, &PyArray_Type, &input,&PyArray_Type, &mask);
  if (!ok)
    return NULL;

  /* Check that fun is callable */
  if(!PyCallable_Check(fun)) {
    PyErr_Format(PyExc_TypeError, "fun is not callable");
    return NULL;
  }

  // To be sure we are using doubles
  inputarr = PyArray_FROM_OT(input, NPY_DOUBLE);
  npy_double* ptr = (npy_double*) PyArray_GETPTR2(inputarr, 0, 0);
  npy_bool* pmask = (npy_bool*) PyArray_GETPTR2(mask, 0, 0);

/*  printf("%p %p\n", ptr, pmask);
  printf("%"NPY_DOUBLE_FMT" %d\n", *ptr, *pmask);

  npy_intp* dims = PyArray_DIMS(inputarr);
*/

  // Calling the function
  PyObject* argl = Py_BuildValue("([ii])",3,4);
  PyObject* result = NULL;

  result = PyEval_CallObject(fun, argl);
  Py_DECREF(argl);

  if(!result) {
    Py_DECREF(inputarr);
    return NULL;
  }

  double valor = PyFloat_AS_DOUBLE(result);
  printf ("%g\n", valor);
  Py_DECREF(result);

  Py_DECREF(inputarr); //??

  Py_INCREF(Py_None);
  return Py_None;
}

void method_mean(double data[], int size, double* c, double* var, int* number)
{
  if (size == 0)
  {
    *c = *var = 0.0;
    *number = 0;
    return;
  }

  if (size == 1)
  {
    *c = data[0];
    *var = 0.0;
    *number = 1;
    return;
  }

  double sum = 0.0;
  double sum2 = 0.0;
  int i;
  for (i = 0; i < size; ++i)
  {
    sum += data[i];
    sum2 += data[i] * data[i];
  }

  *c = sum / size;
  *number = size;
  *var = sum2 / (size - 1) - (sum * sum) / (size * (size - 1));
}

static PyObject* py_combine(PyObject *self, PyObject *args)
{
  int ok;
  int i;
  PyObject *method, *result = NULL, *variance = NULL, *number = NULL;

  PyObject *inputs, *shapes, *masks, *noffsets;
  int ninputs = 0;

  PyObject *resultarr, *variancearr, *numberarr;
  //Py_ssize_t ii;
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

  if (!PyList_Check(masks))
  {
    PyErr_SetString(CombineError, "combine: masks is not a sequence");
    return NULL;
  }

  /* checked the rest of sequences...
   * we're sure that everything is all right
   */
  PyObject **arr;
  arr = malloc(ninputs * sizeof(PyObject*));
  PyObject **msk;
  msk = malloc(ninputs * sizeof(PyObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *a = PySequence_GetItem(inputs, i);
    if (!a)
    {
      free(arr);
      return NULL;
    }
    arr[i] = PyArray_FROM_OTF(a, NPY_NOTYPE, NPY_IN_ARRAY);

    if (!arr[i])
    {
      free(arr);
      return NULL;
    }

    Py_DECREF(a);

    PyObject *b = PySequence_GetItem(masks, i);
    if (!b)
    {
      free(msk);
      return NULL;
    }
    msk[i] = PyArray_FROM_OT(b, PyArray_BOOL);//NPY_NOTYPE, NPY_IN_ARRAY);

    if (!msk[i])
    {
      free(msk);
      return NULL;
    }

    Py_DECREF(b);

    int pat = PyArray_TYPE(arr[i]);
    printf("PyArray_TYPE %d %d %d %d\n", pat, NPY_DOUBLE, NPY_FLOAT, NPY_INT);
    pat = PyArray_TYPE(msk[i]);
    printf("PyArray_TYPE %d %d %d %d\n", pat, NPY_DOUBLE, NPY_FLOAT, NPY_INT);

  }

  resultarr = PyArray_FROM_OTF(result, NPY_NOTYPE, NPY_INOUT_ARRAY);

  if (resultarr == NULL)
  {
    PyArray_XDECREF_ERR(resultarr);
    free(arr);
    return NULL;
  }

  variancearr = PyArray_FROM_OTF(variance, NPY_NOTYPE, NPY_INOUT_ARRAY);

  if (variancearr == NULL)
  {
    PyArray_XDECREF_ERR(variancearr);
    free(arr);
    return NULL;
  }

  numberarr = PyArray_FROM_OTF(number, NPY_NOTYPE, NPY_INOUT_ARRAY);

  if (numberarr == NULL)
  {
    PyArray_XDECREF_ERR(numberarr);
    free(arr);
    return NULL;
  }

  /* TODO: check that the dimensions nad sizes of the images are equal */
  npy_intp* dims = PyArray_DIMS(resultarr);

  double* data;
  data = malloc(ninputs * sizeof(double));

  /* Assuming 2D arrays */
  int iindex, jindex;
  for (iindex = 0; iindex < dims[0]; ++iindex)
    for (jindex = 0; jindex < dims[1]; ++jindex)
    {
      int used = 0;
      /* Collect the valid values */
      for (i = 0; i < ninputs; ++i)
      {
        void* pp = PyArray_GETPTR2(msk[i], iindex, jindex);
        int mask = 0;
        if (mask)
          continue;

        data[i] = *((double*) PyArray_GETPTR2(arr[i], iindex, jindex));
        ++used;
      }

      double* p = (double*) PyArray_GETPTR2(resultarr, iindex, jindex);
      double* v = (double*) PyArray_GETPTR2(variancearr, iindex, jindex);
      int* n = (int*) PyArray_GETPTR2(numberarr, iindex, jindex);

      /* Compute the results*/
      method_mean(data, used, p, v, n);
    }

  free(data);

  // Cleaning up
  for (i = 0; i < ninputs; i++)
  {
    Py_DECREF(arr[i]);
    Py_DECREF(msk[i]);
  }

  free(arr);
  free(msk);

  return Py_BuildValue("(O,O,O)", resultarr, variancearr, numberarr);
}

static PyMethodDef combine_methods[] = {
    { "test", py_test, METH_VARARGS, test__doc__ },
    { "_combine", py_combine, METH_VARARGS, combine_fun__doc__ },
    { NULL, NULL, 0, NULL } /* sentinel */
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

