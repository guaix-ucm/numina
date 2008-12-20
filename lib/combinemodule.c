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
PyDoc_STRVAR(test1__doc__, "test1 doc");
PyDoc_STRVAR(test2__doc__, "test2 doc");
PyDoc_STRVAR(method1__doc__, "method1 doc");

void method_mean(double data[], int size, double* c, double* var, long* number)
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

static PyObject *CombineError;

static PyObject* py_method1(PyObject *self, PyObject *args)
{
  int ok;
  PyObject *pydata = NULL;
  PyObject *item = NULL;
  ok = PyArg_ParseTuple(args, "O!:method1", &PyList_Type, &pydata);
  if (!ok)
    return NULL;

  /* data is forced to be a list */
  int ndata = PyList_GET_SIZE(pydata);
  if (ndata == 0)
    return Py_BuildValue("(d,d,i)", 0., 0., 0);
  if (ndata == 1)
    return Py_BuildValue("(d,d,i)", 1.0, 0., 1);

  /* Computing when n >= 2 */
  double* data = malloc(ndata * sizeof(double));
  int i;
  for (i = 0; i < ndata; ++i)
  {
    item = PyList_GetItem(pydata, i);
    /*  if (!PyFloat_Check(item))
     {
     free(data);
     PyErr_SetString(PyExc_TypeError, "expected sequence of floats");
     return NULL;
     }*/
    data[i] = PyFloat_AsDouble(item);
  }

  double val, var;
  long number;
  method_mean(data, ndata, &val, &var, &number);
  free(data);
  return Py_BuildValue("(d,d,i)", val, var, number);
}

static PyObject* py_test1(PyObject *self, PyObject *args, PyObject *keywds)
{
  int i;
  PyObject *fun = NULL;
  PyObject *inputs = NULL;
  PyObject *masks = NULL;
  PyObject *res = Py_None;
  PyObject *var = Py_None;
  PyObject *num = Py_None;
  PyObject *resarr = NULL;
  PyObject *vararr = NULL;
  PyObject *numarr = NULL;

  static char *kwlist[] = { "fun", "inputs", "mask", "res", "var", "num", NULL };

  int ok = PyArg_ParseTupleAndKeywords(args, keywds, "OO!O!|OOO:test1", kwlist, &fun,
      &PyList_Type, &inputs, &PyList_Type, &masks, &res, &var, &num);
  if (!ok)
    return NULL;

  /* Check that fun is callable */
  if (!PyCallable_Check(fun))
  {
    PyErr_Format(PyExc_TypeError, "fun is not callable");
    return NULL;
  }

  /* inputs is forced to be a list */
  int ninputs = PyList_GET_SIZE(inputs);

  /* we don't like empty lists */
  if(ninputs == 0)
  {
    PyErr_Format(PyExc_TypeError, "inputs is empty"); // TODO: check this exception
    return NULL;
  }

  /* number of masks must be equal to the number of inputs */
  if(PyList_GET_SIZE(masks) != ninputs)
  {
    PyErr_Format(PyExc_TypeError, "masks and inputs are not of the same length"); // TODO: check this exception
  }

  /* getting the contents */
  PyObject **iarr = malloc(ninputs * sizeof(PyObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *item = PyList_GetItem(inputs, i);
    if (!item)
    {
      /* Problem here */
      /* Clean up */
      free(iarr);
      PyErr_Format(PyExc_TypeError, "can't get object from inputs list"); // TODO: check this exception
      return NULL;
    }
    /* To be sure is double */
    iarr[i] = PyArray_FROM_OT(item, NPY_DOUBLE);

    /* We don't need item anymore */
    Py_DECREF(item);

    if (!iarr[i])
    {
      /* Can't be converted to array */
      /* Clean up */
      free(iarr);
      PyErr_Format(PyExc_TypeError, "object can't be converted into a numpy float array"); // TODO: check this exception
      return NULL;
    }

    // checking dimensions
    if (((PyArrayObject*) iarr[i])->nd < 2) // Array must be (atleast 2D)
    {
      /* Clean up */
      free(iarr);
      PyErr_Format(PyExc_TypeError, "input array must be (at least) bidimensional"); // TODO: check this exception
      return NULL;
    }

    // checking sizes
    npy_intp* refdims = PyArray_DIMS(iarr[0]);
    npy_intp* thisdims = PyArray_DIMS(iarr[i]);
    if (refdims[0] != thisdims[0] || refdims[1] != thisdims[1])
    {
      /* Clean up */
      free(iarr);
      PyErr_Format(PyExc_TypeError, "arrays must have the same shape"); // TODO: check this exception
      return NULL;
    }

  }

  npy_intp* refdims = PyArray_DIMS(iarr[0]);

  /* getting the contents */
  PyObject **marr = malloc(ninputs * sizeof(PyObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *item = PyList_GetItem(masks, i);
    if (!item)
    {
      /* Problem here */
      /* Clean up */
      free(iarr);
      free(marr);
      PyErr_Format(PyExc_TypeError, "can't get object from masks list"); // TODO: check this exception
      return NULL;
    }

    /* To be sure is bool */
    marr[i] = PyArray_FROM_OT(item, NPY_BOOL);
    /* We don't need item anymore */
    Py_DECREF(item);

    if (!marr[i])
    {
      /* Can't be converted to array */
      /* Clean up */
      free(iarr);
      free(marr);
      PyErr_Format(PyExc_TypeError, "object can't be converted into a numpy bool array"); // TODO: check this excepti
      return NULL;
    }

    // checking dimensions
    if (((PyArrayObject*) marr[i])->nd < 2) // Array must be (atleast 2D)
     {
       /* Clean up */
       free(iarr);
       free(marr);
       PyErr_Format(PyExc_TypeError, "mask array must be (at least) bidimensional"); // TODO: check this exception
       return NULL;
     }

     // checking sizes

     npy_intp* thisdims = PyArray_DIMS(marr[i]);
     if (refdims[0] != thisdims[0] || refdims[1] != thisdims[1])
     {
       /* Clean up */
       free(iarr);
       free(marr);
       PyErr_Format(PyExc_TypeError, "mask arrays must have the same shape"); // TODO: check this exception
       return NULL;
     }

  }

  /* checks */
  /* All the images have equal size and are 2D */

  /* If res is none, create a new image, else
   * check that the size and shape are equal to the rest of images
   * and use it as output
   */
  /* I should check the return values of the functions and
   * return if some fails
   */
  if (res == Py_None)
  {
    resarr = PyArray_SimpleNew(2, refdims, NPY_DOUBLE);
    if(resarr == NULL)
    {
      /* Do something */
      free(iarr);
      free(marr);
      return NULL;
    }
  } else
  {
    resarr = PyArray_FROM_OT(res, NPY_DOUBLE);
  }

  if (var == Py_None)
  {
    vararr = PyArray_SimpleNew(2, refdims, NPY_DOUBLE);
  } else
  {
    vararr = PyArray_FROM_OT(var, NPY_DOUBLE);
  }

  if (num == Py_None)
  {
    numarr = PyArray_SimpleNew(2, refdims, NPY_LONG);
  } else
  {
    numarr = PyArray_FROM_OT(num, NPY_LONG);
  }


  /*
   * This is ok if we are passing the data to a C function
   * but, as we are creating here a PyList, perhaps it's better
   * to build the PyList with PyObjects and make the conversion to doubles
   * inside the final function only
   */
  npy_double* data;
  data = malloc(ninputs * sizeof(npy_double));
  npy_intp* dims = PyArray_DIMS(iarr[0]);

  npy_intp ii, jj;
  for (ii = 0; ii < dims[0]; ++ii)
    for (jj = 0; jj < dims[1]; ++jj)
    {
      int used = 0;
      /* Collect the valid values */
      for (i = 0; i < ninputs; ++i)
      {
        npy_bool *pmask = (npy_bool*) PyArray_GETPTR2(marr[i], ii, jj);
        if (*pmask == NPY_FALSE) // <- This decides how the mask is used
          continue;

        npy_double *pdata = PyArray_GETPTR2(iarr[i], ii, jj);
        data[i] = *pdata;
        ++used;
      }
      /* Create a PyList with the values */
      PyObject* pydata = PyList_New(used);

      /* Fill it */
      for (i = 0; i < used; ++i)
      {
        PyObject* value = PyFloat_FromDouble(data[i]);
        PyList_SET_ITEM(pydata, i, value);
      }

      // Calling the function with the pylist
      PyObject* argl = Py_BuildValue("(O)", pydata);
      Py_DECREF(pydata);
      PyObject* result = NULL;
      result = PyEval_CallObject(fun, argl);
      Py_DECREF(argl);

      if (!result)
      {
        /* Clean up */
        free(data);
        free(iarr);
        free(marr);

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

  free(data);
  free(iarr);
  free(marr);

  // Increasing the reference before returning
  if (res != Py_None)
  {
    Py_INCREF(resarr);
  }
  if (var != Py_None)
  {
      Py_INCREF(vaearr);
  }
  if (num != Py_None)
  {
      Py_INCREF(numarr);
  }
  return Py_BuildValue("(N,N,N)", resarr, vararr, numarr);
}

static PyObject* py_test2(PyObject *self, PyObject *args, PyObject *keywds)
{
  int ok;
  int i;

  PyObject *inputs = NULL;
  PyObject *masks = NULL;
  PyObject *res = Py_None;
  PyObject *resarr = NULL;
  PyObject *vararr = NULL;
  PyObject *numarr = NULL;

  static char *kwlist[] = { "inputs", "mask", "res", NULL };

  ok = PyArg_ParseTupleAndKeywords(args, keywds, "O!O!|O:test2", kwlist,
      &PyList_Type, &inputs, &PyList_Type, &masks, &res);
  if (!ok)
    return NULL;

  int ninputs = PyList_GET_SIZE(inputs);

  /* getting the contents */
  PyObject **iarr = malloc(ninputs * sizeof(PyObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *a = PyList_GetItem(inputs, i);
    if (!a)
    {
      /* Problem here */
      /* Clean up */
      free(iarr);
      return NULL;
    }
    /* To be sure it is double */
    iarr[i] = PyArray_FROM_OT(a, NPY_DOUBLE);

    if (!iarr[i])
    {
      /* Can't be converted to array */
      /* Clean up */
      free(iarr);
      return NULL;
    }

    /* We don't need a anymore */
    Py_DECREF(a);
  }

  /* getting the contents */
  PyObject **marr = malloc(ninputs * sizeof(PyObject*));

  for (i = 0; i < ninputs; i++)
  {
    PyObject *a = PyList_GetItem(masks, i);
    if (!a)
    {
      /* Problem here */
      /* Clean up */
      free(iarr);
      free(marr);
      return NULL;
    }
    /* To be sure is bool */
    marr[i] = PyArray_FROM_OT(a, NPY_BOOL);

    if (!marr[i])
    {
      /* Can't be converted to array */
      /* Clean up */
      free(iarr);
      free(marr);
      return NULL;
    }

    /* We don't need a anymore */
    Py_DECREF(a);
  }

  /* Creating output images */
  npy_intp* dims = PyArray_DIMS(iarr[0]);
  int nd = ((PyArrayObject*) iarr[0])->nd; // Must be 2

  if (res == Py_None)
  {
    int nd = ((PyArrayObject*) iarr[0])->nd; // Must be 2

    resarr = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
  } else
  {
    resarr = PyArray_FROM_OT(res, NPY_DOUBLE);
  }

  {

    vararr = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    numarr = PyArray_SimpleNew(nd, dims, NPY_LONG);
  }

  double* data = malloc(ninputs * sizeof(double));
  /* Assuming 2D arrays */
  npy_intp ii, jj;
  for (ii = 0; ii < dims[0]; ++ii)
    for (jj = 0; jj < dims[1]; ++jj)
    {
      int used = 0;
      /* Collect the valid values */
      for (i = 0; i < ninputs; ++i)
      {
        npy_bool *pmask = (npy_bool*) PyArray_GETPTR2(marr[i], ii, jj);
        if (*pmask == NPY_FALSE) // <- This decides how the mask is used
          continue;

        npy_double *pdata = PyArray_GETPTR2(iarr[i], ii, jj);
        data[i] = *pdata;
        ++used;
      }

      double* p = (double*) PyArray_GETPTR2(resarr, ii, jj);
      double* v = (double*) PyArray_GETPTR2(vararr, ii, jj);
      long* n = (long*) PyArray_GETPTR2(numarr, ii, jj);

      /* Compute the results*/
      method_mean(data, used, p, v, n);
    }

  free(data);

  free(iarr);
  free(marr);

  if (res != Py_None)
  {
    // If res comes from outside
    Py_INCREF(resarr);
  }
  /* If the arrays are created inside, we should use N instead of O */
  return Py_BuildValue("(N,N,N)", resarr, vararr, numarr);
}

static PyMethodDef combine_methods[] = { { "test1", (PyCFunction) py_test1,
    METH_VARARGS | METH_KEYWORDS, test1__doc__ }, { "test2",
    (PyCFunction) py_test2, METH_VARARGS | METH_KEYWORDS, test2__doc__ }, {
    "method1", py_method1, METH_VARARGS, method1__doc__ }, { NULL, NULL, 0,
    NULL } /* sentinel */
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

