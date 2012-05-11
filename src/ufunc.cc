/*
 * Copyright 2008-2012 Universidad Complutense de Madrid
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

#define PY_UFUNC_UNIQUE_SYMBOL numina_UFUNC_API
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <algorithm>
#include <vector>

// Ufunc

/*
 * When installing pyemir with pip, this ufunc breaks the compillation
 * Somehow, the symbol PyUFunc_dd_d is not loaded
 * So I remove it
 */

/*
double test1(double a, double b) {
  return 2 * a + b;
}

static char test1_sigs[] = {
    NPY_FLOAT64, NPY_FLOAT64, NPY_FLOAT64
};

static PyUFuncGenericFunction test1_functions[] = {NULL};
static void* test1_data[] = {(void*)test1};
static char test1_doc[] = "Test1 docstring.";

static void add_test1(PyObject *dictionary) {

  test1_functions[0] = PyUFunc_dd_d;

  PyObject* f = PyUFunc_FromFuncAndData(
      test1_functions,
          test1_data,
          test1_sigs,
          1,  // The number of type signatures.
          2,  // The number of inputs.
          1,  // The number of outputs.
          PyUFunc_None,  // The identity element for reduction.
                         // No good one to use for this function,
                         // unfortunately.
          "test1",  // The name of the ufunc.
          test1_doc,
          0  // Dummy for API backwards compatibility.
  );
  PyDict_SetItemString(dictionary, "test1", f);
  Py_DECREF(f);
}
*/

// Generic Ufunc
// http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
/*
 *  This implements the function  out = inner1d(in1, in2)  with
 *       out[K] = sum_i { in1[K, i] * in2[K, i] }
 *    and multi-index K, as described on
 *    http://scipy.org/scipy/numpy/wiki/GeneralLoopingFunctions
 *    and on http://projects.scipy.org/scipy/numpy/ticket/887.
 */

template<class Result, class Arg1, class Arg2>
static void
DOUBLE_test2(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
  /* Standard ufunc loop length and strides. */
  npy_intp dn = dimensions[0];
  npy_intp s0 = steps[0];
  npy_intp s1 = steps[1];
  npy_intp s2 = steps[2];

  npy_intp n;

  /* Additional loop length and strides for dimension "i" in
   * elementary function. */
  npy_intp di = dimensions[1];
  npy_intp i_s1 = steps[3];
  npy_intp i_s2 = steps[4];
  npy_intp i;

  /* Outer loop: equivalent to standard ufuncs */
  for (n = 0; n < dn; n++, args[0] += s0, args[1] += s1, args[2] +=s2) {
    char *ip1 = args[0], *ip2 = args[1], *op = args[2];

    /* Implement elementary function:  out = sum_i { in1[i] * in2[i]
     * }  */
    Result sum = 0;
    for (i = 0; i < di; i++) {
      sum += (*(Arg1 *)ip1) * (*(Arg2 *)ip2);
      ip1 += i_s1; /* Pointer to first element */
      ip2 += i_s2; /* Pointer to second element */
    }
    *(Result *)op = sum;
  }
}


/* Actually create the ufunc object */

static PyUFuncGenericFunction test2_functions[] = { DOUBLE_test2<npy_double, npy_double, npy_double> };
static void *test2_data[] = { (void *)NULL };
static char test2_sigs[] = { PyArray_DOUBLE, PyArray_DOUBLE, PyArray_DOUBLE };

static void add_test2(PyObject *dictionary) {
  PyObject *f = PyUFunc_FromFuncAndDataAndSignature(
    test2_functions,
    test2_data,
    test2_sigs,
    1,
    2,
    1,
    PyUFunc_None,
    "test2",
    "inner on the last dimension and broadcast on the other dimensions",
    0,
    "(i),(i)->()");
  PyDict_SetItemString(dictionary, "test2", f);
  Py_DECREF(f);
}

// New g-ufunc

template<class Result, class Arg>
static void
DOUBLE_median(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
  /* Standard ufunc loop length and strides. */
  npy_intp dn = dimensions[0];
  npy_intp s0 = steps[0];
  npy_intp s1 = steps[1];

  npy_intp n;

  /* Additional loop length and strides for dimension "i" in
   * elementary function. */
  npy_intp di = dimensions[1];
  npy_intp i_s1 = steps[2];
  npy_intp i;

  /* Outer loop: equivalent to standard ufuncs */
  for (n = 0; n < dn; n++, args[0] += s0, args[1] += s1) {
    char *ip1 = args[0], *op = args[1];

    std::vector<Arg> data;
    for (i = 0; i < di; i++) {
      data.push_back((*(Arg *)ip1));
      ip1 += i_s1;
    }

    npy_intp midpt = di / 2;

    if (di % 2 == 1) {
      nth_element(data.begin(), data.begin() + midpt, data.end());
      *(Result *)op = data[midpt];
    }
    else {
      nth_element(data.begin(), data.begin() + midpt, data.end());
      Arg pt1 = *(data.begin() + midpt);
      nth_element(data.begin(), data.begin() + midpt - 1, data.end());
      Arg pt2 = *(data.begin() + midpt - 1);
      *(Result *)op = 0.5 * (pt1 + pt2);
    }
  }
}


/* Actually create the ufunc object */

static PyUFuncGenericFunction median_functions[] = { DOUBLE_median<npy_double, npy_double> };
static void *median_data[] = { (void *)NULL };
static char median_sigs[] = { PyArray_DOUBLE, PyArray_DOUBLE};

static void add_median(PyObject *dictionary) {
  PyObject *f = PyUFunc_FromFuncAndDataAndSignature(
    median_functions,
    median_data,
    median_sigs,
    1,
    1,
    1,
    PyUFunc_None,
    "test3",
    "test3",
    0,
    "(i)->()");
  PyDict_SetItemString(dictionary, "test3", f);
  Py_DECREF(f);
}

PyMODINIT_FUNC init_ufunc(void)
{

  PyObject* m = Py_InitModule3("_ufunc", NULL, "Ufunc tests and examples");

  if (m == NULL)
    return;

  PyObject *d = PyModule_GetDict(m);
  if (d == NULL)
    return;

  import_array();
  import_ufunc();

//  add_test1(d);
  add_test2(d);
  add_median(d);
}
