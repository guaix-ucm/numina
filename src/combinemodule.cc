/*
 * Copyright 2008-2011 Sergio Pascual
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


#include <vector>
#include <memory>
#include <algorithm>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#include <numpy/arrayobject.h>

#include "unravel.h"
#include "method_factory.h"
#include "reject_factory.h"
#include "method_exception.h"

#include "nu_combine_methods.h"
#include "nu_combine.h"

using Numina::CombineMethodFactory;
using Numina::RejectMethodFactory;
using Numina::CombineMethod;
using Numina::RejectMethod;
using Numina::MethodException;
using Numina::UnRavel;

typedef std::vector<PyArrayIterObject*> VectorPyArrayIter;

PyDoc_STRVAR(combine__doc__, "Internal combine module, not to be used directly.");
PyDoc_STRVAR(internal_combine__doc__, "Combines identically shaped images");


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

// Convenience check macro
#define COMBINE_CHECK_1D_ARRAYS(ARRAY, NIMAGES) \
  if (PyArray_NDIM(ARRAY) != 1) \
  { \
    return PyErr_Format(CombineError, #ARRAY" dimension != 1"); \
  } \
  if (PyArray_SIZE(ARRAY) != NIMAGES) \
  { \
    return PyErr_Format(CombineError, #ARRAY" size != number of images"); \
  }

// Convenience check macro
#define CHECK_1D_ARRAYS(ARRAY, NIMAGES) \
  if (PyArray_NDIM(ARRAY) != 1) \
  { \
    PyErr_SetString(CombineError, #ARRAY" dimension != 1"); \
    goto exit; \
  } \
  if (PyArray_SIZE(ARRAY) != NIMAGES) \
  { \
    PyErr_SetString(CombineError, #ARRAY" size != number of images"); \
    goto exit; \
  }

#define STORE_AND_CONVERT(OUTTYPE, OUTVAR, MSG) \
    if (clean == NULL) \
    { \
      std::for_each(cleanup.begin(), cleanup.end(), My_Py_Decref); \
      return PyErr_Format(CombineError, MSG); \
    } else \
    { \
      cleanup.push_back(clean); \
      OUTVAR = (OUTTYPE) clean; \
    }

// An exception in this module
static PyObject* CombineError;

static PyObject* py_internal_combine(PyObject *self, PyObject *args,
    PyObject *kwds)
{

  // Output has one dimension more than the inputs, of size
  // OUTDIM
  const size_t OUTDIM = NU_COMBINE_OUTDIM;
  char* method = "average";
  char* reject = "none";
  PyObject *images = NULL;
  PyObject *masks = NULL;

  PyObject *out[OUTDIM] = { NULL, NULL, NULL };
  PyObject* margs = NULL;
  PyObject* rargs = NULL;

  PyArrayObject* scales = NULL;
  PyArrayObject* zeros = NULL;
  PyArrayObject* weights = NULL;

  static char *kwlist[] = {"data", "masks", "out0", "out1", "out2",
		  "method", "margs", "reject", "rargs", "zeros", "scales", "weights", NULL };

  int ok = PyArg_ParseTupleAndKeywords(args, kwds,
      "O!O!O!O!O!sO!sO!O!O!O!:internal_combine", kwlist,
      &PyList_Type, &images, &PyList_Type, &masks,
      &PyArray_Type, &out[0], &PyArray_Type, &out[1], &PyArray_Type, &out[2],
      &method, &PyTuple_Type, &margs, &reject, &PyTuple_Type, &rargs,
      &PyArray_Type, &zeros, &PyArray_Type, &scales, &PyArray_Type, &weights);

  if (!ok)
  {
    return NULL;
  }

  // Reject class
  std::auto_ptr<RejectMethod> reject_ptr;

  {
    // Method class
    std::auto_ptr<CombineMethod> method_ptr;
    try
    {
      // A factory class for named functions
      method_ptr = CombineMethodFactory::create(method, margs);
    } catch (MethodException& ex)
    {
        // If there is a problem during construction
        return PyErr_Format(CombineError,
          "error during the construction of the combination method \"%s\": %s",
          method, ex.what());
    }
    // If we don't have a method by the name
    if (not method_ptr.get())
    {
      return PyErr_Format(CombineError, "invalid combination method \"%s\"",
          method);
    }

    try
    {
      // A factory class for named functions
      reject_ptr = RejectMethodFactory::create(reject, rargs, method_ptr);
    } catch (MethodException& ex)
    {
        // If there is a problem during construction
        return PyErr_Format(CombineError,
          "error during the construction of the rejection method \"%s\": %s",
          method, ex.what());
    }
    // If we don't have a method by the name
    if (not reject_ptr.get())
    {
      return PyErr_Format(CombineError, "invalid rejection method \"%s\"",
          reject);
    }
  }
  /* images are forced to be in list */
  const Py_ssize_t nimages = PyList_GET_SIZE(images);

  if (nimages == 0)
    return PyErr_Format(CombineError, "data list is empty");

  // getting the contents inside vectors
  std::vector<PyObject*> iarr(nimages);

  // the first image
  // borrowed reference, no decref
  iarr[0] = PyList_GetItem(images, 0);

  if (not PyArray_Check(iarr[0]))
  {
    return PyErr_Format(CombineError,
        "item %i in data list is not a ndarray or subclass", 0);
  }

  for (Py_ssize_t i = 1; i < nimages; i++)
  {
    // Borrowed reference, no decref
    iarr[i] = PyList_GetItem(images, i);

    // checking we have and image
    if (not PyArray_Check(iarr[i]))
    {
      return PyErr_Format(CombineError,
          "item %zd in data list is not and ndarray or subclass", i);
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(iarr[0], iarr[i]))
      return PyErr_Format(CombineError,
          "item %zd in data list has inconsistent dtype", i);
  }

  // Masks
  std::vector<PyObject*> marr(nimages);

  // checking we have and image
  marr[0] = PyList_GetItem(masks, 0);
  if (not PyArray_Check(marr[0]))
  {
    return PyErr_Format(CombineError,
        "item %i in masks list is not a ndarray or subclass", 0);
  }

  for (Py_ssize_t i = 1; i < nimages; i++)
  {
    // Borrowed reference, no decref
    marr[i] = PyList_GetItem(masks, i);

    // checking we have and image
    if (not PyArray_Check(marr[i]))
    {
      return PyErr_Format(CombineError,
          "item %zd in masks list is not and ndarray or subclass", i);
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(marr[0], marr[i]))
      return PyErr_Format(CombineError,
          "item %zd in masks list has inconsistent dtype", i);

  }

  // Checking zeros, scales and weights
  COMBINE_CHECK_1D_ARRAYS(zeros, nimages);
  COMBINE_CHECK_1D_ARRAYS(scales, nimages);
  COMBINE_CHECK_1D_ARRAYS(weights, nimages);

  // Select the functions we are going to use
  // to transform the data in arrays into
  // the doubles we're working on

  PyArray_Descr* descr = 0;

  // Conversion for inputs
  descr = PyArray_DESCR(iarr[0]);
  // Convert from the array to NPY_DOUBLE
  PyArray_VectorUnaryFunc* datum_converter = PyArray_GetCastFunc(descr,
      NPY_DOUBLE);
  // Swap bytes
  PyArray_CopySwapFunc* datum_swap = descr->f->copyswap;
  bool datum_need_to_swap = PyArray_ISBYTESWAPPED(iarr[0]);

  // Conversion for masks
  descr = PyArray_DESCR(marr[0]);
  // Convert from the array to NPY_BOOL
  PyArray_VectorUnaryFunc* mask_converter =
      PyArray_GetCastFunc(descr, NPY_BOOL);
  // Swap bytes
  PyArray_CopySwapFunc* mask_swap = descr->f->copyswap;
  bool mask_need_to_swap = PyArray_ISBYTESWAPPED(marr[0]);

  // Conversion for outputs
  descr = PyArray_DESCR(out[0]);
  // Swap bytes
  PyArray_CopySwapFunc* out_swap = descr->f->copyswap;
  // Inverse cast
  PyArray_Descr* descr_to = PyArray_DescrFromType(NPY_DOUBLE);
  // We cast from double to the type of out array
  PyArray_VectorUnaryFunc* out_converter = PyArray_GetCastFunc(descr_to,
      PyArray_TYPE(out[0]));

  bool out_need_to_swap = PyArray_ISBYTESWAPPED(out[0]);
  // This is probably not needed
  Py_DECREF(descr_to);

  // A buffer used to store intermediate results during swapping
  char buffer[NPY_BUFSIZE];

  // Iterators
  VectorPyArrayIter iiter(nimages);
  std::transform(iarr.begin(), iarr.end(), iiter.begin(), &My_PyArray_IterNew);

  VectorPyArrayIter miter(nimages);
  std::transform(marr.begin(), marr.end(), miter.begin(), &My_PyArray_IterNew);

  VectorPyArrayIter oiter(OUTDIM);
  std::transform(out, out + OUTDIM, oiter.begin(), &My_PyArray_IterNew);

  // basic iterator, we move through the
  // first result image
  PyArrayIterObject* iter = oiter[0];

  // Data and data weights
  std::vector<double> data;
  data.reserve(nimages);
  std::vector<double> wdata;
  wdata.reserve(nimages);

  // pointers to the pixels in out[0,1,2] arrays
  double* pvalues[OUTDIM];
  double values[OUTDIM];

  for (int i = 0; i < OUTDIM; ++i)
    pvalues[i] = &values[i];

  while (iter->index < iter->size)
  {
    int ii = 0;
    VectorPyArrayIter::const_iterator i = iiter.begin();
    VectorPyArrayIter::const_iterator m = miter.begin();
    for (; i != iiter.end(); ++i, ++m, ++ii)
    {

      void* m_dtpr = (*m)->dataptr;
      // Swap the value if needed and store it in the buffer
      mask_swap(buffer, m_dtpr, mask_need_to_swap, NULL);

      npy_bool m_val = NPY_FALSE;
      // Convert to NPY_BOOL
      mask_converter(buffer, &m_val, 1, NULL, NULL);

      if (not m_val) // <- True values are skipped
      {
        // If mask converts to NPY_FALSE,
        // we store the value of the image array

        double* zero = static_cast<double*> (PyArray_GETPTR1(zeros, ii));
        double* scale = static_cast<double*> (PyArray_GETPTR1(scales, ii));
        double* weight = static_cast<double*> (PyArray_GETPTR1(weights, ii));
        if (zero and scale and weight)
        {
          void* d_dtpr = (*i)->dataptr;
          // Swap the value if needed and store it in the buffer
          datum_swap(buffer, d_dtpr, datum_need_to_swap, NULL);

          double d_val = 0;
          // Convert to NPY_DOUBLE
          datum_converter(buffer, &d_val, 1, NULL, NULL);

          // Subtract zero and divide by scale
          const double converted = (d_val - *zero) / (*scale);

          data.push_back(converted);
          wdata.push_back(*weight);
        } else
        {
          std::for_each(miter.begin(), miter.end(), My_PyArray_Iter_Decref);
          std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Decref);
          std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Decref);
          return PyErr_Format(CombineError,
              "null pointer in zero %p scale %p weight %p", zero, scale, weight);
        }
      }
    }

    // And pass the data to the combine method
    reject_ptr->combine(data.begin(), data.end(), wdata.begin(), pvalues);

    // Conversion from NPY_DOUBLE to the type of output
    for (size_t i = 0; i < OUTDIM; ++i)
    {
      // Cast to out
      out_converter(pvalues[i], buffer, 1, NULL, NULL);
      // Swap if needed
      out_swap(oiter[i]->dataptr, buffer, out_need_to_swap, NULL);
    }

    // We clean up the data storage
    data.clear();
    wdata.clear();

    // And move all the iterators to the next point
    std::for_each(iiter.begin(), iiter.end(), My_PyArray_Iter_Next);
    std::for_each(miter.begin(), miter.end(), My_PyArray_Iter_Next);
    std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Next);
  }
  // Clean up memory
  std::for_each(iiter.begin(), iiter.end(), My_PyArray_Iter_Decref);
  std::for_each(miter.begin(), miter.end(), My_PyArray_Iter_Decref);
  std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Decref);
  return Py_BuildValue("(O,O,O)", out[0], out[1], out[2]);

}

static PyObject* py_internal_combine_with_offsets(PyObject *self,
    PyObject *args, PyObject *kwds)
{

  // Output has one dimension more than the inputs, of size
  // OUTDIM
  const size_t OUTDIM = NU_COMBINE_OUTDIM;
  char *method = "average";
  char *reject = "none";
  PyObject *images = NULL;
  PyObject *masks = NULL;

  PyObject *out[OUTDIM] = { NULL, NULL, NULL };
  PyObject* margs = NULL;
  PyObject* rargs = NULL;

  PyArrayObject* scales = NULL;
  PyArrayObject* zeros = NULL;
  PyArrayObject* weights = NULL;
  PyArrayObject* offsets = NULL;

  PyObject* clean = NULL;
  std::vector<PyObject*> cleanup;

  static char *kwlist[] = {"data", "masks", "offsets", "out0", "out1", "out2",
		  "method", "margs","reject", "rargs", "zeros", "scales", "weights", NULL };

  int ok = PyArg_ParseTupleAndKeywords(args, kwds,
      "O!O!O!O!O!O!sO!sO!O!O!O!:internal_combine_with_offsets", kwlist,
      &PyList_Type, &images, &PyList_Type, &masks, &PyArray_Type, &offsets,
      &PyArray_Type, &out[0], &PyArray_Type, &out[1], &PyArray_Type, &out[2],
      &method, &PyTuple_Type, &margs, &reject, &PyTuple_Type, &rargs,
      &PyArray_Type, &zeros, &PyArray_Type, &scales, &PyArray_Type, &weights);

  if (!ok)
  {
    return NULL;
  }



  // Reject class
  std::auto_ptr<RejectMethod> reject_ptr;

  {
    // Method class
    std::auto_ptr<CombineMethod> method_ptr;
    try
    {
      // A factory class for named functions
      method_ptr = CombineMethodFactory::create(method, margs);
    } catch (MethodException& ex)
    {
        // If there is a problem during construction
        return PyErr_Format(CombineError,
          "error during the construction of the combination method \"%s\": %s",
          method, ex.what());
    }
    // If we don't have a method by the name
    if (not method_ptr.get())
    {
      return PyErr_Format(CombineError, "invalid combination method \"%s\"",
          method);
    }

    try
    {
      // A factory class for named functions
      reject_ptr = RejectMethodFactory::create(reject, rargs, method_ptr);
    } catch (MethodException& ex)
    {
        // If there is a problem during construction
        return PyErr_Format(CombineError,
          "error during the construction of the rejection method \"%s\": %s",
          method, ex.what());
    }
    // If we don't have a method by the name
    if (not reject_ptr.get())
    {
      return PyErr_Format(CombineError, "invalid rejection method \"%s\"",
          reject);
    }
  }

  /* images are forced to be in list */
  const Py_ssize_t nimages = PyList_GET_SIZE(images);

  if (nimages == 0)
    return PyErr_Format(CombineError, "data list is empty");

  // getting the contents inside vectors
  std::vector<PyObject*> iarr(nimages);

  // the first image
  // borrowed reference, no decref
  iarr[0] = PyList_GetItem(images, 0);

  if (not PyArray_Check(iarr[0]))
  {
    return PyErr_Format(CombineError,
        "item %i in data list is not a ndarray or subclass", 0);
  }

  for (Py_ssize_t i = 1; i < nimages; i++)
  {
    // Borrowed reference, no decref
    iarr[i] = PyList_GetItem(images, i);

    // checking we have and image
    if (not PyArray_Check(iarr[i]))
    {
      return PyErr_Format(CombineError,
          "item %zd in data list is not and ndarray or subclass", i);
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(iarr[0], iarr[i]))
      return PyErr_Format(CombineError,
          "item %zd in data list has inconsistent dtype", i);
  }

  // Masks
  std::vector<PyObject*> marr(nimages);

  // checking we have and image
  marr[0] = PyList_GetItem(masks, 0);
  if (not PyArray_Check(marr[0]))
  {
    return PyErr_Format(CombineError,
        "item %i in masks list is not a ndarray or subclass", 0);
  }

  for (Py_ssize_t i = 1; i < nimages; i++)
  {
    // Borrowed reference, no decref
    marr[i] = PyList_GetItem(masks, i);

    // checking we have and image
    if (not PyArray_Check(marr[i]))
    {
      return PyErr_Format(CombineError,
          "item %zd in masks list is not and ndarray or subclass", i);
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(marr[0], marr[i]))
      return PyErr_Format(CombineError,
          "item %zd in masks list has inconsistent dtype", i);

  }

  int ndim = PyArray_NDIM(iarr[0]);

  // Checking offsets
  if (PyArray_NDIM(offsets) != 2)
  {
    return PyErr_Format(CombineError, "offsets dimension != 2");
  }
  if (PyArray_SIZE(offsets) != nimages * ndim)
  {
    return PyErr_Format(CombineError, "offsets size != ndim * number of images");
  }

  // Checking zeros, scales and weights
  COMBINE_CHECK_1D_ARRAYS(zeros, nimages);
  COMBINE_CHECK_1D_ARRAYS(scales, nimages);
  COMBINE_CHECK_1D_ARRAYS(weights, nimages);

  // Select the functions we are going to use
  // to transform the data in arrays into
  // the doubles we're working on

  PyArray_Descr* descr = 0;

  // Conversion for inputs
  descr = PyArray_DESCR(iarr[0]);
  // Convert from the array to NPY_DOUBLE
  PyArray_VectorUnaryFunc* datum_converter = PyArray_GetCastFunc(descr,
      NPY_DOUBLE);
  // Swap bytes
  PyArray_CopySwapFunc* datum_swap = descr->f->copyswap;
  bool datum_need_to_swap = PyArray_ISBYTESWAPPED(iarr[0]);

  // Conversion for masks
  descr = PyArray_DESCR(marr[0]);
  // Convert from the array to NPY_BOOL
  PyArray_VectorUnaryFunc* mask_converter =
      PyArray_GetCastFunc(descr, NPY_BOOL);
  // Swap bytes
  PyArray_CopySwapFunc* mask_swap = descr->f->copyswap;
  bool mask_need_to_swap = PyArray_ISBYTESWAPPED(marr[0]);

  // Conversion for outputs
  descr = PyArray_DESCR(out[0]);
  // Swap bytes
  PyArray_CopySwapFunc* out_swap = descr->f->copyswap;
  // Inverse cast
  PyArray_Descr* descr_to = PyArray_DescrFromType(NPY_DOUBLE);
  // We cast from double to the type of out array
  PyArray_VectorUnaryFunc* out_converter = PyArray_GetCastFunc(descr_to,
      PyArray_TYPE(out[0]));

  bool out_need_to_swap = PyArray_ISBYTESWAPPED(out[0]);
  // This is probably not needed
  Py_DECREF(descr_to);

  // A buffer used to store intermediate results during swapping
  char buffer[NPY_BUFSIZE];

  // Iterators
  VectorPyArrayIter iiter(nimages);
  VectorPyArrayIter miter(nimages);

  for (size_t i = 0; i < iiter.size(); ++i)
  {
    clean = PyArray_IterNew(iarr[i]);
    STORE_AND_CONVERT(PyArrayIterObject*, iiter[i], "NULL during data array iterator");
    clean = PyArray_IterNew(marr[i]);
    STORE_AND_CONVERT(PyArrayIterObject*, miter[i], "NULL during mask array iterator");
  }

  VectorPyArrayIter oiter(OUTDIM);

  for (size_t i = 0; i < OUTDIM; ++i)
  {
    clean = PyArray_IterNew(out[i]);
    STORE_AND_CONVERT(PyArrayIterObject*, oiter[i], "NULL during output array iterator")
  }

  // basic iterator, we move through the
  // first result image
  PyArrayIterObject* iter = oiter[0];

  // Data and data weights
  std::vector<double> data;
  data.reserve(nimages);
  std::vector<double> wdata;
  wdata.reserve(nimages);

  // pointers to the pixels in out[0,1,2] arrays
  double* pvalues[OUTDIM];
  double values[OUTDIM];

  for (int i = 0; i < OUTDIM; ++i)
    pvalues[i] = &values[i];

  npy_intp new_coordinates[NPY_MAXDIMS];

  const UnRavel unravel(PyArray_DIMS(out[0]), ndim);

  while (iter->index < iter->size)
  {
    int ii = 0;

    VectorPyArrayIter::const_iterator i = iiter.begin();
    VectorPyArrayIter::const_iterator m = miter.begin();
    for (; i != iiter.end(); ++i, ++m, ++ii)
    {

      bool valid_coordinate = true;

      unravel.index_copy(iter->index, new_coordinates, ndim);

      for (int jj = 0; jj < ndim; ++jj)
      {
        npy_int* off = static_cast<npy_int*> (PyArray_GETPTR2(offsets, ii, jj));
        new_coordinates[jj] -= *off;
        if ((new_coordinates[jj] < 0) or (new_coordinates[jj]
            > (*m)->dims_m1[jj]))
        {
          valid_coordinate = false;
          break;
        }
      }

      if (not valid_coordinate)
        continue;

      PyArray_ITER_GOTO(*m, new_coordinates);

      void* m_dtpr = (*m)->dataptr;
      // Swap the value if needed and store it in the buffer
      mask_swap(buffer, m_dtpr, mask_need_to_swap, NULL);

      npy_bool m_val = NPY_FALSE;
      // Convert to NPY_BOOL
      mask_converter(buffer, &m_val, 1, NULL, NULL);

      if (not m_val) // <- True values are skipped
      {
        // If mask converts to NPY_FALSE,
        // we store the value of the image array

        double* zero = static_cast<double*> (PyArray_GETPTR1(zeros, ii));
        double* scale = static_cast<double*> (PyArray_GETPTR1(scales, ii));
        double* weight = static_cast<double*> (PyArray_GETPTR1(weights, ii));

        if (zero and scale and weight)
        {
          PyArray_ITER_GOTO(*i, new_coordinates);

          void* d_dtpr = (*i)->dataptr;
          // Swap the value if needed and store it in the buffer
          datum_swap(buffer, d_dtpr, datum_need_to_swap, NULL);

          double d_val = 0;
          // Convert to NPY_DOUBLE
          datum_converter(buffer, &d_val, 1, NULL, NULL);

          // Subtract zero and divide by scale
          const double converted = (d_val - *zero / *scale);

          data.push_back(converted);
          wdata.push_back(*weight);

        } else
        {
          std::for_each(cleanup.begin(), cleanup.end(), My_Py_Decref);
          return PyErr_Format(CombineError,
              "null pointer in zero %p scale %p weight %p", zero, scale, weight);
        }
      }
    }

    // And pass the data to the combine method
    reject_ptr->combine(data.begin(), data.end(), wdata.begin(), pvalues);

    // Conversion from NPY_DOUBLE to the type of output
    for (size_t i = 0; i < OUTDIM; ++i)
    {
      // Cast to out
      out_converter(pvalues[i], buffer, 1, NULL, NULL);
      // Swap if needed
      out_swap(oiter[i]->dataptr, buffer, out_need_to_swap, NULL);
    }

    // We clean up the data storage
    data.clear();
    wdata.clear();

    std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Next);
  }

  std::for_each(cleanup.begin(), cleanup.end(), My_Py_Decref);
  return Py_BuildValue("(O,O,O)", out[0], out[1], out[2]);
}

static PyObject* py_generic_combine(PyObject *self, PyObject *args)
{
  /* Arguments */
  PyObject *images = NULL;
  PyObject *masks = NULL;
  // Output has one dimension more than the inputs, of size
  // OUTDIM
  const size_t OUTDIM = NU_COMBINE_OUTDIM;
  PyObject *out[OUTDIM] = {NULL, NULL, NULL};
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

  int ok = PyArg_ParseTuple(args,
      "OOO!O!O!OO!O!O!:generic_combine",
      &images,
      &masks,
      &PyArray_Type, &out[0],
      &PyArray_Type, &out[1],
      &PyArray_Type, &out[2],
      &fnc,
      &PyArray_Type, &zeros,
      &PyArray_Type, &scales,
      &PyArray_Type, &weights);

  if (!ok)
  {
    goto exit;
  }

  images_seq = PySequence_Fast(images, "expected a sequence");
  nimages = PySequence_Size(images_seq);

  masks_seq = PySequence_Fast(masks, "expected a sequence");
  nmasks = PySequence_Size(masks_seq);

  if (nimages != nmasks) {
    PyErr_Format(CombineError, "number of images (%zd) and masks (%zd) is different", nimages, nmasks);
    goto exit;
  }

  if (nimages == 0) {
    PyErr_Format(CombineError, "data list is empty");
    goto exit;
  }

  allimages = PySequence_Fast_ITEMS(images_seq);
  allmasks = PySequence_Fast_ITEMS(masks_seq);

  for(ui = 0; ui < nimages; ++ui) {
    if (not PyArray_Check(allimages[ui])) {
      PyErr_Format(CombineError,
              "item %zd in data list is not a ndarray or subclass", ui);
      goto exit;
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(allimages[0], allimages[ui])) {
      PyErr_Format(CombineError,
          "item %zd in data list has inconsistent dtype", ui);
      goto exit;
    }

    if (not PyArray_Check(allmasks[ui])) {
      PyErr_Format(CombineError,
              "item %zd in masks list is not a ndarray or subclass", ui);
      goto exit;
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(allmasks[0], allmasks[ui])) {
      PyErr_Format(CombineError,
          "item %zd in masks list has inconsistent dtype", ui);
      goto exit;

    }
  }

  if (PyCObject_Check(fnc)) {
      func = PyCObject_AsVoidPtr(fnc);
      data = PyCObject_GetDesc(fnc);
  } else {
      PyErr_SetString(PyExc_RuntimeError,
                                      "function parameter is not callable");
      goto exit;
  }

  // Checking zeros, scales and weights
  CHECK_1D_ARRAYS(zeros, nimages);
  CHECK_1D_ARRAYS(scales, nimages);
  CHECK_1D_ARRAYS(weights, nimages);

  zeros_arr = PyArray_FROM_OTF(zeros, NPY_DOUBLE, NPY_IN_ARRAY);
  scales_arr = PyArray_FROM_OTF(scales, NPY_DOUBLE, NPY_IN_ARRAY);
  weights_arr = PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_IN_ARRAY);

  if( not NU_generic_combine(allimages, allmasks, nimages, out,
      (CombineFunc)func, data,
      (double*)PyArray_DATA(zeros_arr),
      (double*)PyArray_DATA(scales_arr),
      (double*)PyArray_DATA(weights_arr)
      )
    )
    goto exit;

exit:
  Py_XDECREF(images_seq);
  Py_XDECREF(masks_seq);
  Py_XDECREF(zeros_arr);
  Py_XDECREF(scales_arr);
  Py_XDECREF(weights_arr);
  return PyErr_Occurred() ? NULL : Py_BuildValue("");

}

static PyObject *
py_method_mean(PyObject *obj, PyObject *args) {
  return PyCObject_FromVoidPtr((void*)NU_mean_function, NULL);
}

static PyObject *
py_method_median(PyObject *obj, PyObject *args) {
  return PyCObject_FromVoidPtr((void*)NU_median_function, NULL);
}

static PyObject *
py_method_minmax(PyObject *obj, PyObject *args) {
  unsigned int nmin = 0;
  unsigned int nmax = 0;
  if (not PyArg_ParseTuple(args, "II", &nmin, &nmax)) {
    PyErr_SetString(PyExc_RuntimeError, "invalid parameters");
    return NULL;
  }

  unsigned int *funcdata = (unsigned int*)malloc(2 * sizeof(unsigned int));

  funcdata[0] = nmin;
  funcdata[1] = nmax;

  return PyCObject_FromVoidPtrAndDesc((void*)NU_minmax_function, funcdata, NU_destructor_function);
}

static PyObject *
py_method_sigmaclip(PyObject *obj, PyObject *args) {
  double low = 0.0;
  double high = 0.0;
  if (not PyArg_ParseTuple(args, "dd", &low, &high)) {
    PyErr_SetString(PyExc_RuntimeError, "invalid parameters");
    return NULL;
  }

  double *funcdata = (double*)malloc(2 * sizeof(double));

  funcdata[0] = low;
  funcdata[1] = high;

  return PyCObject_FromVoidPtrAndDesc((void*)NU_sigmaclip_function, funcdata, NU_destructor_function);

}
static PyObject *
py_method_quantileclip(PyObject *obj, PyObject *args) {
  double fclip = 0.0;
  if (not PyArg_ParseTuple(args, "d", &fclip)) {
    PyErr_SetString(PyExc_RuntimeError, "invalid parameters");
    return NULL;
  }

  double *funcdata = (double*)malloc(sizeof(double));
  *funcdata = fclip;
  return PyCObject_FromVoidPtrAndDesc((void*)NU_quantileclip_function, funcdata, NU_destructor_function);
}



static PyMethodDef combine_methods[] = {
    { "internal_combine", (PyCFunction) py_internal_combine, METH_VARARGS | METH_KEYWORDS,
    internal_combine__doc__ },
    { "internal_combine_with_offsets", (PyCFunction) py_internal_combine_with_offsets, METH_VARARGS
        | METH_KEYWORDS, internal_combine__doc__ },
    {"generic_combine", (PyCFunction) py_generic_combine, METH_VARARGS, ""},
    {"mean_method", (PyCFunction) py_method_mean, METH_VARARGS, ""},
    {"median_method", (PyCFunction) py_method_median, METH_VARARGS, ""},
    {"minmax_method", (PyCFunction) py_method_minmax, METH_VARARGS, ""},
    {"sigmaclip_method", (PyCFunction) py_method_sigmaclip, METH_VARARGS, ""},
    {"quantileclip_method", (PyCFunction) py_method_sigmaclip, METH_VARARGS, ""},
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
    CombineError = PyErr_NewException("_combine.CombineError", NULL, NULL);
  }
  Py_INCREF(CombineError);
  PyModule_AddObject(m, "CombineError", CombineError);
}
