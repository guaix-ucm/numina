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
#include <algorithm>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

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

int NU_generic_combine(PyObject** images, PyObject** masks, int size,
    PyObject* out[NU_COMBINE_OUTDIM],
    CombineFunc function,
    void* vdata,
    double* zeros,
    double* scales,
    double* weights)
{

  // Select the functions we are going to use
  // to transform the data in arrays into
  // the doubles we're working on

  PyArray_Descr* descr = NULL;

  // Conversion for inputs
  descr = PyArray_DESCR(images[0]);

  // Convert from the array to NPY_DOUBLE
  PyArray_VectorUnaryFunc* datum_converter = PyArray_GetCastFunc(descr,
      NPY_DOUBLE);
  // Swap bytes
  PyArray_CopySwapFunc* datum_swap = descr->f->copyswap;
  bool datum_need_to_swap = PyArray_ISBYTESWAPPED(images[0]);

  // Conversion for masks
  descr = PyArray_DESCR(masks[0]);
  // Convert from the array to NPY_BOOL
  PyArray_VectorUnaryFunc* mask_converter =
      PyArray_GetCastFunc(descr, NPY_BOOL);
  // Swap bytes
  PyArray_CopySwapFunc* mask_swap = descr->f->copyswap;
  bool mask_need_to_swap = PyArray_ISBYTESWAPPED(masks[0]);

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

  const size_t OUTDIM = NU_COMBINE_OUTDIM;
  // Iterators
  VectorPyArrayIter iiter(size);
  std::transform(images, images + size, iiter.begin(), &My_PyArray_IterNew);
  VectorPyArrayIter miter(size);
  std::transform(masks, masks + size, miter.begin(), &My_PyArray_IterNew);
  VectorPyArrayIter oiter(OUTDIM);
  std::transform(out, out + OUTDIM, oiter.begin(), &My_PyArray_IterNew);

  // basic iterator, we move through the
  // first result image
  PyArrayIterObject* iter = oiter[0];

  // Data and data weights
  std::vector<double> data;
  data.reserve(size);
  std::vector<double> wdata;
  wdata.reserve(size);

  // pointers to the pixels in out[0,1,2] arrays
  double* pvalues[OUTDIM];
  double values[OUTDIM];

  for (size_t i = 0; i < OUTDIM; ++i)
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
        //if (zero and scale and weight)
        {
          void* d_dtpr = (*i)->dataptr;
          // Swap the value if needed and store it in the buffer
          datum_swap(buffer, d_dtpr, datum_need_to_swap, NULL);

          double d_val = 0;
          // Convert to NPY_DOUBLE
          datum_converter(buffer, &d_val, 1, NULL, NULL);

          // Subtract zero and divide by scale
          const double converted = (d_val - zeros[ii]) / (scales[ii]);

          data.push_back(converted);
          wdata.push_back(weights[ii]);
        }
      }
    }

    // And pass the data to the combine method

    function(&data[0], &wdata[0], data.size(), pvalues, vdata);
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

  return 1;
}
