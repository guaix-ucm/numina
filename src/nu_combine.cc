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
#include <memory>

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


class Handler {
public:
  Handler() {}
  virtual ~Handler() {}
  virtual void value(char buffer[], int i, void* val) = 0;
  virtual inline void advance() = 0;
};

class NullHandler : public Handler {
public:
  NullHandler() {}
  virtual ~NullHandler() {}
  inline void value(char buffer[], int i, void* val) {}
  inline void advance() {}
};

class ImageHandler : public Handler {
public:
  ImageHandler(PyObject** frames, size_t size) :
    Handler(),
    m_frames(frames),
    m_size(size),
    m_iters(size)
  {

    // Conversion for masks
    PyArray_Descr* descr = PyArray_DESCR(m_frames[0]);
    // Convert from the array to NPY_BOOL
    m_converter = PyArray_GetCastFunc(descr, NPY_BOOL);
    // Swap bytes
    m_swap = descr->f->copyswap;
    m_need_to_swap = PyArray_ISBYTESWAPPED(m_frames[0]);

    std::transform(m_frames, m_frames + m_size, m_iters.begin(), &My_PyArray_IterNew);
  }

  void value(char buffer[], int i, void* val) {
    // Swap the value if needed and store it in the buffer
    printf("index %i %p\n",i, m_iters[i]->dataptr);
    m_swap(buffer, m_iters[i]->dataptr, m_need_to_swap, NULL);
    // Convert
    m_converter(buffer, val, 1, NULL, NULL);
  }

  void advance() {
    std::for_each(m_iters.begin(), m_iters.end(), My_PyArray_Iter_Next);
  }

  virtual ~ImageHandler() {
    std::for_each(m_iters.begin(), m_iters.end(), My_PyArray_Iter_Decref);
  }

private:
  PyObject** m_frames;
  size_t m_size;
  PyArray_VectorUnaryFunc* m_converter;
  bool m_need_to_swap;
  PyArray_CopySwapFunc* m_swap;
  VectorPyArrayIter m_iters;
};

// Checking for images
bool NU_combine_image_check(PyObject* exception, PyObject* image,
    PyObject* ref, PyObject* typeref, const char* name, size_t index) {
    if (not PyArray_Check(image)) {
      PyErr_Format(exception,
              "item %zd in %s list is not a ndarray or subclass", index, name);
      return false;
    }

    int image_ndim = PyArray_NDIM(image);

    if (PyArray_NDIM(ref) != image_ndim) {
      PyErr_Format(exception,
          "item %zd in %s list has inconsistent number of axes", index, name);
      return false;
    }

    for(int i = 0; i < image_ndim; ++i) {
      int image_dim_i = PyArray_DIM(image, i);
      if (PyArray_DIM(ref, i) != image_dim_i) {
        PyErr_Format(exception,
            "item %zd in %s list has inconsistent dimension (%i) in axis %i", index, name, image_dim_i, i);
        return false;
      }
    }

    // checking dtype is the same
    if (not PyArray_EquivArrTypes(typeref, image)) {
      PyErr_Format(exception,
          "item %zd in %s list has inconsistent dtype", index, name);
      return false;
    }
    return true;
  }

int NU_generic_combine(PyObject** images, PyObject** masks, size_t size,
    PyObject* out[NU_COMBINE_OUTDIM],
    CombineFunc function,
    void* vdata,
    const double* zeros,
    const double* scales,
    const double* weights)
{

  // Select the functions we are going to use
  // to transform the data in arrays into
  // the doubles we're working on

  PyArray_Descr* descr = NULL;

  // Conversion for inputs
  std::auto_ptr<Handler> image_handler(new ImageHandler(images, size));

  // Conversion for masks (if they exist)
  std::auto_ptr<Handler> mask_handler;

  // Mask handler
  if (masks == NULL) {
    mask_handler.reset(new NullHandler());
  } else {
    mask_handler.reset(new ImageHandler(masks, size));
  }

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

  // A buffer used to store intermediate results during swapping
  char buffer[NPY_BUFSIZE];

  const size_t OUTDIM = NU_COMBINE_OUTDIM;
  // Iterators

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
    printf("%d %d\n", iter->index, iter->size);
    for (size_t ii = 0; ii < size; ++ii)
    {


      if(weights[ii] < 0) {

        continue;
      }

      npy_bool m_val = NPY_FALSE;
      mask_handler->value(buffer, ii, &m_val);

      // True values are skipped
      if (m_val == NPY_TRUE) {
        continue;
      }

      double d_val = 0;
      image_handler->value(buffer, ii, &d_val);

      // Subtract zero and divide by scale
      const double converted = (d_val - zeros[ii]) / (scales[ii]);

      data.push_back(converted);
      wdata.push_back(weights[ii]);
    }
    for(int i = 0; i < data.size(); ++i)
      printf("%f\n", data[i]);
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
    image_handler->advance();
    mask_handler->advance();
    std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Next);
  }

  // Clean up memory (automatic for images and masks)
  std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Decref);

  return 1;
}
