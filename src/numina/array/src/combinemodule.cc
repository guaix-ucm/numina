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

// An exception in this module
static PyObject* CombineError;

// Convenience check function
static inline int check_1d_array(PyObject* array, size_t nimages, const char* name) {
  if (PyArray_NDIM((PyArrayObject*)array) != 1)  //  error: cannot convert ‘PyObject*’ {aka ‘_object*’} to ‘const PyArrayObject*’
  {
    PyErr_Format(PyExc_ValueError, "%s dimension %i != 1", name, PyArray_NDIM((PyArrayObject*)array));
    return 0;
  }
  if (PyArray_SIZE((PyArrayObject*)array) != (npy_intp)nimages) //  cannot convert ‘PyObject*’ {aka ‘_object*’} to ‘const PyArrayObject*’
  {
    PyErr_Format(PyExc_ValueError, "%s size %zd != number of images", name, PyArray_SIZE((PyArrayObject*)array));
    return 0;
  }
  return 1;
}

static inline int advance_multi_iter(const std::vector<NpyIter_IterNextFunc*>& iter_next, const std::vector<NpyIter*>& iters) {
  int val;
  for(int i=0; i < iters.size(); ++i) {
      val = iter_next[i](iters[i]);
      //printf("iter %d, return val %d\n", i, val);
      if(val == 0)
        break;
  }
  return val;
}


static PyObject* py_generic_combine(PyObject *self, PyObject *args, PyObject *kwargs)
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
  std::vector<double> pdata;
  std::vector<double> wdata;
  double results_buffer[3];
  double* pvalues[3];

  int outval;
  const char *kwlist[] = {"method", "images", "out_res", "out_var", "out_pix",
    "masks", "zeros", "scales", "weights", NULL};

  outval = PyArg_ParseTupleAndKeywords(args, kwargs,
      "OO|OOOOOOO:generic_combine", const_cast<char**>(kwlist),
      &fnc, /* capsule */
      &images, /* sequence of images */
      &out_res, /* result */
      &out_var, /* variance */
      &out_pix, /* npix */
      &masks, /* sequence of masks */
      &zeros, /* sequence of zeros */
      &scales,/* sequence of scales */
      &weights /* sequence of weights */
      );

  if (!outval) return NULL;

  images_seq = PySequence_Fast(images, "expected a sequence of data arrays");
  nimages = PySequence_Size(images_seq);

  if (nimages == 0) {
    PyErr_SetString(PyExc_ValueError, "data sequence is empty");
    goto exit;
  }
  if (masks != NULL && masks != Py_None) {
    masks_seq = PySequence_Fast(masks, "expected a sequence of mask arrays");
    nmasks = PySequence_Size(masks_seq);
    if (nmasks != nimages) {
      PyErr_Format(PyExc_ValueError,
          "number of images (%zd) and masks (%zd) is different", nimages, nmasks);

      goto exit;
    }
  }

  if (out_res == Py_None) {
     out_res = NULL;
  }

  if (out_var == Py_None) {
     out_var = NULL;
  }

  if (out_pix == Py_None) {
     out_pix = NULL;
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
  op_dtypes.push_back(dtype_res);
  std::fill_n(std::back_inserter(op_flags), nout,
                  NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO| NPY_ITER_ALIGNED);
  /* masks */
  if (nmasks != 0) {
    tmp_list = PySequence_Fast_ITEMS(masks_seq);
    std::copy(tmp_list, tmp_list + nmasks, std::back_inserter(ops));
    std::fill_n(std::back_inserter(op_flags), nmasks, NPY_ITER_READONLY | NPY_ITER_NBO);
    std::fill_n(std::back_inserter(op_dtypes), nmasks, dtype_pix);
  }

  iter = NpyIter_MultiNew(nimages + nout + nmasks, (PyArrayObject**)(&ops[0]),
                  NPY_ITER_BUFFERED,
                  NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                  &op_flags[0], &op_dtypes[0]);
  Py_DECREF(dtype_res);
  Py_DECREF(dtype_pix);
  op_dtypes.resize(0); /* clean up */
  dtype_res = dtype_pix = NULL;

  if (iter == NULL) {
      goto exit;
  }

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);

  pdata.reserve(nimages);
  wdata.reserve(nimages);

  // We need allocated memory
  for (ui = 0; ui < 3; ++ui)
    pvalues[ui] = &results_buffer[ui];

  do {
      double* p_val;
      double converted;

      int ncalc;

      for(ui = 0; ui < nimages; ++ui) {
        npy_bool m_val = NPY_FALSE;

        if(wbuffer[ui] < 0)
          continue;

        if(nmasks == nimages) {
           m_val = (* (int*)dataptr[nimages + nout + ui]) > 0;
        }

        if (m_val == NPY_TRUE)
          continue;

        p_val = (double*)dataptr[ui];
        converted = (*p_val - zbuffer[ui]) / (sbuffer[ui]);
        pdata.push_back(converted);
        wdata.push_back(wbuffer[ui]);
      }

      // And pass the data to the combine method
      if (pdata.size() > 0) {
          outval = ((CombineFunc)capsule_func)(&pdata[0], &wdata[0], pdata.size(), pvalues, capsule_data);
          if(!outval) {
             PyErr_SetString(PyExc_RuntimeError, "unknown error in combine method");
             NpyIter_Deallocate(iter);
             goto exit;
          }
      }
      else {
          /* without points, 0 and 1 should be nan */
         *pvalues[0] = *pvalues[1] = *pvalues[2] = 0.0;
      }

      memcpy(dataptr[nimages + 0], pvalues[0], sizeof(double));
      memcpy(dataptr[nimages + 1], pvalues[1], sizeof(double));
      memcpy(dataptr[nimages + 2], pvalues[2], sizeof(double));


      pdata.clear();
      wdata.clear();


    } while(iternext(iter));

    if (out_res == NULL) {
        out_res = (PyObject*)NpyIter_GetOperandArray(iter)[nimages + 0];
     }
    Py_INCREF(out_res);

    if (out_var == NULL) {
        out_var = (PyObject*)NpyIter_GetOperandArray(iter)[nimages + 1];
    }
    Py_INCREF(out_var);
    if (out_pix == NULL) {
        out_pix = (PyObject*)NpyIter_GetOperandArray(iter)[nimages + 2];
    }
    Py_INCREF(out_pix);

  NpyIter_Deallocate(iter);

exit:
  Py_XDECREF(images_seq);
  Py_XDECREF(masks_seq);

  if (zeros == Py_None || zeros == NULL)
    PyMem_Free(zbuffer);

  Py_XDECREF(zeros_arr);

  if (scales == Py_None || scales == NULL)
    PyMem_Free(sbuffer);

  Py_XDECREF(scales_arr);

  if (weights == Py_None || scales == NULL)
    PyMem_Free(wbuffer);

  Py_XDECREF(weights_arr);

  return PyErr_Occurred() ? NULL : Py_BuildValue("(N,N,N)",out_res, out_var, out_pix);
}



static PyObject* py_generic_combine_miter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  /* workaround the limitation in NPY_MAXARGS by having several iterators
     if some day the limitation is removed, we could use the impl in "py_generic_combine"

     We can have a mask for each image (1) or no masks (2). In both cases, the first iter
     will contain out1/out2/out3 as the last operands (WRITEONLY | ALLOCATE) and the
     following posible iters will out1 as the last operand as READONLY. The purpose of including
     out1 in all iters is to have a common shape to broadcast

     In 1) each image and its mask are inserted together, in pairs. The first iterator
     will contain  img/mask/img/mask/.../out1/out2/out3. The following iters will
     contain img/mask/img/mask/..../out1.

     In 2), the first iterator
     will contain  img/img/img/.../out1/out2/out3. The following iters will
     contain img/img/img/.../out1.
  */

  /* Inputs */
  PyObject *images = NULL;
  PyObject *images_seq = NULL;
  Py_ssize_t nimages = 0;
  /* masks */
  PyObject *masks = NULL;
  PyObject *masks_seq = NULL;
  Py_ssize_t nmasks = 0;
  PyObject** tmp_list_i = NULL;
  PyObject** tmp_list_m = NULL;

  npy_bool has_masks = NPY_FALSE;
  /* outputs */
  const Py_ssize_t nout = 3;
  const Py_ssize_t nouti = 1;
  Py_ssize_t nout_x;
  PyObject *out_arr[nout] = {NULL, NULL, NULL};
  Py_ssize_t ui;

  /* MAXARGS */
  const Py_ssize_t CAP = NPY_MAXARGS;  // 64 in np2, 32 in np1
  // has_masks == T <-> group_size == 2
  // has_masks == F <-> group_size == 1
  Py_ssize_t group_size = 1;

  Py_ssize_t max_ins0, max_insi, max_ins;

  /* scales, zeros and weights */
  PyObject* scales = NULL;
  PyObject* zeros = NULL;
  PyObject* weights = NULL;
  PyObject* zeros_arr = NULL;
  PyObject* scales_arr = NULL;
  PyObject* weights_arr = NULL;
  double* zbuffer = NULL;
  double* sbuffer = NULL;
  double* wbuffer = NULL;

  /* Capsule */
  PyObject* fnc = NULL;
  void *capsule_func = (void*)NU_mean_function;
  void *capsule_data = NULL;

  /* Iterator */
  PyArray_Descr* dtype_res = NULL;
  PyArray_Descr* dtype_pix = NULL;
  npy_uint32 iflags;
  npy_uint32 flag_out = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO| NPY_ITER_ALIGNED;
  npy_uint32 flag_in = NPY_ITER_READONLY | NPY_ITER_NBO;
  npy_uint32 flag_pix = NPY_ITER_READONLY | NPY_ITER_NBO;
  std::vector<npy_uint32> op_flags;
  std::vector<PyArray_Descr*> op_dtypes;
  std::vector<PyObject*> ops;
  /* base iterator */
  NpyIter *tmp_iter = NULL;

  std::vector<NpyIter*> iters;
  std::vector<NpyIter_IterNextFunc*> iter_next;
  std::vector<char**> dataptrs;

  std::vector<Py_ssize_t> iter_insert;
  std::vector<Py_ssize_t> img_iter;
  std::vector<Py_ssize_t> img_pos;

  //
  int iter_count = 0;
  int base_out = 0;

  Py_ssize_t remain_groups = 0;
  Py_ssize_t inserts;
  Py_ssize_t base_idx;

  /* NU_generic_combine */
  std::vector<double> pdata;
  std::vector<double> wdata;
  double results_buffer[nout];
  double* pvalues[nout];

  int outval;
  const char *arr_name[nout] = {"res", "var", "pix"};
  const char *kwlist[] = {"method", "images", "out_res", "out_var", "out_pix",
    "masks", "zeros", "scales", "weights", NULL};

  outval = PyArg_ParseTupleAndKeywords(args, kwargs,
      "OO|OOOOOOO:generic_combine", const_cast<char**>(kwlist),
      &fnc, /* capsule */
      &images, /* sequence of images */
      &out_arr[0], /* result */
      &out_arr[1], /* variance */
      &out_arr[2], /* npix */
      &masks, /* sequence of masks */
      &zeros, /* sequence of zeros */
      &scales,/* sequence of scales */
      &weights /* sequence of weights */
      );
  //printf("outval %d\n", outval);

  if (!outval) return NULL;

  images_seq = PySequence_Fast(images, "expected a sequence of data arrays");
  nimages = PySequence_Size(images_seq);
  // printf("We have nimages=%d\n", nimages);

  if (nimages == 0) {
    PyErr_SetString(PyExc_ValueError, "data sequence is empty");
    goto exit;
  }
  if (masks != NULL && masks != Py_None) {
    masks_seq = PySequence_Fast(masks, "expected a sequence of mask arrays");
    nmasks = PySequence_Size(masks_seq);
    if (nmasks == nimages) {
      has_masks = NPY_TRUE;
      group_size = 2;
    }
    else if (nmasks == 0) {
      has_masks = NPY_FALSE;
      group_size = 1;
      }
    else {
      PyErr_Format(PyExc_ValueError,
          "number of images (%zd) and masks (%zd) is different", nimages, nmasks);

      goto exit;
    }
  }
  // printf("We have nmasks=%d\n", nmasks);

  for(int i = 0; i < nout; ++i) {
    if (out_arr[i] == Py_None) {
          out_arr[i] = NULL;
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

  pdata.reserve(nimages);
  wdata.reserve(nimages);

  // We need allocated memory
  for (ui = 0; ui < nout; ++ui)
    pvalues[ui] = &results_buffer[ui];

  // Insertions in iter0 and iterI
  max_ins0 = (CAP - nout) / group_size;
  max_insi = (CAP - nouti) / group_size;

  //printf("MAX_insert0 %d\n", max_ins0);
  //printf("MAX_insertI %d\n", max_insi);
  //printf("MAX_cap0 %d\n", max_ins0 * group_size);
  //printf("MAX_capI %d\n", max_insi * group_size);

  remain_groups = nimages;
  base_idx = 0;
  /* iters*/
  dtype_res = PyArray_DescrFromType(NPY_DOUBLE);
  dtype_pix = PyArray_DescrFromType(NPY_INT);

  tmp_list_i = PySequence_Fast_ITEMS(images_seq);
  tmp_list_m = NULL;
  if (has_masks == NPY_TRUE) {
    tmp_list_m = PySequence_Fast_ITEMS(masks_seq);
  }

  /* insert outputs in ops, op_dtypes and op_flags*/
  while (remain_groups > 0) {
    /* images stored in iter 0 */
    if (iter_count == 0) {
      max_ins = max_ins0;
      nout_x = nout;
    }
    else {
      max_ins = max_insi;
      nout_x = nouti;
    }

    inserts = std::min(remain_groups, max_ins);
    iter_insert.push_back(inserts);

    /*    printf(".add iter %d\n", iter_count);
    printf("...insert %d\n", inserts);
    printf("...occupancy %d  (%d)\n", inserts * group_size, inserts * group_size + nout_x);
    */

    for(int i=base_idx; i < base_idx + inserts; ++i) {
      // Insert alternating image/mask
      //printf("...a i iter=%d index=%d\n",iter_count, i);
      ops.push_back(tmp_list_i[i]);
      op_flags.push_back(flag_in);
      op_dtypes.push_back(dtype_res);
      // Counters
      img_iter.push_back(iter_count);
      img_pos.push_back((i - base_idx) * group_size);
      if (has_masks == NPY_TRUE) {
        //printf("...a m iter=%d index=%d\n",iter_count, i);
        ops.push_back(tmp_list_m[i]);
        op_flags.push_back(flag_pix);
        op_dtypes.push_back(dtype_pix);
      }
    }

    // Output at the end
    if(iter_count == 0) {
      // First iter
      for(int i=0; i < nout_x; ++i) {
        ops.push_back(out_arr[i]);
        op_dtypes.push_back(dtype_res);
        op_flags.push_back(flag_out);
      }
    } else {
      // We insert one copy of the output, as read-only
      // to allow broadcasting
      if (out_arr[0] == NULL) {
        ops.push_back((PyObject*)NpyIter_GetOperandArray(iters[0])[iter_insert[0]]);
      }
      else {
        ops.push_back(out_arr[0]);
      }
      op_dtypes.push_back(dtype_res);
      op_flags.push_back(flag_in);
    }

    remain_groups -= inserts;
    base_idx += inserts;
    // printf("...remaining %d, groups=%d\n", remain_groups * group_size, remain_groups);
    // printf("......%d %d %d %d\n", cap0 + nout_x, ops.size(), op_flags.size(), op_dtypes.size());

    tmp_iter = NpyIter_MultiNew(ops.size(), (PyArrayObject**)(&ops[0]),
                  NPY_ITER_BUFFERED,
                  NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                  &op_flags[0], &op_dtypes[0]);

    /* clean up vectors */
    ops.clear();
    op_dtypes.clear();
    op_flags.clear();

    /* if we fail in iter creation, clean up */
    if (tmp_iter == NULL) {
      Py_DECREF(dtype_res);
      Py_DECREF(dtype_pix);
      dtype_res = dtype_pix = NULL;
      goto exit;
    }

    // Store iter and auxiliary functions
    iters.push_back(tmp_iter);
    iter_next.push_back(NpyIter_GetIterNext(tmp_iter, NULL));
    dataptrs.push_back(NpyIter_GetDataPtrArray(tmp_iter));
    iter_count++;
    // Repeat while there are remaining images
  }
  //printf("we have %d (%d) iters\n", iter_count, iters.size());
  // printf("img indirection %d (%d)\n", img_iter.size(), img_pos.size());

  base_out = iter_insert[0] * group_size;

  do {
    double* p_val;
    double converted;

    for(ui = 0; ui < nimages; ++ui) {
      npy_bool m_val = NPY_FALSE;
      int idx_iter;
      int idx_pos;

      // If a weight is negative, we ignore the image
      if(wbuffer[ui] < 0)
        continue;

      // Iterator and position within for ehac image
      idx_iter = img_iter[ui];
      idx_pos = img_pos[ui];

      //printf("+++-----------------\n");
      //printf("img %d iter %d pos %d\n", ui, idx_iter, idx_pos);
      // printf("img %d iter %d pos %d NOP %d\n", ui, idx_iter, idx_pos, NpyIter_GetNOp(iters[idx_iter]));

      if(has_masks == NPY_TRUE) {
        /* access dataptrs[i][j + 1] */
        //m_val = (* (int*)dataptrs[idx_iter][idx_pos + 1]);
        //printf("mask of img=%d, value=%d\n", ui, m_val);
        m_val = (* (int*)dataptrs[idx_iter][idx_pos + 1]) > 0;
      }

      if (m_val == NPY_TRUE)
        continue;

      /* access dataptrs[i][j] */
      p_val = (double*)dataptrs[idx_iter][idx_pos];
      //printf("img=%d, value=%f\n", ui, *p_val);

      converted = (*p_val - zbuffer[ui]) / (sbuffer[ui]);
      pdata.push_back(converted);
      wdata.push_back(wbuffer[ui]);
      //printf("------------------\n");
    }

    // And pass the data to the combine method
    if (pdata.size() > 0) {
      outval = ((CombineFunc)capsule_func)(&pdata[0], &wdata[0], pdata.size(), pvalues, capsule_data);
      if(!outval) {
        PyErr_SetString(PyExc_RuntimeError, "unknown error in combine method");
        goto exit;
      }
    }
    else {
      /* without points, 0 and 1 should be nan */
      *pvalues[0] = *pvalues[1] = *pvalues[2] = 0.0;
    }

    //printf("results  %f  %f %f \n",*pvalues[0], *pvalues[1], *pvalues[2]);

    memcpy(dataptrs[0][base_out + 0], pvalues[0], sizeof(double));
    memcpy(dataptrs[0][base_out + 1], pvalues[1], sizeof(double));
    memcpy(dataptrs[0][base_out + 2], pvalues[2], sizeof(double));

    pdata.clear();
    wdata.clear();

  } while(advance_multi_iter(iter_next, iters));

  // Recover outputs
  for(int i=0; i < nout; ++i) {
    if (out_arr[i] == NULL) {
      out_arr[i] = (PyObject*)NpyIter_GetOperandArray(iters[0])[base_out + i];
    }
    Py_INCREF(out_arr[i]);
  }

exit:

  for(auto the_iter: iters) {
    NpyIter_Deallocate(the_iter);
  }

  Py_XDECREF(images_seq);
  Py_XDECREF(masks_seq);

  if (zeros == Py_None || zeros == NULL)
    PyMem_Free(zbuffer);

  Py_XDECREF(zeros_arr);

  if (scales == Py_None || scales == NULL)
    PyMem_Free(sbuffer);

  Py_XDECREF(scales_arr);

  if (weights == Py_None || scales == NULL)
    PyMem_Free(wbuffer);

  Py_XDECREF(weights_arr);

  return PyErr_Occurred() ? NULL : Py_BuildValue("(N,N,N)",out_arr[0], out_arr[1], out_arr[2]);
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

int NU_debug_function(double *data, double *weights,
    size_t size, double *out[3], void *func_data)
{
  *out[2] = 0;
  *out[0] = 0;
  *out[1] = 0;

  return 1;
}

static PyObject *
py_method_debug(PyObject *obj, PyObject *args) {

  return PyCapsule_New((void*)NU_mean_function, "numina.cmethod", NULL);
}


static PyMethodDef module_functions[] = {
    {"generic_combine", (PyCFunction) py_generic_combine_miter, METH_VARARGS | METH_KEYWORDS, "generic combine function"},
    {"generic_combine_1iter", (PyCFunction) py_generic_combine, METH_VARARGS | METH_KEYWORDS, "generic combine function"},
    {"generic_combine_miter", (PyCFunction) py_generic_combine_miter, METH_VARARGS | METH_KEYWORDS, "generic combine function with multiple iters"},
    {"mean_method", py_method_mean, METH_NOARGS, ""},
    {"median_method", py_method_median, METH_NOARGS, ""},
    {"minmax_method", py_method_minmax, METH_VARARGS, ""},
    {"sigmaclip_method", py_method_sigmaclip, METH_VARARGS, ""},
    {"quantileclip_method", py_method_quantileclip, METH_VARARGS, ""},
    {"debug_method", py_method_debug, METH_VARARGS, ""},
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
   PyObject *mod;
   mod = PyModule_Create(&moduledef);
   if (mod == NULL)
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
   PyModule_AddObject(mod, "CombineError", CombineError);
   return mod;
}
