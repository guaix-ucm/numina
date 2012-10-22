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

#include <vector>
#include <numeric>

#include <numpy/arrayobject.h>
#include <iostream>
#include "nu_ramp.h"

#define MASK_GOOD 0
#define MASK_SATURATION 3

typedef void (*LoopFunc)(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, double dt, double gain, double ron, double nsig);

template<typename Result, typename Arg>
static void py_ramp_loop(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, double dt, double gain, double ron, double nsig){
        npy_intp count = *innersizeptr;
        int blank = 0; // this will be an argument in the future
        static std::vector<Arg> internal;

        // The stride for 2-6 pointers is 0, so no advance is needed
        // they are always the same
        Result* rvalue = reinterpret_cast<Result*>(dataptr[2]);
        Result* rvariance = reinterpret_cast<Result*>(dataptr[3]);
        *rvalue = Result(blank);
        *rvariance = Result(blank);

        if(*dataptr[1] != MASK_GOOD) {
          // mask is copied from badpixels
          // all the rest are 0
          *dataptr[5] = *dataptr[1];
        }
        else {
          while (count--) {
            // Recovering values
            Arg value = *((Arg*)dataptr[0]);

            if(value < saturation)
               internal.push_back(value);
            else {
              // Stop as soon as you encounter the first
              // saturated value
              break;
            }
            // Advance pointers
            // to get all values in the ramp
            for(int ui=0; ui < size; ++ui)
              dataptr[ui] += strideptr[ui];
          }

          // we don't have enough non saturated points
          if(internal.size() <= 1) {
            *dataptr[5] = MASK_SATURATION;
          }
          else {
            Numina::RampResult<Result> result = Numina::ramp<Result>(internal.begin(), internal.end(),
                 dt, gain, ron, nsig);
            *rvalue = result.value;
            std::cout << result.value << std::endl;
            *rvariance = result.variance;
            *dataptr[4] = result.map; 
            *dataptr[5] = result.mask;
            *dataptr[6] = result.crmask;
          }
          internal.clear();
        }
}

// In numpy private API
static int _zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(ret->descr)) {
        PyObject *zero = PyInt_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        npy_intp n = PyArray_NBYTES(ret);
        memset(ret->data, 0, n);
    }
    return 0;
}


static PyObject* py_loopover(PyObject *self, PyObject *args, PyObject *kwds)
{

  PyArrayObject* inp = NULL;
  PyArrayObject* badpixels = NULL;

  PyArrayObject* value = NULL;
  PyArrayObject* var = NULL;
  PyArrayObject* nmap = NULL; // uint8
  PyArrayObject* mask = NULL; // uint8
  PyArrayObject* crmask = NULL; // uint8

  PyObject* ret = NULL; // A five tuple: (value, var, nmap, mask, crmask)

  double ron = 0.0;
  double gain = 1.0;
  double nsig = 4.0;
  double dt = 1.0;
  int saturation = 65631;
  int blank = 0;

  npy_intp out_dims[2];
  int ui = 0;
  int valid = 1;

  LoopFunc loopfunc = NULL;

  NpyIter_IterNextFunc *iternext = NULL;
  char** dataptr = NULL;
  npy_intp* strideptr = NULL;
  npy_intp* innersizeptr = NULL;

  char *kwlist[] = {"inpt", "dt", "gain", "ron", "badpixels", "out",
      "saturation", "nsig", "blank", NULL};

  const int func_nloops = 10;
  const char func_sigs[] = {'d', 'f', 'q', 'i', 'h', 'b', 'Q', 'I', 'H', 'B'};
  const LoopFunc func_loops[] = {
      py_ramp_loop<npy_float64, npy_float64>,
      py_ramp_loop<npy_float32, npy_float32>,
      py_ramp_loop<npy_int64, npy_int64>,
      py_ramp_loop<npy_int32, npy_int32>,
      py_ramp_loop<npy_int16, npy_int16>,
      py_ramp_loop<npy_int8, npy_int8>,
      py_ramp_loop<npy_uint64, npy_uint64>,
      py_ramp_loop<npy_uint32, npy_uint32>,
      py_ramp_loop<npy_uint16, npy_uint16>,
      py_ramp_loop<npy_uint8, npy_uint8>,
  };

  NpyIter* iter;
  PyArrayObject* arrs[7];
  npy_uint32 whole_flags = NPY_ITER_EXTERNAL_LOOP|NPY_ITER_BUFFERED|NPY_ITER_REDUCE_OK|NPY_ITER_DELAY_BUFALLOC;
  npy_uint32 op_flags[7];

  NPY_ORDER order = NPY_ANYORDER;
  NPY_CASTING casting = NPY_UNSAFE_CASTING;

  PyArray_Descr* dtypes[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  PyArray_Descr* common = NULL;

  int oa_ndim = 3; /* # iteration axes */
  int op_axes1[] = {0, 1, -1};
  int* op_axes[] = {NULL, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1};

  if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&ddd|O&O&Oidi:loopover_ramp_c", kwlist,
        &PyArray_Converter, &inp,
        &dt, &gain, &ron,
        &PyArray_Converter, &badpixels,
        &PyArray_OutputConverter, &value,
        &saturation, &nsig, &blank)
        )
    return NULL;

  if (badpixels == NULL) {
    out_dims[0] = PyArray_DIM(inp, 0);
    out_dims[1] = PyArray_DIM(inp, 1);
    badpixels = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_UINT8, 0);
  }


  if (gain <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, gain <= 0");
    goto exit;
  }
  if (ron < 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, ron < 0");
    goto exit;
  }
  if (nsig <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, nsig <= 0");
    goto exit;
  }
  if (dt <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, dt <= 0");
    goto exit;
  }
  if (saturation <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, saturation <= 0");
    goto exit;
  }

  op_flags[0] = NPY_ITER_READONLY;
  op_flags[1] = NPY_ITER_READONLY | NPY_ITER_NO_BROADCAST;
  for(ui=2; ui <= 6; ++ui)
    op_flags[ui] = NPY_ITER_READWRITE | NPY_ITER_ALLOCATE;

  // Using arrs as a temporary
  arrs[0] = inp;
  if(value != NULL)
   valid = 2;
  arrs[1] = value;

  common = PyArray_ResultType(valid, arrs, 0, NULL);

  for(ui = 0; ui < func_nloops; ++ui) {
    if (common->type == func_sigs[ui]) {
      loopfunc = func_loops[ui];
      break;
    }
  }
  if (loopfunc == NULL) {
    // FIXME
    printf("No registered loopfunc\n");
    Py_DECREF(common);
    goto exit;
  }

  // Filling arrs with all the arrays
  arrs[1] = badpixels;
  arrs[2] = value;
  // The following are dynamically allocated always
  arrs[3] = var;
  arrs[4] = nmap;
  arrs[5] = mask;
  arrs[6] = crmask;

  dtypes[0] = PyArray_DescrFromType(common->type); // input
  dtypes[1] = PyArray_DescrFromType(NPY_UINT8); // badpixels
  dtypes[2] = PyArray_DescrFromType(common->type); // value
  dtypes[3] = PyArray_DescrFromType(common->type); // variance
  dtypes[4] = PyArray_DescrFromType(NPY_UINT8); // number of pixels
  dtypes[5] = PyArray_DescrFromType(NPY_UINT8); // new mask of bad pixels
  dtypes[6] = PyArray_DescrFromType(NPY_UINT8); // new mask of cosmic rays

  iter = NpyIter_AdvancedNew(7, arrs, whole_flags, order, casting,
                              op_flags, dtypes, oa_ndim, op_axes,
                              NULL, 0);
  if (iter == NULL)
    goto exit;

  // FIXME: check this return value
  NpyIter_Reset(iter, NULL);

  // Filling all with 0
  for(ui=2; ui<7; ++ui)
    _zerofill(NpyIter_GetOperandArray(iter)[ui]);

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);
  strideptr = NpyIter_GetInnerStrideArray(iter);
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  do {
       loopfunc(7, dataptr, strideptr, innersizeptr, saturation, dt, gain, ron, nsig);
    } while(iternext(iter));


  // the result is a 5-tuple
  ret = Py_BuildValue("(O,O,O,O,O)",
      (PyObject*)NpyIter_GetOperandArray(iter)[2],
      (PyObject*)NpyIter_GetOperandArray(iter)[3],
      (PyObject*)NpyIter_GetOperandArray(iter)[4],
      (PyObject*)NpyIter_GetOperandArray(iter)[5],
      (PyObject*)NpyIter_GetOperandArray(iter)[6]
  );

  NpyIter_Deallocate(iter);

exit:
  // Using PyArray_Converter requires DECREF
  Py_XDECREF(inp);
  Py_XDECREF(badpixels);
  //
  Py_XDECREF(common);
  for(ui=0; ui<7; ++ui)
    Py_XDECREF(dtypes[ui]);

  return ret;
}

static PyMethodDef ramp_methods[] = {
    {"process_ramp_c", (PyCFunction) py_loopover, METH_VARARGS|METH_KEYWORDS, "Follow-up-the-ramp processing"},
    {"loopover_ramp_c", (PyCFunction) py_loopover, METH_VARARGS|METH_KEYWORDS, "Follow-up-the-ramp processing"},
    { NULL, NULL, 0, NULL } /* sentinel */
};

PyMODINIT_FUNC init_ramp(void)
{
  PyObject *m;
  m = Py_InitModule("_ramp", ramp_methods);
  import_array();

  if (m == NULL)
    return;
}
