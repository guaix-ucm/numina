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
#include "nu_ramp.h"
#include "nu_fowler.h"

#define MASK_GOOD 0
#define MASK_SATURATION 3

typedef void (*LoopFuncRamp)(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, int blank, double dt, double gain, double ron, double nsig);

typedef void (*LoopFuncFowler)(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, int blank);


template<typename Result, typename Arg>
static void py_ramp_loop(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, int blank, double dt, double gain, double ron, double nsig){
        npy_intp count = *innersizeptr;
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

            // FIXME: saturation is int
            // value may be unsigned...
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
            *rvariance = result.variance;
            *dataptr[4] = result.map; 
            *dataptr[5] = result.mask;
            *dataptr[6] = result.crmask;
          }
          internal.clear();
        }
}

template<typename Result, typename Arg>
static void py_fowler_loop(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr,
    int saturation, int blank){
        npy_intp count = *innersizeptr;
        // count is even
        npy_intp hsize = count / 2;
        static std::vector<Arg> internal;

        // The stride for 2-6 pointers is 0, so no advance is needed
        // they are always the same
        Result* rvalue = reinterpret_cast<Result*>(dataptr[2]);
        Result* rvariance = reinterpret_cast<Result*>(dataptr[3]);
        *rvalue = Result(blank);
        *rvariance = Result(blank);

        if(*dataptr[1] != MASK_GOOD) {
          // mask is copied from badpixels
          // all the rest are blank
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
            // to get all values in the Fowler exposure
            for(int ui=0; ui < size; ++ui)
              dataptr[ui] += strideptr[ui];
          }

          // we don't have enough non saturated points
          if(internal.size() <= hsize) {
            *dataptr[5] = MASK_SATURATION;
          }
          else {
            Numina::FowlerResult<Result> result =
                Numina::fowler<Result>(internal.begin(), internal.end(), hsize);

            *rvalue = result.value;
            *rvariance = result.variance;
            *dataptr[4] = result.map; 
            *dataptr[5] = result.mask;
          }
          internal.clear();
        }
}

// In numpy private API
static int _zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(ret->descr)) {
#if PY_MAJOR_VERSION >= 3
        PyObject *zero = PyLong_FromLong(0);
#else
        PyObject *zero = PyInt_FromLong(0);
#endif
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


static PyObject* py_ramp_array(PyObject *self, PyObject *args, PyObject *kwds)
{

  PyArrayObject* inp = NULL;
  PyArrayObject* badpixels = NULL;

  PyArrayObject* value = NULL;
  PyArrayObject* var = NULL;
  PyArrayObject* nmap = NULL; // uint8
  PyArrayObject* mask = NULL; // uint8
  PyArrayObject* crmask = NULL; // uint8
  PyArray_Descr* outtype = NULL; // default is float64

  PyObject* ret = NULL; // A 5-tuple: (value, var, nmap, mask, crmask)

  const int NOPS = 7; // 2+5
  double ron = 0.0;
  double gain = 1.0;
  double nsig = 4.0;
  double dt = 1.0;
  int saturation = 65631;
  int blank = 0;

  npy_intp out_dims[2];
  int ui = 0;

  LoopFuncRamp loopfunc = NULL;

  NpyIter_IterNextFunc *iternext = NULL;
  char** dataptr = NULL;
  npy_intp* strideptr = NULL;
  npy_intp* innersizeptr = NULL;

  const int FUNC_NLOOPS = 11;
  const char func_sigs[] = {'g', 'd', 'f', 'l', 'i', 'h', 'b', 'L', 'I', 'H', 'B'};
  const LoopFuncRamp func_loops[] = {
      py_ramp_loop<npy_float128, npy_float128>,
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
  PyArrayObject* arrs[NOPS];
  npy_uint32 whole_flags = NPY_ITER_EXTERNAL_LOOP|NPY_ITER_BUFFERED|NPY_ITER_REDUCE_OK|NPY_ITER_DELAY_BUFALLOC;
  npy_uint32 op_flags[NOPS];

  NPY_ORDER order = NPY_ANYORDER;
  NPY_CASTING casting = NPY_UNSAFE_CASTING;

  PyArray_Descr* dtypes[NOPS] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  PyArray_Descr* common = NULL;

  int oa_ndim = 3; /* # iteration axes */
  int op_axes1[] = {0, 1, -1};
  int* op_axes[] = {NULL, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1};

  char *kwlist[] = {"rampdata", "dt", "gain", "ron", "badpixels", "dtype",
      "saturation", "nsig", "blank", NULL};

  if(!PyArg_ParseTupleAndKeywords(args, kwds, 
        "O&ddd|O&O&idi:ramp_array_c", kwlist,
        &PyArray_Converter, &inp,
        &dt, &gain, &ron,
        &PyArray_Converter, &badpixels,
        &PyArray_DescrConverter2, &outtype,
        &saturation, &nsig, &blank)
        )
    return NULL;

  if (badpixels == NULL) {
    out_dims[0] = PyArray_DIM(inp, 0);
    out_dims[1] = PyArray_DIM(inp, 1);
    badpixels = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_UINT8, 0);
  }

  if (outtype == NULL) {
    // Default dtype is float64
    outtype = PyArray_DescrFromType(NPY_FLOAT64); 
  }

  if (gain <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, gain <= 0.0");
    goto exit;
  }
  if (ron < 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, ron < 0.0");
    goto exit;
  }
  if (nsig <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, nsig <= 0.0");
    goto exit;
  }
  if (dt <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, dt <= 0.0");
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

  // Using arrs and dtypes as a temporary
  arrs[0] = inp;
  dtypes[0] = outtype;
  // Common dtype
  common = PyArray_ResultType(1, arrs, 1, dtypes);

  // Looking for the correct loop
  for(ui = 0; ui < FUNC_NLOOPS; ++ui) {
    if (common->type == func_sigs[ui]) {
      loopfunc = func_loops[ui];
      break;
    }
  }

  if (loopfunc == NULL) {
    PyErr_Format(PyExc_TypeError, "no registered loopfunc '%c'", common->type);
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

  iter = NpyIter_AdvancedNew(NOPS, arrs, whole_flags, order, casting,
                              op_flags, dtypes, oa_ndim, op_axes,
                              NULL, 0);
  if (iter == NULL)
    goto exit;

  if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
    NpyIter_Deallocate(iter);
    goto exit;
  }

  // Filling all with 0
  for(ui=2; ui<NOPS; ++ui)
    _zerofill(NpyIter_GetOperandArray(iter)[ui]);

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);
  strideptr = NpyIter_GetInnerStrideArray(iter);
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  do {
       loopfunc(NOPS, dataptr, strideptr, innersizeptr, saturation, blank,
           dt, gain, ron, nsig);
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
  Py_XDECREF(outtype);
  for(ui=0; ui<NOPS; ++ui)
    Py_XDECREF(dtypes[ui]);

  return ret;
}

static PyObject* py_fowler_array(PyObject *self, PyObject *args, PyObject *kwds)
{

  PyArrayObject* inp = NULL;
  PyArrayObject* badpixels = NULL;

  PyArrayObject* value = NULL;
  PyArrayObject* var = NULL;
  PyArrayObject* nmap = NULL; // uint8
  PyArrayObject* mask = NULL; // uint8
  PyArray_Descr* outtype = NULL; // default is float64

  PyObject* ret = NULL; // A 4-tuple: (value, var, nmap, mask)

  const int NOPS = 6; // (2+4)-tuple
  int saturation = 65631;
  int blank = 0;

  npy_intp out_dims[2];
  int ui = 0;
  int hsize = 0;

  LoopFuncFowler loopfunc = NULL;

  NpyIter_IterNextFunc *iternext = NULL;
  char** dataptr = NULL;
  npy_intp* strideptr = NULL;
  npy_intp* innersizeptr = NULL;

  const int FUNC_NLOOPS = 11;
  const char func_sigs[] = {'g', 'd', 'f', 'l', 'i', 'h', 'b', 'L', 'I', 'H', 'B'};
  const LoopFuncFowler func_loops[] = {
      py_fowler_loop<npy_float128, npy_float128>,
      py_fowler_loop<npy_float64, npy_float64>,
      py_fowler_loop<npy_float32, npy_float32>,
      py_fowler_loop<npy_int64, npy_int64>,
      py_fowler_loop<npy_int32, npy_int32>,
      py_fowler_loop<npy_int16, npy_int16>,
      py_fowler_loop<npy_int8, npy_int8>,
      py_fowler_loop<npy_uint64, npy_uint64>,
      py_fowler_loop<npy_uint32, npy_uint32>,
      py_fowler_loop<npy_uint16, npy_uint16>,
      py_fowler_loop<npy_uint8, npy_uint8>,
  };

  NpyIter* iter;
  PyArrayObject* arrs[NOPS];
  npy_uint32 whole_flags = NPY_ITER_EXTERNAL_LOOP|NPY_ITER_BUFFERED|NPY_ITER_REDUCE_OK|NPY_ITER_DELAY_BUFALLOC;
  npy_uint32 op_flags[NOPS];

  NPY_ORDER order = NPY_ANYORDER;
  NPY_CASTING casting = NPY_UNSAFE_CASTING;

  PyArray_Descr* dtypes[NOPS] = {NULL, NULL, NULL, NULL, NULL, NULL};
  PyArray_Descr* common = NULL;

  int oa_ndim = 3; /* # iteration axes */
  int op_axes1[] = {0, 1, -1};
  int* op_axes[] = {NULL, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1, op_axes1};

  char *kwlist[] = {"fowlerdata", "badpixels", "dtype",
      "saturation", "blank", NULL};

  if(!PyArg_ParseTupleAndKeywords(args, kwds, 
        "O&|O&O&ii:fowler_array_c", kwlist,
        &PyArray_Converter, &inp,
        &PyArray_Converter, &badpixels,
        &PyArray_DescrConverter2, &outtype,
        &saturation, &blank)
        )
    return NULL;

  if (PyArray_NDIM(inp) != 3) {
    PyErr_SetString(PyExc_ValueError, "input array is not 3D");
        goto exit;
  }

  hsize = PyArray_DIM(inp, 2) / 2;

  if (hsize * 2 != PyArray_DIM(inp, 2)) {
     PyErr_SetString(PyExc_ValueError, "axis-2 in fowlerdata must be even");
         goto exit;
  }

  if (badpixels == NULL) {
    out_dims[0] = PyArray_DIM(inp, 0);
    out_dims[1] = PyArray_DIM(inp, 1);
    badpixels = (PyArrayObject*)PyArray_ZEROS(2, out_dims, NPY_UINT8, 0);
  }

  if (outtype == NULL) {
    // Default dtype is float64
    outtype = PyArray_DescrFromType(NPY_FLOAT64); 
  }

  if (saturation <= 0) {
    PyErr_SetString(PyExc_ValueError, "invalid parameter, saturation <= 0");
    goto exit;
  }

  op_flags[0] = NPY_ITER_READONLY;
  op_flags[1] = NPY_ITER_READONLY | NPY_ITER_NO_BROADCAST;
  for(ui=2; ui < NOPS; ++ui)
    op_flags[ui] = NPY_ITER_READWRITE | NPY_ITER_ALLOCATE;

  // Using arrs and dtypes as a temporary
  arrs[0] = inp;
  dtypes[0] = outtype;
  // Common dtype
  common = PyArray_ResultType(1, arrs, 1, dtypes);

  // Looking for the correct loop
  for(ui = 0; ui < FUNC_NLOOPS; ++ui) {
    if (common->type == func_sigs[ui]) {
      loopfunc = func_loops[ui];
      break;
    }
  }

  if (loopfunc == NULL) {
    PyErr_Format(PyExc_TypeError, "no registered loopfunc '%c'", common->type);
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

  dtypes[0] = PyArray_DescrFromType(common->type); // input
  dtypes[1] = PyArray_DescrFromType(NPY_UINT8); // badpixels
  dtypes[2] = PyArray_DescrFromType(common->type); // value
  dtypes[3] = PyArray_DescrFromType(common->type); // variance
  dtypes[4] = PyArray_DescrFromType(NPY_UINT8); // number of pixels
  dtypes[5] = PyArray_DescrFromType(NPY_UINT8); // new mask of bad pixels

  iter = NpyIter_AdvancedNew(NOPS, arrs, whole_flags, order, casting,
                              op_flags, dtypes, oa_ndim, op_axes,
                              NULL, 0);
  if (iter == NULL)
    goto exit;

  if (NpyIter_Reset(iter, NULL) != NPY_SUCCEED) {
    NpyIter_Deallocate(iter);
    goto exit;
  }

  // Filling all with 0
  for(ui=2; ui<NOPS; ++ui)
    _zerofill(NpyIter_GetOperandArray(iter)[ui]);

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);
  strideptr = NpyIter_GetInnerStrideArray(iter);
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  do {
       loopfunc(NOPS, dataptr, strideptr, innersizeptr, saturation, blank);
    } while(iternext(iter));


  // the result is a 4-tuple
  ret = Py_BuildValue("(O,O,O,O)",
      (PyObject*)NpyIter_GetOperandArray(iter)[2],
      (PyObject*)NpyIter_GetOperandArray(iter)[3],
      (PyObject*)NpyIter_GetOperandArray(iter)[4],
      (PyObject*)NpyIter_GetOperandArray(iter)[5]
  );

  NpyIter_Deallocate(iter);

exit:
  // Using PyArray_Converter requires DECREF
  Py_XDECREF(inp);
  Py_XDECREF(badpixels);
  //
  Py_XDECREF(common);
  Py_XDECREF(outtype);
  for(ui=0; ui<NOPS; ++ui)
    Py_XDECREF(dtypes[ui]);

  return ret;
}

static PyMethodDef module_functions[] = {
    {"ramp_array", (PyCFunction) py_ramp_array, METH_VARARGS|METH_KEYWORDS, "Follow-up-the-ramp processing"},
    {"fowler_array", (PyCFunction) py_fowler_array, METH_VARARGS|METH_KEYWORDS, "Fowler processing"},
    { NULL, NULL, 0, NULL } /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_nirproc",     /* m_name */
        "Processing of typical read modes of nIR detectors",  /* m_doc */
        -1,                  /* m_size */
        module_functions,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit__nirproc(void)
  {
   PyObject *m;
   m = PyModule_Create(&moduledef);
   if (m == NULL)
     return NULL;

   import_array();
   return m;
  }
#else
  PyMODINIT_FUNC init_nirproc(void)
  {
   PyObject *m;
   m = Py_InitModule("_nirproc", module_functions);
   import_array();

   if (m == NULL)
     return;
  }
#endif
