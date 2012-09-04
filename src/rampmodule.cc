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

typedef void (*LoopFunc)(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr, int saturation, double dt, double gain, double ron, double nsig);

template<typename Result, typename Arg>
static void py_ramp_loop(int size, char** dataptr, npy_intp* strideptr, npy_intp* innersizeptr, int saturation, double dt, double gain, double ron, double nsig){
        npy_intp count = *innersizeptr;
        static std::vector<Arg> internal;
        while (count--) {
            internal.push_back(*((Arg*)dataptr[0]));
            for(int ui=0; ui < size; ++ui) 
              dataptr[ui] += strideptr[ui];
        }
        Numina::ramp<Result>(internal.begin(), internal.end(), dataptr[1], saturation, dt, gain, ron, nsig);
        internal.clear();
}

static PyObject* py_ramp(PyObject *self, PyObject *args, PyObject *kwds)
{

  PyArrayObject* inp = NULL;
  PyArrayObject* out = NULL;
  PyObject* ret = NULL;
  int axis = 2;
  double ron = 0.0;
  double gain = 1.0;
  double nsig = 4.0;
  double dt = 1.0;
  int saturation = 65631;
  int ui = 0;
  int valid = 1;
  LoopFunc loopfunc = NULL;

  NpyIter_IterNextFunc *iternext = NULL;
  char** dataptr = NULL;
  npy_intp* strideptr = NULL;
  npy_intp* innersizeptr = NULL;

  char *kwlist[] = {"inp", "out", "axis", "ron", "gain", 
                            "nsig", "dt", "saturation", NULL};

  const int func_nloops = 10;
  const char func_sigs[] = {'d', 'f', 'q', 'i', 'h', 'b', 'Q', 'I', 'H', 'B'};
  const void* func_loops[] = {
     (void*) py_ramp_loop<npy_float64, npy_float64>,
     (void*) py_ramp_loop<npy_float32, npy_float32>,
     (void*) py_ramp_loop<npy_int64, npy_int64>,
     (void*) py_ramp_loop<npy_int32, npy_int32>,
     (void*) py_ramp_loop<npy_int16, npy_int16>,
     (void*) py_ramp_loop<npy_int8, npy_int8>,
     (void*) py_ramp_loop<npy_uint64, npy_uint64>,
     (void*) py_ramp_loop<npy_uint32, npy_uint32>,
     (void*) py_ramp_loop<npy_uint16, npy_uint16>,
     (void*) py_ramp_loop<npy_uint8, npy_uint8>,
  };

  NpyIter* iter;
  PyArrayObject* arrs[2];
  npy_uint32 whole_flags = NPY_ITER_EXTERNAL_LOOP|NPY_ITER_BUFFERED|NPY_ITER_REDUCE_OK|NPY_ITER_DELAY_BUFALLOC;
  npy_uint32 op_flags[2];
  NPY_ORDER order = NPY_KEEPORDER;
  NPY_CASTING casting = NPY_SAFE_CASTING;
  PyArray_Descr* dtypes[2] = {NULL, NULL};
  PyArray_Descr* common = NULL;

  int oa_ndim = 3; /* # iteration axes */
  int op_axes1[] = {0, 1, -1};   
  int* op_axes[] = {NULL, op_axes1};

  if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&ddddi:process_ramp_c", 
        kwlist, &PyArray_Converter, &inp, 
        &PyArray_OutputConverter, &out,
        &PyArray_AxisConverter, &axis,
        &ron, &gain, &nsig, &dt, &saturation)
        )
    return NULL;

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
  op_flags[1] = NPY_ITER_READWRITE | NPY_ITER_ALLOCATE;

  arrs[0] = inp;

  if(out != NULL)
   valid = 2;

  arrs[1] = out;
  common = PyArray_ResultType(valid, arrs, 0, NULL);
  
  for(ui = 0; ui < func_nloops; ++ui) {
    if (common->type == func_sigs[ui]) {
      loopfunc = (LoopFunc) func_loops[ui];
      break;
    }
  }
  if (loopfunc == NULL) {
    // FIXME
    printf("No registered loopfunc\n");
    Py_DECREF(common);
    goto exit;
  }


  dtypes[0] = PyArray_DescrFromType(common->type);
  dtypes[1] = PyArray_DescrFromType(common->type);

  iter = NpyIter_AdvancedNew(2, arrs, whole_flags, order, casting,
                              op_flags, dtypes, oa_ndim, op_axes, 
                              NULL, 0);
  if (iter == NULL)
    goto exit;
  
  // FIXME: check this return value
  NpyIter_Reset(iter, NULL);

  iternext = NpyIter_GetIterNext(iter, NULL);
  dataptr = NpyIter_GetDataPtrArray(iter);
  strideptr = NpyIter_GetInnerStrideArray(iter);
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  do {
       loopfunc(2, dataptr, strideptr, innersizeptr, saturation, dt, gain, ron, nsig);
    } while(iternext(iter));

  ret = (PyObject*)NpyIter_GetOperandArray(iter)[1];
  Py_INCREF(ret);
  NpyIter_Deallocate(iter);
exit:
  Py_XDECREF(inp);
  Py_XDECREF(common);
  Py_XDECREF(dtypes[0]);
  Py_XDECREF(dtypes[1]);
  return ret;
}

static PyMethodDef ramp_methods[] = {
    {"process_ramp_c", (PyCFunction) py_ramp, METH_VARARGS|METH_KEYWORDS, "Follow-up-the-ramp processing"},
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
