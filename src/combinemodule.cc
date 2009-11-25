/*
 * Copyright 2008-2009 Sergio Pascual
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

#include <map>
#include <string>
#include <vector>
#include <iostream>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "numpytypes.h"
#include "methods.h"

#define MASKS_CASE_FIXED(DN, MN) \
	case MN: \
		Array_combine<numpy::fixed_type<DN>::data_type, \
		              numpy::fixed_type<MN>::data_type> \
		              (method_ptr, iarr, marr, out1, out2, out3); \
		break;

#define DATA_CASE_FIXED(DN) \
	case DN: \
	switch (masks_type) { \
		MASKS_CASE_FIXED(DN, NPY_UINT8) \
		MASKS_CASE_FIXED(DN, NPY_INT16) \
		MASKS_CASE_FIXED(DN, NPY_INT32) \
		MASKS_CASE_FIXED(DN, NPY_FLOAT32) \
		MASKS_CASE_FIXED(DN, NPY_FLOAT64) \
	default: \
		std::cout << "Mask type " <<  masks_type << " not implemented" << std::endl; \
		break; \
	} \
	break;

typedef void (*GenericMethodPtr)(const double* data, size_t size, double* c,
		double* var, double* number, void* params);
typedef std::map<std::string, GenericMethodPtr> MethodMap;

PyDoc_STRVAR(combine__doc__, "Module doc");
PyDoc_STRVAR(internal_combine__doc__, "Combines identically shaped images");

static PyObject *CombineError;

static PyObject* py_internal_combine(PyObject *self, PyObject *args,
		PyObject *keywds) {
	const char *method_name = NULL;
	PyObject *images = NULL;
	PyObject *masks = NULL;
	PyObject *res = NULL;
	PyObject *var = NULL;
	PyObject *num = NULL;

	static char *kwlist[] = { "method", "nimages", "nmasks", "result",
			"variance", "numbers", NULL };

	int ok = PyArg_ParseTupleAndKeywords(args, keywds,
			"sO!O!|OOO:internal_combine", kwlist, &method_name, &PyList_Type,
			&images, &PyList_Type, &masks, &res, &var, &num);
	if (!ok)
		return NULL;

	// TODO: this is not efficient, it's constructed
	// each time the function is run
	MethodMap methods;
	methods["mean"] = method_mean;

	/* Check if method is registered in our table */
	GenericMethodPtr method_ptr = NULL;
	MethodMap::iterator el = methods.find(method_name);
	if (el != methods.end()) {
		method_ptr = el->second;
	}

	if (!method_ptr) {
		PyErr_Format(PyExc_TypeError, "invalid combination method %s",
				method_name);
		return NULL;
	}

	/* images are forced to be a list */
	const Py_ssize_t nimages = PyList_GET_SIZE(images);

	/* getting the contents */
	std::vector<PyObject*> iarr(nimages);

	for (Py_ssize_t i = 0; i < nimages; i++) {
		PyObject *item = PyList_GetItem(images, i);
		/* To be sure is double */
		iarr[i] = PyArray_FROM_OT(item, NPY_DOUBLE);
		/* We don't need item anymore */
		Py_DECREF(item);
	}

	/* getting the contents */
	std::vector<PyObject*> marr(nimages);

	for (Py_ssize_t i = 0; i < nimages; i++) {
		PyObject *item = PyList_GetItem(masks, i);
		/* To be sure is bool */
		marr[i] = PyArray_FROM_OT(item, NPY_BOOL);
		/* We don't need item anymore */
		Py_DECREF(item);
	}

	/*
	 * This is ok if we are passing the data to a C function
	 * but, as we are creating here a PyList, perhaps it's better
	 * to build the PyList with PyObjects and make the conversion to doubles
	 * inside the final function only
	 */
	std::vector<npy_double> data;
	data.reserve(nimages);
	npy_intp* dims = PyArray_DIMS(iarr[0]);

	for (npy_intp ii = 0; ii < dims[0]; ++ii)
		for (npy_intp jj = 0; jj < dims[1]; ++jj) {

			size_t used = 0;
			/* Collect the valid values */
			for (Py_ssize_t i = 0; i < nimages; ++i) {
				npy_bool *pmask = (npy_bool*) PyArray_GETPTR2(marr[i], ii, jj);
				if (*pmask == NPY_TRUE) // <- True values are skipped
					continue;

				npy_double *pdata = static_cast<double*> (PyArray_GETPTR2(
						iarr[i], ii, jj));
				data.push_back(*pdata);
				++used;
			}

			npy_double* p = (npy_double*) PyArray_GETPTR2(res, ii, jj);
			npy_double* v = (npy_double*) PyArray_GETPTR2(var, ii, jj);
			long* n = (long*) PyArray_GETPTR2(num, ii, jj);
			double nn;
			/* Compute the results*/
			void *params = NULL;
			method_ptr(&data[0], used, p, v, &nn, params);
			*n = (long) nn;
			data.clear();

		}
	Py_INCREF( Py_None);
	return Py_None;
}

template<typename ImageType, typename MaskType>
static void Array_combine(GenericMethodPtr method_ptr, const std::vector<
		PyObject*>& iarr, const std::vector<PyObject*>& marr, PyObject* out1,
		PyObject* out2, PyObject* out3) {

	const size_t nimages = iarr.size();

	std::vector<double> data;
	data.reserve(nimages);

	npy_intp* dims = PyArray_DIMS(iarr[0]);

	// Iterators
	std::vector<PyArrayIterObject*> iiter(nimages);
	for (size_t i = 0; i < nimages; ++i) {
		iiter[i] = (PyArrayIterObject*) PyArray_IterNew(iarr[i]);
	}

	std::vector<PyArrayIterObject*> miter(nimages);
	for (size_t i = 0; i < nimages; ++i) {
		miter[i] = (PyArrayIterObject*) PyArray_IterNew(marr[i]);
	}

	std::vector<PyArrayIterObject*> oiter(3);
	oiter[0] = (PyArrayIterObject*) PyArray_IterNew(out1);
	oiter[1] = (PyArrayIterObject*) PyArray_IterNew(out2);
	oiter[2] = (PyArrayIterObject*) PyArray_IterNew(out3);

	// basic iter
	PyArrayIterObject* iter = oiter[0];

	while (iter->index < iter->size) {
		/* do something with the data at it->dataptr */
		size_t used = 0;

		for (size_t i = 0; i < nimages; ++i) {
			MaskType *pmask = (MaskType*) miter[i]->dataptr;
			if (*pmask) // <- True values are skipped
				continue;
			ImageType* pdata = (ImageType*) iiter[i]->dataptr;
			data.push_back(*pdata);
			++used;
		}

		npy_double* p = (npy_double*) oiter[0]->dataptr;
		npy_double* v = (npy_double*) oiter[1]->dataptr;
		npy_double* n = (npy_double*) oiter[2]->dataptr;

		/* Compute the results*/
		void *params = NULL;
		method_ptr(&data[0], used, p, v, n, params);

		data.clear();

		for (size_t i = 0; i < nimages; ++i) {
			PyArray_ITER_NEXT(iiter[i]);
			PyArray_ITER_NEXT(miter[i]);
		}

		for (size_t i = 0; i < 3; ++i) {
			PyArray_ITER_NEXT(oiter[i]);
		}
	}
}

static PyObject* py_internal_combine2(PyObject *self, PyObject *args,
		PyObject *kwds) {
	const char *method_name = NULL;

	PyObject *images = NULL;
	PyObject *masks = NULL;

	// Offsets are ignored
	PyObject *offsets = NULL;
	PyObject *out1 = NULL;
	PyObject *out2 = NULL;
	PyObject *out3 = NULL;

	static char *kwlist[] = { "method", "data", "masks", "offsets", "out1",
			"out2", "out3", NULL };

	int ok = PyArg_ParseTupleAndKeywords(args, kwds,
			"sO!O!O!O!O!O!:internal_combine2", kwlist, &method_name,
			&PyList_Type, &images, &PyList_Type, &masks, &PyArray_Type,
			&offsets, &PyArray_Type, &out1, &PyArray_Type, &out2,
			&PyArray_Type, &out3);

	if (!ok) {
		return NULL;
	}

	// TODO: this is not efficient, it's constructed
	// each time the function is run
	MethodMap methods;
	methods["mean"] = method_mean;

	/* Check if method is registered in our table */
	GenericMethodPtr method_ptr = NULL;
	MethodMap::iterator el = methods.find(method_name);
	if (el != methods.end()) {
		method_ptr = el->second;
	}

	if (!method_ptr) {
		PyErr_Format(PyExc_TypeError, "invalid combination method %s",
				method_name);
		return NULL;
	}

	/* images are forced to be a list */
	const Py_ssize_t nimages = PyList_GET_SIZE(images);

	/* getting the contents */
	std::vector<PyObject*> iarr(nimages);

	// Images
	iarr[0] = PyList_GetItem(images, 0);
	const int images_type = PyArray_TYPE(iarr[0]);

	for (Py_ssize_t i = 1; i < nimages; i++) {
		// Borrowed reference, no decref
		PyObject *item = PyList_GetItem(images, i);
		if (PyArray_Check(item)) {
			iarr[i] = item;
		}
	}

	// Masks
	std::vector<PyObject*> marr(nimages);
	marr[0] = PyList_GetItem(masks, 0);
	const int masks_type = PyArray_TYPE(marr[0]);

	for (Py_ssize_t i = 1; i < nimages; i++) {
		// Borrowed reference, no decref
		PyObject *item = PyList_GetItem(masks, i);
		if (PyArray_Check(item)) {
			marr[i] = item;
		}
	}

	switch (images_type) {
	DATA_CASE_FIXED(NPY_FLOAT64)
	DATA_CASE_FIXED(NPY_FLOAT32)
	DATA_CASE_FIXED(NPY_INT32)
	DATA_CASE_FIXED(NPY_INT16)
	DATA_CASE_FIXED(NPY_UINT8)
	default:
		// Not implemented
		std::cout << "Image type " << images_type << " not implemented"
				<< std::endl;
		break;
	}

	Py_INCREF( Py_None);
	return Py_None;
}

static PyMethodDef combine_methods[] = { { "internal_combine",
		(PyCFunction) py_internal_combine, METH_VARARGS | METH_KEYWORDS,
		internal_combine__doc__ }, { "internal_combine2",
		(PyCFunction) py_internal_combine2, METH_VARARGS | METH_KEYWORDS,
		internal_combine__doc__ }, { NULL, NULL, 0, NULL } /* sentinel */
};

PyMODINIT_FUNC init_combine(void) {
	PyObject *m;
	m = Py_InitModule3("_combine", combine_methods, combine__doc__);
	import_array();

	if (m == NULL)
		return;

	if (CombineError == NULL) {
		/*
		 * A different base class can be used as base of the exception
		 * passing something instead of NULL
		 */
		CombineError = PyErr_NewException("_combine.CombineError", NULL, NULL);
	}
	Py_INCREF(CombineError);
	PyModule_AddObject(m, "CombineError", CombineError);
}

