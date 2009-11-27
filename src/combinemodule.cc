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

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#include <numpy/arrayobject.h>

#include "numpytypes.h"
#include "method_factory.h"
#include "method_exception.h"
#include "methods_python.h"

#define CASE_MASKS_FIXED(DN, MN) \
	case MN: \
		selection_ptr = &Array_select<numpy::fixed_type<DN>::data_type,  \
									  numpy::fixed_type<MN>::data_type>; \
		break;

#define CASE_MASKS_GENERIC(DN, MN) \
	case MN: \
		selection_ptr = &Array_select<numpy::generic_type<DN>::data_type,  \
									  numpy::generic_type<MN>::data_type>; \
		break;

#define CASE_DATA_FIXED(DN) \
	case DN: \
	switch (masks_type) { \
		CASE_MASKS_FIXED(DN, NPY_BOOL) \
		CASE_MASKS_FIXED(DN, NPY_UINT8) \
		CASE_MASKS_FIXED(DN, NPY_INT16) \
		CASE_MASKS_FIXED(DN, NPY_INT32) \
		CASE_MASKS_FIXED(DN, NPY_FLOAT32) \
		CASE_MASKS_FIXED(DN, NPY_FLOAT64) \
	default: \
		return PyErr_Format(CombineError, "mask type %d not implemented", masks_type); \
		break; \
	} \
	break;

#define DATA_CASE_GENERIC(DN) \
	case DN: \
	switch (masks_type) { \
		CASE_MASKS_GENERIC(DN, NPY_BOOL) \
		CASE_MASKS_GENERIC(DN, NPY_SHORT) \
		CASE_MASKS_GENERIC(DN, NPY_USHORT) \
		CASE_MASKS_GENERIC(DN, NPY_INT) \
		CASE_MASKS_GENERIC(DN, NPY_UINT) \
		CASE_MASKS_GENERIC(DN, NPY_LONG) \
		CASE_MASKS_GENERIC(DN, NPY_ULONG) \
		CASE_MASKS_GENERIC(DN, NPY_FLOAT) \
		CASE_MASKS_GENERIC(DN, NPY_DOUBLE) \
		CASE_MASKS_GENERIC(DN, NPY_LONGDOUBLE) \
	default: \
		return PyErr_Format(CombineError, "mask type %d not implemented", masks_type); \
		break; \
	} \
	break;

using namespace Numina;

typedef void
(*SelectionMethodPtr)(size_t nimages,
		const std::vector<PyArrayIterObject*>& iiter, const std::vector<
				PyArrayIterObject*>& miter, std::vector<double>& data);

PyDoc_STRVAR(combine__doc__, "Internal combine module, not to be used directly.");
PyDoc_STRVAR(internal_combine__doc__, "Combines identically shaped images");

static PyObject* CombineError;

template<typename ImageType, typename MaskType>
static void Array_select(size_t nimages,
		const std::vector<PyArrayIterObject*>& iiter, const std::vector<
				PyArrayIterObject*>& miter, std::vector<double>& data) {
	for (size_t i = 0; i < nimages; ++i) {
		MaskType *pmask = (MaskType*) miter[i]->dataptr;
		if (not *pmask) // <- True values are skipped
			continue;
		ImageType* pdata = (ImageType*) iiter[i]->dataptr;
		data.push_back(*pdata);
	}
}

static PyObject* py_internal_combine(PyObject *self, PyObject *args,
		PyObject *kwds) {

	// Output has one dimension more than the inputs, of size
	// OUTDIM
	const size_t OUTDIM = 3;
	PyObject *method;
	PyObject *images = NULL;
	PyObject *masks = NULL;

	// Offsets are ignored
	PyObject *out[OUTDIM] = { NULL, NULL, NULL };
	PyObject* margs = NULL;

	static char *kwlist[] = { "method", "data", "masks", "out0", "out1",
			"out2", "args", NULL };

	int ok = PyArg_ParseTupleAndKeywords(args, kwds,
			"OO!O!O!O!O!O!:internal_combine", kwlist, &method, &PyList_Type,
			&images, &PyList_Type, &masks, &PyArray_Type, &out[0],
			&PyArray_Type, &out[1], &PyArray_Type, &out[2], &PyTuple_Type,
			&margs);

	if (!ok) {
		return NULL;
	}

	// Method class
	std::auto_ptr<Numina::Method> method_ptr;

	// If method is a string
	if (PyString_Check(method)) {

		// We have a string
		const char* method_name = PyString_AS_STRING(method);

		try {
			method_ptr.reset(NamedMethodFactory::create(method_name, margs));
		} catch (MethodException&) {
			return PyErr_Format(
					CombineError,
					"error during the construction of the combination method \"%s\"",
					method_name);
		}

		if (not method_ptr.get()) {
			return PyErr_Format(CombineError,
					"invalid combination method \"%s\"", method_name);
		}

	} // If method is callable
	else if (PyCallable_Check(method)) {
		method_ptr.reset(new PythonMethod(method, 0));
	} else
		return PyErr_Format(PyExc_TypeError,
				"method is neither a string nor callable");

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

	// Select the function we are going to use

	SelectionMethodPtr selection_ptr;

	switch (images_type) {
	CASE_DATA_FIXED(NPY_BOOL)
	CASE_DATA_FIXED(NPY_FLOAT64)
	CASE_DATA_FIXED(NPY_FLOAT32)
	CASE_DATA_FIXED(NPY_INT32)
	CASE_DATA_FIXED(NPY_INT16)
	CASE_DATA_FIXED(NPY_UINT8)
	default:
		// Data type not implemented
		return PyErr_Format(CombineError, "image type %d not implemented",
				images_type);
		break;
	}

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
	for (size_t i = 0; i < OUTDIM; ++i)
		oiter[i] = (PyArrayIterObject*) PyArray_IterNew(out[i]);

	// basic iterator
	PyArrayIterObject* iter = oiter[0];

	std::vector<double> data;
	data.reserve(nimages);
	double* values[OUTDIM];

	while (iter->index < iter->size) {

		selection_ptr(nimages, iiter, miter, data);

		for (size_t i = 0; i < OUTDIM; ++i)
			values[i] = (npy_double*) oiter[i]->dataptr;

		method_ptr->run(&data[0], data.size(), values);

		data.clear();

		for (size_t i = 0; i < nimages; ++i) {
			PyArray_ITER_NEXT(iiter[i]);
			PyArray_ITER_NEXT(miter[i]);
		}

		for (size_t i = 0; i < OUTDIM; ++i) {
			PyArray_ITER_NEXT(oiter[i]);
		}
	}

	Py_INCREF( Py_None);
	return Py_None;
}

static PyMethodDef combine_methods[] = { { "internal_combine",
		(PyCFunction) py_internal_combine, METH_VARARGS | METH_KEYWORDS,
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

