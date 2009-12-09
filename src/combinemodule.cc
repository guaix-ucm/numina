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
#include <algorithm>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#include <numpy/arrayobject.h>

#include "method_factory.h"
#include "transformation_factory.h"
#include "method_exception.h"
#include "methods_python.h"

using Numina::NamedMethodFactory;
using Numina::Method;
using Numina::MethodException;
using Numina::PythonMethod;

typedef std::vector<PyArrayIterObject*> VectorPyArrayIter;

PyDoc_STRVAR(combine__doc__, "Internal combine module, not to be used directly.");
PyDoc_STRVAR(internal_combine__doc__, "Combines identically shaped images");

// Convenience function to avoid the PyArray_ITER_NEXT macro
static inline void My_PyArray_Iter_Next(PyArrayIterObject* it) {
	PyArray_ITER_NEXT(it);
}

// Convenience PyArrayIterObject* creator
static inline PyArrayIterObject* My_PyArray_IterNew(PyObject* obj) {
	return (PyArrayIterObject*) PyArray_IterNew(obj);
}

// An exception in this module
static PyObject* CombineError;

static PyObject* py_internal_combine(PyObject *self, PyObject *args,
		PyObject *kwds) {

	// Output has one dimension more than the inputs, of size
	// OUTDIM
	const size_t OUTDIM = 3;
	PyObject *method;
	PyObject *images = NULL;
	PyObject *masks = NULL;

	PyObject *out[OUTDIM] = { NULL, NULL, NULL };
	PyObject* margs = NULL;

	PyArrayObject* scales = NULL;
	PyArrayObject* zeros = NULL;
	PyArrayObject* weights = NULL;

	static char *kwlist[] = { "method", "data", "masks", "out0", "out1",
			"out2", "args", "zeros", "scales", "weights", NULL };

	int ok = PyArg_ParseTupleAndKeywords(args, kwds,
			"OO!O!O!O!O!O!O!O!O!:internal_combine", kwlist, &method,
			&PyList_Type, &images, &PyList_Type, &masks, &PyArray_Type,
			&out[0], &PyArray_Type, &out[1], &PyArray_Type, &out[2],
			&PyTuple_Type, &margs, &PyArray_Type, &zeros, &PyArray_Type,
			&scales, &PyArray_Type, &weights);

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
			// A factory class for named functions
			method_ptr.reset(NamedMethodFactory::create(method_name, margs));
		} catch (MethodException& ex) {
			// If there is a problem during construction
			return PyErr_Format(
					CombineError,
					"error during the construction of the combination method \"%s\": %s",
					method_name, ex.what());
		}
		// If we don't have a method by the name
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

	/* images are forced to be in list */
	const Py_ssize_t nimages = PyList_GET_SIZE(images);

	// getting the contents inside vectors
	std::vector<PyObject*> iarr(nimages);

	// the first image
	// borrowed reference, no decref
	iarr[0] = PyList_GetItem(images, 0);

	if (not PyArray_Check(iarr[0])) {
		return PyErr_Format(CombineError,
				"item %i in data list is not a ndarray or subclass", 0);
	}

	const int images_type = PyArray_TYPE(iarr[0]);

	for (Py_ssize_t i = 1; i < nimages; i++) {
		// Borrowed reference, no decref
		iarr[i] = PyList_GetItem(images, i);

		// checking we have and image
		if (not PyArray_Check(iarr[i])) {
			return PyErr_Format(CombineError,
					"item %zd in data list is not and ndarray or subclass", i);
		}

		// checking dtype is the same
		if (images_type != PyArray_TYPE(iarr[i]))
			return PyErr_Format(CombineError,
					"item %zd in data list has inconsistent dtype", i);
	}

	// Masks
	std::vector<PyObject*> marr(nimages);

	// checking we have and image
	marr[0] = PyList_GetItem(masks, 0);
	if (not PyArray_Check(marr[0])) {
		return PyErr_Format(CombineError,
				"item %i in masks list is not a ndarray or subclass", 0);
	}

	const int masks_type = PyArray_TYPE(marr[0]);

	for (Py_ssize_t i = 1; i < nimages; i++) {
		// Borrowed reference, no decref
		marr[i] = PyList_GetItem(masks, i);

		// checking we have and image
		if (not PyArray_Check(marr[i])) {
			return PyErr_Format(CombineError,
					"item %zd in masks list is not and ndarray or subclass", i);
		}

		// checking dtype is the same
		if (masks_type != PyArray_TYPE(marr[i]))
			return PyErr_Format(CombineError,
					"item %zd in masks list has inconsistent dtype", i);

	}

	// Checking zeros, scales and weights
	if (PyArray_NDIM(zeros) != 1) {
		return PyErr_Format(CombineError, "zeros dimension != 1");
	}
	if (PyArray_SIZE(zeros) != nimages) {
		return PyErr_Format(CombineError, "zeros size != number of images");
	}
	if (PyArray_TYPE(zeros) != NPY_DOUBLE) {
		return PyErr_Format(CombineError, "zeros has wrong type");
	}

	if (PyArray_NDIM(scales) != 1) {
		return PyErr_Format(CombineError, "scales dimension != 1");
	}
	if (PyArray_SIZE(scales) != nimages) {
		return PyErr_Format(CombineError, "scales size != number of images");
	}
	if (PyArray_TYPE(scales) != NPY_DOUBLE) {
		return PyErr_Format(CombineError, "scale has wrong type");
	}

	if (PyArray_NDIM(weights) != 1) {
		return PyErr_Format(CombineError, "weights dimension != 1");
	}
	if (PyArray_SIZE(weights) != nimages) {
		return PyErr_Format(CombineError, "weights size != number of images");
	}
	if (PyArray_TYPE(weights) != NPY_DOUBLE) {
		return PyErr_Format(CombineError, "weights has wrong type");
	}

	// Select the functions we are going to use
	// to transform the data in arrays into
	// the doubles we're working on

	// mask conversion, the only need a boolean
	bool (*mask_converter)(void*) = transformation_factory<bool>(masks_type);

	if (not mask_converter) {
		// Mask type not implemented
		return PyErr_Format(CombineError, "mask type %d not implemented",
				masks_type);
	}

	// image conversion, into a double
	double
	(*datum_converter)(void*) = transformation_factory<double>(images_type);

	if (not datum_converter) {
		// Datum type not implemented
		return PyErr_Format(CombineError, "image type %d not implemented",
				masks_type);
	}

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
	double* values[OUTDIM];

	while (iter->index < iter->size) {
		int ii = 0;
		VectorPyArrayIter::const_iterator i = iiter.begin();
		VectorPyArrayIter::const_iterator m = miter.begin();
		for (; i != iiter.end(); ++i, ++m, ++ii) {

			if (not mask_converter((*m)->dataptr)) // <- True values are skipped
			{
				// If mask converts to False, we store the value of the image array

				double* zero =
						static_cast<double*> (PyArray_GETPTR1(zeros, ii));
				double* scale = static_cast<double*> (PyArray_GETPTR1(scales,
						ii));
				double* weight = static_cast<double*> (PyArray_GETPTR1(weights,
						ii));
				if (zero and scale and weight) {
					const double original = datum_converter((*i)->dataptr);
					const double converted = (original - *zero) / (*scale);
					data.push_back(converted);
					wdata.push_back(*weight);
				} else {
					return PyErr_Format(CombineError,
							"null pointer in zero %p scale %p weight %p", zero,
							scale, weight);
				}
			}
		}

		// We obtain pointers to the result arrays
		for (size_t i = 0; i < OUTDIM; ++i)
			values[i] = (double*) oiter[i]->dataptr;

		// And pass the data to the combine method
		method_ptr->run(&data[0], &wdata[0], data.size(), values);

		// We clean up the data storage
		data.clear();
		wdata.clear();

		// And move all the iterators to the next point
		std::for_each(iiter.begin(), iiter.end(), My_PyArray_Iter_Next);
		std::for_each(miter.begin(), miter.end(), My_PyArray_Iter_Next);
		std::for_each(oiter.begin(), oiter.end(), My_PyArray_Iter_Next);
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

