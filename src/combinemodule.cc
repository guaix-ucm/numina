/*
 * Copyright 2008 Sergio Pascual
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

#include <Python.h>
#include <numpy/arrayobject.h>

#include "methods.h"

typedef void (*GenericMethodPtr)(const double* data, size_t size,
		double* c, double* var, long* number, void* params);
typedef std::map<std::string, GenericMethodPtr> MethodMap;


PyDoc_STRVAR(combine__doc__, "Module doc");
PyDoc_STRVAR(internal_combine__doc__, "Combines identically shaped images");

static PyObject *CombineError;

namespace {


PyObject* py_internal_combine(PyObject *self, PyObject *args, PyObject *keywds) {
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


	MethodMap methods;
	methods["mean"] = method_mean;

	/* Check if method is registered in our table */
	GenericMethodPtr method_ptr = NULL;
	MethodMap::iterator el = methods.find(method_name);
	if(el != methods.end()) {
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
			/* Compute the results*/
			void *params = NULL;
			method_ptr(&data[0], used, p, v, n, params);
			data.clear();

		}

	return Py_None;
}

} // namespace


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

