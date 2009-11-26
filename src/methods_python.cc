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

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


void method_python(const double* data, size_t size, double* results[3],
		void* callback) {

	PyObject *fun = (PyObject*) callback;
	npy_intp dims = size;
	PyObject* pydata = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE,
			(void*) data);

	// Calling the function with the pydata
	PyObject* argl = Py_BuildValue("(O)", pydata);
	Py_DECREF(pydata);

	PyObject* result = PyEval_CallObject(fun, argl);
	Py_DECREF(argl);

	if (PyTuple_Check(result) and PyTuple_Size(result) == 3) {
		for (size_t i; i < 3; ++i) {
			PyObject* dd = PyTuple_GET_ITEM(result, i);
			if(PyFloat_Check(dd)) {
				*results[i] = PyFloat_AsDouble(dd);
			}
		}
	}
	Py_DECREF(result);
}
