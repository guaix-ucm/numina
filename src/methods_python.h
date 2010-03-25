/*
 * Copyright 2008-2010 Sergio Pascual
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


#ifndef PYEMIR_METHODS_PYTHON_H
#define PYEMIR_METHODS_PYTHON_H

#include "method_base.h"

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL numina_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

namespace Numina {

class PythonMethod: public Method {
public:
	PythonMethod(PyObject* callback, PyObject* arguments);
	~PythonMethod() {};
	void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
	PyObject* m_callback;
	PyObject* m_arguments; // This argument is ignored
};

} // namespace Numina

#endif // PYEMIR_METHODS_PYTHON_H
