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

#ifndef PYEMIR_NUMPYTYPES_H
#define PYEMIR_NUMPYTYPES_H

#include <numpy/arrayobject.h>

#define PYEMIR_NUMPY_GENERIC(X, Y) \
	template<> \
	struct generic_type<X> { \
	typedef Y data_type; \
	static const bool specialized = true; \
	};


#define PYEMIR_NUMPY_FIXED(X, Y) \
	template<> \
	struct fixed_type<X> { \
	typedef Y data_type; \
	static const bool specialized = true; \
	};

namespace numpy {

template<int N>
struct generic_type {
	typedef void* data_type;
	static const bool specialized = false;
};

template<int N>
struct fixed_type {
	typedef void* data_type;
	static const bool specialized = false;
};

PYEMIR_NUMPY_GENERIC(NPY_BOOL, npy_bool);
PYEMIR_NUMPY_GENERIC(NPY_BYTE, npy_byte);
PYEMIR_NUMPY_GENERIC(NPY_UBYTE, npy_ubyte);
PYEMIR_NUMPY_GENERIC(NPY_SHORT, npy_short);
PYEMIR_NUMPY_GENERIC(NPY_USHORT, npy_ushort);
PYEMIR_NUMPY_GENERIC(NPY_INT, npy_int);
PYEMIR_NUMPY_GENERIC(NPY_UINT, npy_uint);
PYEMIR_NUMPY_GENERIC(NPY_ULONG, npy_ulong);
PYEMIR_NUMPY_GENERIC(NPY_LONG, npy_long);
PYEMIR_NUMPY_GENERIC(NPY_FLOAT, npy_float);
PYEMIR_NUMPY_GENERIC(NPY_DOUBLE, npy_double);
PYEMIR_NUMPY_GENERIC(NPY_LONGDOUBLE, npy_longdouble);

PYEMIR_NUMPY_FIXED(NPY_UINT8, npy_uint8);
PYEMIR_NUMPY_FIXED(NPY_INT16, npy_int16);
PYEMIR_NUMPY_FIXED(NPY_INT32, npy_int32);
PYEMIR_NUMPY_FIXED(NPY_INT64, npy_int64);
PYEMIR_NUMPY_FIXED(NPY_FLOAT32, npy_float32);
PYEMIR_NUMPY_FIXED(NPY_FLOAT64, npy_float64);

}

#endif // PYEMIR_METHODS_H
