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

#include "numpytypes.h"

template<typename Origin, typename Result>
Result data_conversion(void* data) {
	Origin *p = (Origin*) data;
	return static_cast<Result> (*p);
}

#define CASE_CONVERSION_FIXED(DN) \
	case DN: \
		return &data_conversion<numpy::fixed_type<DN>::data_type,  Result>; \
		break;

#define CASE_CONVERSION_GENERIC(DN) \
	case DN: \
		return &data_conversion<numpy::generic_type<DN>::data_type,  Result>; \
		break;

template<typename Result>
Result (*transformation_factory(int image_type))(void*) {
	switch (image_type) {
			CASE_CONVERSION_GENERIC(NPY_BOOL)
			CASE_CONVERSION_GENERIC(NPY_BYTE)
			CASE_CONVERSION_GENERIC(NPY_UBYTE)
			CASE_CONVERSION_GENERIC(NPY_SHORT)
			CASE_CONVERSION_GENERIC(NPY_USHORT)
			CASE_CONVERSION_GENERIC(NPY_INT)
			CASE_CONVERSION_GENERIC(NPY_UINT)
			CASE_CONVERSION_GENERIC(NPY_ULONG)
			CASE_CONVERSION_GENERIC(NPY_LONG)
			CASE_CONVERSION_GENERIC(NPY_FLOAT)
			CASE_CONVERSION_GENERIC(NPY_DOUBLE)
			CASE_CONVERSION_GENERIC(NPY_LONGDOUBLE)
			}
			return 0; // No conversion in this case, return NULL
		}

// explicit template Instantiations
// we only need these two
template bool (*transformation_factory(int image_type))(void*);
template double (*transformation_factory(int image_type))(void*);
