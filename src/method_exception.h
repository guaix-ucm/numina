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


#ifndef PYEMIR_METHOD_EXCEPTION_H
#define PYEMIR_METHOD_EXCEPTION_H

#include <stdexcept>

namespace Numina {

class MethodException : public std::invalid_argument{
public:
	MethodException(const char* what_arg) :
		std::invalid_argument(what_arg)
	{}
};

} // namespace Numina

#endif // PYEMIR_METHOD_EXCEPTION_H
