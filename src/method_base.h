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

#ifndef PYEMIR_METHOD_BASE_H
#define PYEMIR_METHOD_BASE_H

namespace Numina {

class Method {
public:
	virtual ~Method() {}
	virtual void run(double* data, double* weights, size_t size, double* results[3]) const = 0;
};

} // namespace Numina

#endif // PYEMIR_METHOD_BASE_H
