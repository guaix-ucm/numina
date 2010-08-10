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


#include "method_factory.h"
#include "methods.h"

namespace Numina {

auto_ptr<CombineMethod>
CombineMethodFactory::create(const std::string& name, PyObject* args) {
	if (name == "average") {
	  return auto_ptr<CombineMethod>(new CombineHV<MethodAverage>());
	}
	if (name == "median") {
	  return auto_ptr<CombineMethod>(new CombineHV<MethodMedian>());
	}
	return auto_ptr<CombineMethod>();
}

} // namespace Numina
