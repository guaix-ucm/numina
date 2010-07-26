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


#ifndef PYEMIR_REJECT_FACTORY_H
#define PYEMIR_REJECT_FACTORY_H

#include <string>

#include <Python.h>

#include "method_base.h"

namespace Numina {

class RejectMethodFactory {
public:
	static auto_ptr<RejectMethod> create(const std::string& name, PyObject* args,
			auto_ptr<CombineMethod> combine_method);
};


} // namespace Numina

#endif // PYEMIR_REJECT_BASE_H
