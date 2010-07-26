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


#ifndef PYEMIR_METHOD_BASE_H
#define PYEMIR_METHOD_BASE_H

#include <memory>

namespace Numina {

using std::auto_ptr;

class CombineMethod {
public:
	virtual ~CombineMethod() {}
	virtual void central_tendency(double* data, double* weights, size_t size,
			double* central, double* variance) const = 0;
};

class RejectMethod {
public:
	RejectMethod(auto_ptr<CombineMethod> combine) :
		m_combine(combine)
	{}
	virtual ~RejectMethod() {}
	virtual void run(double* data, double* weights, size_t size, double* results[3]) const = 0;
	void central_tendency(double* data, double* weights, size_t size,
			double* central, double* variance) const {
		m_combine->central_tendency(data, weights, size, data, variance);
	}
private:
	// Non copyable
	RejectMethod(const RejectMethod&);
	const RejectMethod& operator=(const RejectMethod&);

	auto_ptr<CombineMethod> m_combine;
};

} // namespace Numina

#endif // PYEMIR_METHOD_BASE_H
