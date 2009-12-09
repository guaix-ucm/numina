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

#include <cstddef>
#include <algorithm>

#include "methods.h"
#include "method_exception.h"

namespace Numina {

MeanMethod::MeanMethod(PyObject* args) {
	if (not PyArg_ParseTuple(args, "d", &m_dof))
		throw MethodException("problem creating MeanMethod");
}

MeanMethod::~MeanMethod() {
}

// weights are ignored for now
void MeanMethod::run(double* data, double* weights, size_t size,
		double* results[3]) const {

	if (size == 0) {
		*results[0] = *results[1] = *results[2] = 0.0;
		return;
	}

	if (size == 1) {
		*results[0] = data[0];
		*results[1] = 0.0;
		*results[2] = 1;
		return;
	}

	double sum = 0.0;
	double sum2 = 0.0;

	for (size_t i = 0; i < size; ++i) {
		sum += data[i];
		sum2 += data[i] * data[i];
	}

	*results[0] = sum / size;
	*results[2] = size;
	*results[1] = sum2 / (size - m_dof) - (sum * sum) / (size * (size - m_dof));
}

MedianMethod::MedianMethod() {
}

MedianMethod::~MedianMethod() {
}

// weights are ignored for now
void MedianMethod::run(double* data, double* weights, size_t size,
		double* results[3]) const {

	*results[1] = 0.0;
	*results[2] = size;

	switch (size) {
	case 0:
		*results[0] = *results[1] = *results[2] = 0.0;
		break;
	case 1:
		*results[0] = data[0];
		break;
	default: {

		const int midpt = ((size) & 1)?((size) / 2):(((size) / 2) - 1);
		*results[0] = *kth_smallest(data, size, midpt);
		//std::nth_element(data, data + size / 2, data + size);
		//*results[0] = *(data + size / 2);
		break;
	}
	}
}

double* MedianMethod::kth_smallest(double* data, size_t size, size_t kth) const {
	int l = 0;
	int m = size - 1;

	while (l < m) {
		double x = *(data + kth);
		int i = l;
		int j = m;
		do {
			while (*(data + i) < x)
				++i;
			while (x < *(data + j))
				--j;
			if (i <= j) {
				std::swap(*(data + i), *(data + j));
				++i;
				--j;
			}
		} while (i <= j);
		if (j < kth)
			l = i;
		if (kth < i)
			m = j;

	}

	return data + kth;
}

} // namespace Numina
