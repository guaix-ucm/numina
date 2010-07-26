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


//#include <cstddef>
//#include <algorithm>
//#include <cmath>
//#include <functional>
//#include <ext/functional>

#include "methods.h"
#include "method_exception.h"

namespace {

double mean(double* data, size_t size) {
	double sum = 0;
	for (size_t i = 0; i < size; ++i)
		sum += data[i];
	return sum / size;
}

double variance(double* data, size_t size, int dof, double mean) {
	double sum = 0;

	for (size_t i = 0; i < size; ++i) {
		const double fid = data[i] - mean;
		sum += fid * fid;
	}
	return sum / (size - dof);
}

double stdev(double* data, size_t size, int dof, double mean) {
	return sqrt(variance(data, size, dof, mean));
}

double* kth_smallest(double* data, size_t size, size_t kth) {
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

double median(double* data, size_t size) {
	//std::nth_element(data, data + size / 2, data + size);
	//median = *(data + size / 2);
	const int midpt = size % 2 != 0 ? size / 2 : size / 2 - 1;
	return *kth_smallest(data, size, midpt);
}

}

namespace Numina {

AverageMethod::AverageMethod(PyObject* args) {
	if (not PyArg_ParseTuple(args, "d", &m_dof))
		throw MethodException("problem creating MeanMethod");
}

AverageMethod::~AverageMethod() {
}

// weights are ignored for now
void AverageMethod::central_tendency(double* data, double* weights, size_t size,
		double* central, double* var) const {

	if (size == 0) {
		*central = *var = 0.0;
		return;
	}

	if (size == 1) {
		*central = data[0];
		*var = 0.0;
		return;
	}

	*central = mean(data, size);
	*var = variance(data, size, m_dof, *central);
}

MedianMethod::MedianMethod() {
}

MedianMethod::~MedianMethod() {
}

// weights are ignored for now
void MedianMethod::central_tendency(double* data, double* weights, size_t size,
		double* central, double* var) const {

	*var = 0.0;

	switch (size) {
	case 0:
		*central = 0.0;
		break;
	case 1:
		*central = data[0];
		break;
	default: {
		*central = median(data, size);
		//std::nth_element(data, data + size / 2, data + size);
		//*results[0] = *(data + size / 2);

		// Variance of the median from variance of the mean
		// http://mathworld.wolfram.com/StatisticalMedian.html
		const double smean = mean(data, size);
		const double svar = variance(data, size, 1, smean);
		*var = 4 * size / (M_PI * (2 * size + 1)) * svar;

		break;
	}
	}
}

} // namespace Numina
