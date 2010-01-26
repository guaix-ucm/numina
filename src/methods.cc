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

/* $Id$ */

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <functional>
#include <ext/functional>

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
	double sum = 0;

	for (size_t i = 0; i < size; ++i) {
		const double fid = data[i] - mean;
		sum += fid * fid;
	}
	return std::sqrt(sum / (size - dof));
}

}

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
		*results[0] = 0.0;
		break;
	case 1:
		*results[0] = data[0];
		break;
	default: {
		const int midpt = size % 2 != 0 ? size / 2 : size / 2 - 1;
		*results[0] = *kth_smallest(data, size, midpt);
		//std::nth_element(data, data + size / 2, data + size);
		//*results[0] = *(data + size / 2);

		// Variance of the median from variance of the mean
		// http://mathworld.wolfram.com/StatisticalMedian.html
		const double smean = mean(data, size);
		const double svar = variance(data, size, 1, smean);
		*results[1] = 4 * size / (M_PI * (2 * size + 1)) * svar;

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

SigmaClipMethod::SigmaClipMethod(PyObject* args) {
	if (not PyArg_ParseTuple(args, "ddd", &m_low, &m_high, &m_dof))
		throw MethodException("problem creating MeanMethod");
}

SigmaClipMethod::~SigmaClipMethod() {
}

void SigmaClipMethod::run(double* data, double* weights, size_t size,
		double* results[3]) const {
	int delta = 0;
	double c_mean = 0;
	double c_std = 0;
	int c_size = size;
	do {
		c_mean = mean(data, c_size);
		c_std = stdev(data, c_size, m_dof, c_mean);

		double* end = std::partition(data, data + c_size, __gnu_cxx::compose2(
				std::logical_and<bool>(), std::bind2nd(std::greater<double>(),
						c_mean - c_std * m_low), std::bind2nd(
						std::less<double>(), c_mean + c_std * m_high)));
		delta = c_size - (end - data);
		c_size = (end - data);
	} while (delta);
	*results[0] = c_mean;
	*results[1] = c_std;
	*results[2] = c_size;
}

} // namespace Numina
