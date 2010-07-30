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

#include <cmath>

#include "methods.h"
#include "method_exception.h"
#include "operations.h"

namespace Numina {

AverageMethod::AverageMethod(unsigned int dof) :
		m_dof(dof)
{}

AverageMethod::~AverageMethod() {
}

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

	*central = weighted_mean(data, data + size, weights);
	*var = weighted_variance(data, data + size, weights, *central);
}

std::pair<double, double>
AverageMethod::central_tendency2(double* begin, double* end, double* weights) const {

  if (begin == end)
    return std::make_pair(0.0, 0.0);


  if (end - begin == 1)
    return std::make_pair(*begin, 0.0);

  const double central = weighted_mean(begin, end, weights);
  const double var = weighted_variance(begin, end, weights, central);
  return std::make_pair(central, var);
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
		const double smean = mean(data, data + size);
		const double svar = variance(data, data + size, 1, smean);
		*var = 4 * size / (M_PI * (2 * size + 1)) * svar;

		break;
	}
	}
}

} // namespace Numina
