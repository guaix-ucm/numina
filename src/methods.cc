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
/*
std::pair<double, double>
MedianMethod::central_tendency(double* begin, double* end, double* weights) const {

	size_t size = end - begin;
	switch (size) {
	case 0:
		return std::make_pair(0.0, 0.0);
		break;
	case 1:
		return std::make_pair(*begin, 0.0);
		break;
	default: {
		const double central = median(begin, size);
		//std::nth_element(data, data + size / 2, data + size);
		//*results[0] = *(data + size / 2);

		// Variance of the median from variance of the mean
		// http://mathworld.wolfram.com/StatisticalMedian.html
		const double smean = mean(begin, end);
		const double svar = variance(begin, end, 1, smean);
		const double var = 4 * size / (M_PI * (2 * size + 1)) * svar;
		return std::make_pair(central, var);
		break;
	}
	}
}
*/
} // namespace Numina
