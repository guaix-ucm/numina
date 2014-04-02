/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * Numina is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Numina is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Numina.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace Numina {

double* kth_smallest(double* data, size_t size, size_t kth) {
	size_t l = 0;
	size_t m = size - 1;

	while (l < m) {
		double x = *(data + kth);
		size_t i = l;
		size_t j = m;
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

} // namespace Numina
