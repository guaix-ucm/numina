/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
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
