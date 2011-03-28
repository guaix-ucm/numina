/*
 * Copyright 2008-2011 Sergio Pascual
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

#include <memory>
#include <ext/functional>

#include "functional.h"
#include "method_base.h"
#include "operations.h"
#include "zip_iterator.h"

using Numina::ZipIterator;
using Numina::make_zip_iterator;
using Numina::compose;
using Numina::average_central_tendency;
using Numina::median_central_tendency;

typedef std::pair<double*, double*> IterPair;
typedef ZipIterator<IterPair> ZIter;
typedef std::pair<ZIter, ZIter> ZIterPair;
typedef std::pair<double, double> ValuePair;


int NU_mean_function(double *data, double *weights,
    int size, double *out[3], void *func_data)
{
  *out[2] = size;
  ValuePair r = average_central_tendency(data, data + size, weights);

  *out[0] = r.first;
  *out[1] = r.second;

  return 1;
}

int NU_median_function(double *data, double *weights,
    int size, double *out[3], void *func_data)
{
  *out[2] = size;
  ValuePair r = median_central_tendency(data, data + size, weights);

  *out[0] = r.first;
  *out[1] = r.second;

  return 1;
}

int NU_minmax_function(double *data, double *weights,
    int size, double *out[3], void *func_data)
{
  static size_t nmin = 1;
  static size_t nmax = 1;

  ZIterPair result = reject_min_max(make_zip_iterator(data, weights),
      make_zip_iterator(data + size, weights + size), nmin, nmax,
      // Compares two std::pair objects. Returns true
      // if the first component of the first is less than the first component
      // of the second std::pair
      compose(std::less<double>(), __gnu_cxx::select1st<
          typename ZIter::value_type>(), __gnu_cxx::select1st<
          typename ZIter::value_type>()));

  *out[2] = result.second - result.first;
  IterPair beg = result.first.get_iterator_pair();
  IterPair ned = result.second.get_iterator_pair();
  ValuePair r = average_central_tendency(beg.first, ned.first, beg.second);
  *out[0] = r.first;
  *out[1] = r.second;

  return 1;
}

void NU_destructor_function(void* cobject, void *cdata) {
  if (cdata)
      free(cdata);
}



