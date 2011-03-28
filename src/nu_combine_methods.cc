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
#include <cmath>

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
  unsigned int* fdata = (unsigned int*) func_data;
  unsigned int& nmin = *fdata;
  unsigned int& nmax = *(fdata + 1);

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

int NU_sigmaclip_function(double *data, double *weights,
    int size, double *out[3], void *func_data) {

    double* fdata = (double*) func_data;
    double& low = *fdata;
    double& high = *(fdata + 1);

    size_t c_size = size;
    ZIter beg = make_zip_iterator(data, weights);
    ZIter ned = make_zip_iterator(data + size, weights + size);

    double c_mean = 0.0;
    double c_std = 0.0;
    size_t nc_size = c_size;

    do {
      ValuePair r = average_central_tendency(data, data + nc_size, weights);

      c_mean = r.first;
      c_std = sqrt(r.second);
      c_size = std::distance(beg, ned);

      const double low = c_mean - c_std * low;
      const double high = c_mean + c_std * high;
      ned = partition(beg, ned,
          // Checks if first component of a std::pair
          // is inside the range (low, high)
          // equivalent to return (low < x.first) && (high > x.first);
          __gnu_cxx::compose2(std::logical_and<bool>(),
              __gnu_cxx::compose1(
                  std::bind1st(std::less<double>(), low),
                  __gnu_cxx::select1st<typename ZIter::value_type>()),
              __gnu_cxx::compose1(
                  std::bind1st(std::greater<double>(), high),
                  __gnu_cxx::select1st<typename ZIter::value_type>())
            )
          );

      nc_size = std::distance(beg, ned);
      // We stop when std == 0, all the points would be reject otherwise
      // or when no points are removed in the iteration
    } while (c_std > 0 && (nc_size != c_size));

    *out[0] = c_mean;
    *out[1] = c_std;
    *out[2] = c_size;

  return 1;
}


void NU_destructor_function(void* cobject, void *cdata) {
  if (cdata)
      free(cdata);
}



