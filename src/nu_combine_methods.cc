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

#include <Python.h>

#include "nu_combine_defs.h"
#include "functional.h"
#include "operations.h"
#include "zip_iterator.h"

using Numina::ZipIterator;
using Numina::make_zip_iterator;
using Numina::compose;
using Numina::average_central_tendency;
using Numina::average_central_tendency_clip;
using Numina::median_central_tendency;

typedef std::pair<double*, double*> IterPair;
typedef ZipIterator<IterPair> ZIter;
typedef std::pair<ZIter, ZIter> ZIterPair;
typedef std::pair<double, double> ValuePair;

int NU_mean_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data)
{
  *out[2] = size;
  ValuePair r = average_central_tendency(data, data + size, weights);

  *out[0] = r.first;
  *out[1] = r.second;

  return 1;
}

int NU_median_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data)
{
  *out[2] = size;
  ValuePair r = median_central_tendency(data, data + size, weights);

  *out[0] = r.first;
  *out[1] = r.second;

  return 1;
}

int NU_minmax_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data)
{
  unsigned* fdata = (unsigned*) func_data;
  unsigned& nmin = *fdata;
  unsigned& nmax = *(fdata + 1);

  if ((nmin + nmax) == size) {
    *out[0] = 0;
    *out[1] = 0;
    *out[2] = 0;
    return 1;
  }

  if ((nmin + nmax) > size) {
    PyErr_SetString(PyExc_ValueError, "nmin + nmax greater than available points");
    return 0;
  }


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
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data) {

    double* fdata = (double*) func_data;
    double& slow = *fdata;
    double& shigh = *(fdata + 1);

    ZIter beg = make_zip_iterator(data, weights);
    ZIter ned = make_zip_iterator(data + size, weights + size);

    double c_mean = 0.0;
    double c_std = 0.0;
    size_t nc_size = size;

    do {
      ValuePair r = average_central_tendency(data, data + nc_size, weights);

      c_mean = r.first;
      c_std = sqrt(r.second);
      size = std::distance(beg, ned);

      const double low = c_mean - c_std * slow;
      const double high = c_mean + c_std * shigh;
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
    } while (c_std > 0 && (nc_size != size));

    *out[0] = c_mean;
    *out[1] = c_std;
    *out[2] = size;

  return 1;
}

int NU_quantileclip_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data) {

  double& fclip = *(double*) func_data;

  size_t n_elem = size;
  double n_elem_to_clip = n_elem * (fclip);
  size_t nclip = static_cast<size_t>(floor(n_elem_to_clip));
  size_t mclip = static_cast<size_t>(ceil(n_elem_to_clip));

  *out[2] = n_elem - 2 * n_elem_to_clip;

  if (nclip == mclip) {
    // No interpolation
    ValuePair r = average_central_tendency_clip(data, data + size, weights, nclip, nclip);
    *out[0] = r.first;
    *out[1] = r.second;
  }
  else {
    // Interpolation
    ValuePair r1 = average_central_tendency_clip(data, data + size, weights, nclip, nclip);
    ValuePair r2 = average_central_tendency_clip(data, data + size, weights, mclip, mclip);
    *out[0] = r1.first + (n_elem_to_clip - nclip) * (r2.first - r1.first);
    *out[1] = r1.second + (n_elem_to_clip - nclip) * (r2.second - r1.second);
  }

  return 1;
}

void NU_destructor_function(void* cobject, void *cdata) {
  if (cdata)
      free(cdata);
}



