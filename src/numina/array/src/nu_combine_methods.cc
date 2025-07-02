/*
 * Copyright 2008-2024 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

#include "nu_combine_defs.h"
#include "operations.h"
#include "zip_iterator.h"

#ifndef NAN
    static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
    #define NAN (*(const float *) __nan)
#endif

using Numina::ZipIterator;
using Numina::make_zip_iterator;
using Numina::average_central_tendency;
using Numina::sum_central_tendency;
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

int NU_sum_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data)
{
  *out[2] = size;
  ValuePair r = sum_central_tendency(data, data + size, weights);
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
  int* fdata = (int*) func_data;
  int& nmin = *fdata;
  int& nmax = *(fdata + 1);

  const size_t total = static_cast<size_t>(nmin + nmax);

  if (total == size) {
    *out[0] = 0;
    *out[1] = 0;
    *out[2] = 0;
    return 1;
  }

  if (total > size) {
    *out[0] = NAN;
    *out[1] = 0;
    *out[2] = -1;
    //PyErr_SetString(PyExc_ValueError, "nmin + nmax greater than available points");
    return 0;
  }


  ZIterPair result = reject_min_max(make_zip_iterator(data, weights),
      make_zip_iterator(data + size, weights + size), nmin, nmax,
      [](const ValuePair& em1, const ValuePair& em2) {
         return em1.first < em2.first;
      }
  );

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

    if (size == 0) {
      *out[0] = 0.0;
      *out[1] = 0.0;
      *out[2] = 0;

      return 1;
    }


    do {
      ValuePair r = average_central_tendency(data, data + nc_size, weights);

      c_mean = r.first;
      c_std = sqrt(r.second);
      size = std::distance(beg, ned);

      const double low = c_mean - c_std * slow;
      const double high = c_mean + c_std * shigh;

      if (beg != ned) {
        ned = std::partition(beg, ned, [low, high](const auto& em) {
            return (low < em.first()) && (high > em.first());
        });
      }

      nc_size = std::distance(beg, ned);
      // We stop when std == 0, all the points would be reject otherwise
      // or when no points are removed in the iteration
    } while ((c_std > 0) && (nc_size > 0) && (nc_size != size));

    *out[0] = c_mean;
    *out[1] = c_std;
    *out[2] = size;

  return 1;
}

int NU_quantileclip_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data) {

  double& fclip = *(double*) func_data;

  double n_elem_to_clip = size * (fclip);
  size_t nclip = static_cast<size_t>(floor(n_elem_to_clip));
  size_t mclip = static_cast<size_t>(ceil(n_elem_to_clip));

  if (size - mclip - mclip <= 0) {
    // We reject more points that we have locally
    *out[0] = 0.0;
    *out[1] = 0.0;
    *out[2] = size - mclip - mclip;
    return 1;
  }

  *out[2] = size - nclip - mclip;

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
    *out[0] = r1.first + mclip * (r2.first - r1.first);
    *out[1] = r1.second + mclip * (r2.second - r1.second);
  }

  return 1;
}


int NU_crmean_function(double *data, double *weights,
    size_t size, double *out[NU_COMBINE_OUTDIM], void *func_data)
{
  double* fdata = (double*) func_data;

  double& gain = *fdata;
  double& ron = *(fdata + 1);
  double& nsig = *(fdata + 2);

  if (size < 3) {
    *out[0] = 0;
    *out[1] = 0;
    *out[2] = 0;
    return 1;
  }

  ValuePair res = average_central_tendency(data, data + size, weights);
  double n_expect = *std::min_element(data, data + size);
  double c_std = sqrt(res.second);
  double s_expect = sqrt(ron*ron + n_expect / gain);

  if (c_std > nsig * s_expect) {
  //min
    *out[0] = n_expect;
    *out[1] = 0.832 * std::pow(size, -0.148) * c_std; // Why ?? Ask Nico (copilot)
    *out[2] = 1;
  }
  else {
    *out[0] = res.first;
    *out[1] = res.second;
    *out[2] = size;
  }
  return 1;
}



