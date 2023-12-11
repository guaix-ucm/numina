/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_FOWLER_H
#define NU_FOWLER_H

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

#include "operations.h"

#define MASK_SATURATION 3

namespace Numina {

template<typename Result>
struct FowlerResult {
  Result value;
  Result variance;
  char npix;
  char mask;
  FowlerResult() : value(0), variance(0), npix(0), mask(0)
  {}
};

template<typename Result, typename Iterator>
FowlerResult<Result>  fowler(Iterator begin, Iterator end, int hsize) {

  typedef typename std::iterator_traits<Iterator>::value_type T;
  Iterator i1 = begin;
  Iterator i2 = begin + hsize;
  Result accum = 0;
  Result accum2 = 0;
  int npoints = 0;

  for(;(i1!=end) and (i2 != end); ++i1, ++i2) {
    T val = (*i2 - *i1);
    accum += val;
    accum2 += val * val;
    npoints++;
  }

  FowlerResult<Result> result;
  // Probably this can be done better
  result.value = iround<Result>(accum / npoints);
  result.variance = iround<Result>(accum2) / npoints - result.value * result.value;
  result.npix = npoints;
  result.mask = 0;

  return result;
}

typedef FowlerResult<double>  (*axis_fowler_func_t)(const std::vector<double>& buff, double teff, double gain, 
    double ron, double ts, double blank);

FowlerResult<double> 
axis_fowler(const std::vector<double>& buff, double teff, double gain, 
    double ron, double ts, double blank) {
    FowlerResult<double> result;
    result.npix = buff.size();
    double rg = ron / gain;
    double accum = 0;
    if (result.npix == 0) {
        result.value = result.variance = blank;
        result.mask = MASK_SATURATION;
    }
    else {
        for(size_t i = 0; i < buff.size(); ++i) {
            accum += buff[i];
        }

        result.value = accum / result.npix;
        result.variance = 2 * rg * rg / result.npix;
        if(teff > 0) 
          result.variance += result.value / (gain * gain) * (1 + ts / (3 *teff) * (1 / result.npix - result.npix));
    }
    return result;
}

/* A RON limited version of axis_fowler */
FowlerResult<double> 
axis_fowler_ron(const std::vector<double>& buff, double teff, double gain, 
    double ron, double ts, double blank) {
    FowlerResult<double> result;
    result.npix = buff.size();
    double rg = ron / gain;
    double accum = 0;
    if (result.npix == 0) {
        result.value = result.variance = blank;
        result.mask = MASK_SATURATION;
    }
    else {
        for(size_t i = 0; i < buff.size(); ++i) {
            accum += buff[i];
        }

        result.value = accum / result.npix;
        result.variance = 2 * rg * rg / result.npix;
    }
    return result;
}

} // namespace Numina

#endif /* NU_FOWLER_H */
