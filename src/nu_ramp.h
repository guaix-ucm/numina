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

#ifndef NU_RAMP_H
#define NU_RAMP_H

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

#include "operations.h"

#define MASK_SATURATION 3

namespace Numina {

template<typename Iterator>
Iterator median_1(Iterator begin, Iterator end) {
        size_t size = end - begin;
	std::nth_element(begin, begin + size / 2, end);
	return begin + size / 2;
}

template<typename Result>
struct RampResult {
  Result value;
  Result variance;
  char npix;
  char mask;
  char crmask;
};

template<typename Iterator>
RampResult<double>
slope(Iterator begin, Iterator end, double dt, double gain, double ron) {

  typename std::iterator_traits<Iterator>::difference_type nn = end - begin;
  double delt =  12.0 / (nn * (nn + 1) * (nn - 1));
  double delt2 = delt / dt;
  double nf = (nn - 1) / 2.0;
  double variance1, variance2;
  RampResult<double> result;

  double add = 0.0;
  for(Iterator i = begin; i != end; ++i)
    add += *i * ((i - begin) - nf);
  
  result.value = delt * add / dt;
  double rg = ron / gain;
  variance1 = rg * rg * delt2;
  // Photon limiting case
  variance2 = (6 * result.value * (nn * nn + 1)) / (5 * nn * dt * (nn * nn - 1) * gain);
  result.variance = variance1 + variance2;
  result.npix = nn;
  return result;
}

template<typename T>
inline RampResult<T> rround(const RampResult<double>& x) { 
  RampResult<T> res;
  res.value = static_cast<T>(round(x.value));
  res.variance = static_cast<T>(round(x.variance));
  res.npix = x.npix;
  res.mask = x.mask;
  res.crmask = x.crmask;
  return res;
}

template<> inline RampResult<double> rround(const RampResult<double>& x) { 
   return x;
}
template<> inline RampResult<float> rround(const RampResult<double>& x) { 
  RampResult<float> res;
  res.value = static_cast<float>(x.value);
  res.variance = static_cast<float>(x.variance);
  res.npix = x.npix;
  res.mask = x.mask;
  res.crmask = x.crmask;
  return res;
}

template<> inline RampResult<long double> rround(const RampResult<double>& x) { 
  RampResult<long double> res;
  res.value = static_cast<long double>(x.value);
  res.variance = static_cast<long double>(x.variance);
  res.npix = x.npix;
  res.mask = x.mask;
  res.crmask = x.crmask;
  return res;
}

template<typename Result, typename Iterator>
RampResult<Result>  ramp(Iterator begin, Iterator end, double dt, double gain, double ron, double nsig) {

  typedef typename std::iterator_traits<Iterator>::value_type T;
  typedef typename std::iterator_traits<Iterator>::pointer PT;

  const size_t dbuff_s = end - begin - 1;
  PT dbuff = new T[dbuff_s];

  // Differences between consecutive reads
  std::transform(begin + 1, end, begin, dbuff, std::minus<T>());
  // Compute the median...
  PT dbuff2 = new T[end - begin - 1];
  std::copy(dbuff, dbuff + dbuff_s, dbuff2);
  T psmedian = *median_1(dbuff2, dbuff2 + dbuff_s);
  delete [] dbuff2;

  // used to find glitches
  double sigma = std::sqrt(psmedian / gain + 2 * ron * ron);
  std::vector<double> ramp_data;
  std::vector<double> ramp_variances;
  std::vector<char> ramp_map;
  std::vector<char> ramp_cmap;

  PT init = dbuff;
  for(PT i = dbuff; i != dbuff + dbuff_s; ++i) {
    if(std::abs(static_cast<double>(psmedian) - *i) > nsig * sigma) {
      std::ptrdiff_t boff = init - dbuff;
      std::ptrdiff_t eoff = i - dbuff;
      if (i - init + 1 >= 2) {
        RampResult<double> res = slope(begin + boff, begin + eoff + 1, dt, gain, ron);
        ramp_data.push_back(res.value);
        ramp_variances.push_back(1.0 / res.variance);
        ramp_map.push_back(res.npix);
        ramp_cmap.push_back(i-dbuff + 1);
      }
      init = i + 1;
      }
  }
  std::ptrdiff_t boff = init - dbuff;
  if(end - (begin + boff) >= 2) {
    RampResult<double> res =  slope(begin + boff, end, dt, gain, ron);
    ramp_data.push_back(res.value);
    ramp_variances.push_back(1.0 / res.variance);
    ramp_map.push_back(res.npix);
  }
  delete [] dbuff;

  double hatmu = weighted_mean(ramp_data.begin(), ramp_data.end(), ramp_variances.begin());
  double hatmu_var = std::accumulate(ramp_variances.begin(), ramp_variances.end(), 0.0);

  RampResult<Result> result;
  result.value = iround<Result>(hatmu);
  result.variance = iround<Result>(1 / hatmu_var);
  result.npix = std::accumulate(ramp_map.begin(), ramp_map.end(), 0);
  result.mask = 0;
  if(!ramp_cmap.empty())
   result.crmask = ramp_cmap[0];
  else
   result.crmask = 0;
  return result;
}

  struct HWeights {
    std::vector<double> weights;
    size_t order;
    double delt1;
    double delt2;

    HWeights(size_t nn=2) : 
        weights(nn),
        order(nn),
        delt1(nn * (nn + 1) * (nn - 1) / 12.0),
        delt2((6.0 * (nn * nn + 1)) / (5.0 * nn * (nn * nn - 1)))
    {
        const double bb = 0.5 * (nn - 1);

        for(size_t i=0; i < nn / 2; ++i) {
          weights[i] = (i  - bb) / delt1;
          // the table is antisymetric
          weights[nn-i-1] = -weights[i];
        }
        // The central element is 0.0
        // if there is one
        if (nn % 2 == 1) {
          weights[nn / 2] = 0.0;
        }
    }
  };

  typedef std::vector<HWeights> HWeightsStore;

  std::vector<HWeights> create(size_t N) {
      std::vector<HWeights> w;
      for(size_t i=0; i < N-1; ++i) 
          w.push_back(HWeights(i+2));
      return w;
  }

  RampResult<double> axis_ramp(const std::vector<double>& buff, double dt,
    double gain, double ron, const HWeightsStore& wgts_store, double blank) {

    double rg = ron / gain;

    RampResult<double> result;
    result.npix = buff.size();

    if (result.npix >= 2) {
        const HWeights& hwgt = wgts_store[result.npix - 2];
        double acc = 0;
        for(size_t i=0; i < (size_t)result.npix; ++i) {
          acc += buff[i] * hwgt.weights[i];
        }
        acc /= dt;
        result.value = acc;
        // analytic variance
        result.variance= (rg * rg) / (hwgt.delt1 * dt) + acc * hwgt.delt2 / dt;
        result.mask = 0;
    }
    else {
        result.value = result.variance = blank;
        result.mask = MASK_SATURATION;
    }

    return result;
  }

} // namespace Numina

#endif /* NU_RAMP_H */
