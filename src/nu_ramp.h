/*
 * Copyright 2008-2012 Universidad Complutense de Madrid
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

namespace Numina {

template<typename Result>
struct RampResult {
  Result value;
  Result variance;
  char map;
  char mask;
  char crmask;

};

template<typename Iterator>
std::pair<double, double> 
slope(Iterator begin, Iterator end, double dt, double gain, double ron) {

  typename std::iterator_traits<Iterator>::difference_type nn = end - begin;
  double delt =  12.0 / (nn * (nn + 1) * (nn - 1));
  double nf = (nn - 1) / 2.0;
  double final = 0;
  double variance = 0;

  double add = 0.0;
  for(Iterator i = begin; i != end; ++i)
    add += *i * ((i - begin) - nf);
  
  final = delt * add / dt;
  variance = delt * ron * ron / dt / dt;
  return std::make_pair(final, variance);
}

template<typename T>
inline T iround(double x) { return static_cast<T>(round(x));}

template<> inline double iround(double x) { return x;}
template<> inline float iround(double x) { return (float)x;}
template<> inline long double iround(double x) { return (long double)x;}

template<typename Result, typename Iterator>
RampResult<Result>  glitches(Iterator begin, Iterator end, double dt, double gain, double ron, double nsig) {

  typedef typename std::iterator_traits<Iterator>::value_type T;
  typedef typename std::iterator_traits<Iterator>::pointer PT;

  PT dbuff = new T[end - begin - 1];
  const size_t dbuff_s = end - begin - 1;

  std::transform(begin + 1, end, begin, dbuff, std::minus<T>());
  PT dbuff2 = new T[end - begin - 1];
  std::copy(dbuff, dbuff + dbuff_s, dbuff2);
  std::nth_element(dbuff2, dbuff2 + dbuff_s / 2, dbuff2 + dbuff_s);
  T psmedian = dbuff2[dbuff_s / 2];
  delete [] dbuff2;

  // used to find glitches
  double sigma = std::sqrt(psmedian / gain + 2 * ron * ron);
  std::vector<double> ramp_data;
  std::vector<double> ramp_variances;

  PT init = dbuff;
  for(PT i = dbuff; i != dbuff + dbuff_s; ++i) {
    if(std::abs(static_cast<double>(psmedian) - *i) > nsig * sigma) {
      std::ptrdiff_t boff = init - dbuff;
      std::ptrdiff_t eoff = i - dbuff;
      if (i - init + 1 >= 2) {
        std::pair<double, double> res = slope(begin + boff, begin + eoff + 1, dt, gain, ron);
        ramp_data.push_back(res.first);
        ramp_variances.push_back(1.0 / res.second);
      }
      init = i + 1;
      }
  }
  std::ptrdiff_t boff = init - dbuff;
  if(end - (begin + boff) >= 2) {
    std::pair<double, double> res =  slope(begin + boff, end, dt, gain, ron);
    ramp_data.push_back(res.first);
    ramp_variances.push_back(1.0 / res.second);
  }
  delete [] dbuff;

  double hatmu = weighted_mean(ramp_data.begin(), ramp_data.end(), ramp_variances.begin());
  //double hatmu_var = weighted_population_variance(ramp_data.begin(), ramp_data.end(), ramp_variances.begin(), hatmu);

  RampResult<Result> result;
  result.value = iround<Result>(hatmu);
  result.variance = 11;// iround<Result>(hatmu_var);
  result.map = 32;
  result.mask = 12;
  result.crmask = 11;
  return result;
}

template<typename Result, typename Iterator>
void ramp(Iterator begin, Iterator end,
    char* value, char* variance, char* map, char* mask, char* crmask,
    int saturation, double dt, double gain, double ron, double nsig)
{
  typedef std::reverse_iterator<Iterator> ReverseIterator;
  std::reverse_iterator<Iterator> rbeg(end);
  std::reverse_iterator<Iterator> rend(begin);

  Result* rvalue = reinterpret_cast<Result*>(value);
  Result* rvariance = reinterpret_cast<Result*>(variance);

  // The rest are already char type...

  Iterator nend = begin;

  for(ReverseIterator i = rbeg; i != rend; ++i) {
     if (static_cast<int>(*i) < saturation) {
       nend = end - (i - rbeg);
       break;
    }
  }

  RampResult<Result> result = glitches<Result>(begin, nend, dt, gain, ron, nsig);

  *rvalue = result.value;
  *rvariance = result.variance;
  *map = result.map;
  *mask= result.mask;
  *crmask = result.crmask;
 }





} // namespace Numina

#endif /* NU_RAMP_H */
