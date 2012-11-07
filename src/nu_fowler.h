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

#ifndef NU_FOWLER_H
#define NU_FOWLER_H

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

#include "operations.h"

namespace Numina {

template<typename Result>
struct FowlerResult {
  Result value;
  Result variance;
  char map;
  char mask;
};

template<typename Result, typename Iterator>
FowlerResult<Result>  fowler(Iterator begin, Iterator end, int hsize) {

  typedef typename std::iterator_traits<Iterator>::value_type T;
  typedef typename std::iterator_traits<Iterator>::pointer PT;
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
  result.map = npoints;
  result.mask = 0;

  return result;
}

} // namespace Numina

#endif /* NU_FOWLER_H */
