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
FowlerResult<Result>  fowler(Iterator begin, Iterator end) {

  typedef typename std::iterator_traits<Iterator>::value_type T;
  typedef typename std::iterator_traits<Iterator>::pointer PT;

  FowlerResult<Result> result;
  result.value = iround<Result>(0);
  result.variance = iround<Result>(0);
  result.map = 0; //std::accumulate(ramp_map.begin(), ramp_map.end(), 0);
  result.mask = 0;

  return result;
}

} // namespace Numina

#endif /* NU_FOWLER_H */
