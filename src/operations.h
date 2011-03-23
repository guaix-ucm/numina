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

#ifndef PYEMIR_OPERATIONS_H
#define PYEMIR_OPERATIONS_H

#include <iterator>
#include <algorithm>
#include <numeric>

namespace Numina {

namespace Detail // implementation details
{

template<class T>
class CuadSum {
public:
  CuadSum(T mean) :
    m_mean(mean) {
  }
  T operator()(T sum, T val) const {
    const T inter = val - m_mean;
    return sum + inter * inter;
  }
private:
  T m_mean;
};

} // namespace detail

template<typename Iterator>
inline typename std::iterator_traits<Iterator>::value_type median(Iterator begin,
    Iterator end) {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  std::size_t size = end - begin;
  std::size_t middle = size / 2;
  Iterator miditer = begin + middle;
  std::nth_element(begin, miditer, end);

  // Odd number of objects
  if (size % 2 != 0)
    return *miditer;

  // Even number of objects
  value_type store = *miditer;
  Iterator miditer2 = miditer - 1;
  std::nth_element(begin, miditer2, end);

  return (store + *miditer2) / 2.0;
}

template<typename Iterator>
inline typename std::iterator_traits<Iterator>::value_type mean(Iterator begin,
    Iterator end) {
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  return std::accumulate(begin, end, value_type(0.0)) / std::distance(begin, end);
}

template<typename Iterator>
inline typename std::iterator_traits<Iterator>::value_type variance(
    Iterator begin, Iterator end, unsigned int dof,
    typename std::iterator_traits<Iterator>::value_type mean) {

  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return std::accumulate(begin, end, value_type(0),
      Detail::CuadSum<value_type>(mean)) / std::distance(begin, end);
}

template<typename Iterator1, typename Iterator2>
inline typename std::iterator_traits<Iterator1>::value_type weighted_mean(
    Iterator1 begin, Iterator1 end, Iterator2 wbegin) {
  typedef typename std::iterator_traits<Iterator2>::value_type T;

  const T allw = std::accumulate(wbegin, wbegin + (end - begin), T(0));
  return std::inner_product(begin, end, wbegin, T(0)) / allw;
}

// A weighted_mean where the weights add up to one
template<typename Iterator1, typename Iterator2>
inline typename std::iterator_traits<Iterator1>::value_type weighted_mean_unit(
    Iterator1 begin, Iterator1 end, Iterator2 wbegin) {
  typedef typename std::iterator_traits<Iterator2>::value_type T;

  return std::inner_product(begin, end, wbegin, T(0));
}

template<typename Iterator1, typename Iterator2, typename T>
inline T weighted_variance(Iterator1 begin, Iterator1 end, Iterator2 wbegin,
    T mean) {

  T v1 = T(0);
  T v2 = T(0);
  T sum = T(0);

  while (begin != end) {
    v1 += *wbegin;
    v2 += (*wbegin) * (*wbegin);
    const T val = *begin - mean;
    sum += *wbegin * val * val;

    ++begin;
    ++wbegin;

  }

  return v1 / (v1 * v1 - v2) * sum;
}

template<typename Iterator1, typename Iterator2, typename T>
inline T weighted_variance_unit(Iterator1 begin, Iterator1 end,
    Iterator2 wbegin, T mean) {

  T v2 = T(0);
  T sum = T(0);

  while (begin != end) {
    v2 += (*wbegin) * (*wbegin);
    const T val = *begin - mean;
    sum += *wbegin * val * val;

    ++begin;
    ++wbegin;

  }

  return 1 / (1 - v2) * sum;
}

template<typename Iterator1, typename Iterator2>
inline std::pair<typename std::iterator_traits<Iterator1>::value_type,
    typename std::iterator_traits<Iterator1>::value_type> average_central_tendency(
    Iterator1 begin, Iterator1 end, Iterator2 weights) {
  typedef typename std::iterator_traits<Iterator1>::value_type T;
  if (begin == end)
    return std::make_pair(T(0.0), T(0.0));
  if (begin + 1 == end)
    return std::make_pair(*begin, T(0.0));

  const T m = weighted_mean(begin, end, weights);
  const T v = weighted_variance(begin, end, weights, m);
  return std::make_pair(m, v);
}

template<typename Iterator1, typename Iterator2>
inline std::pair<typename std::iterator_traits<Iterator1>::value_type,
    typename std::iterator_traits<Iterator1>::value_type> median_central_tendency(
    Iterator1 begin, Iterator1 end, Iterator2 weights) {
  typedef typename std::iterator_traits<Iterator1>::value_type T;
  if (begin == end)
    return std::make_pair(T(0.0), T(0.0));
  if (begin + 1 == end)
    return std::make_pair(*begin, T(0.0));

  // Weights are ignored
  const T m = median(begin, end);
  // FIXME
  const T v = weighted_variance(begin, end, weights, m);
  return std::make_pair(m, v);
}

template<typename Iterator>
inline std::pair<Iterator, Iterator> reject_min_max(Iterator begin,
    Iterator end, size_t nmin, size_t nmax) {
  Iterator pbegin = begin;

  if (nmin >= 1) {
    pbegin = begin + nmin - 1;
    std::nth_element(begin, pbegin, end);
    pbegin += 1;
  }

  Iterator pend = end;

  if (nmax >= 1) {
    pend = end - nmax - 1;
    std::nth_element(pbegin, pend, end);
    pend += 1;
  }

  return std::make_pair(pbegin, pend);
}

template<typename Iterator, typename StrictWeakOrdering>
inline std::pair<Iterator, Iterator> reject_min_max(Iterator begin,
    Iterator end, size_t nmin, size_t nmax, StrictWeakOrdering comp) {

  Iterator pbegin = begin;
  if (nmin >= 1) {
    pbegin = begin + nmin - 1;
    std::nth_element(begin, pbegin, end, comp);
    pbegin += 1;
  }

  Iterator pend = end;
  if (nmax >= 1) {
    pend = end - nmax - 1;
    std::nth_element(pbegin, pend, end, comp);
    pend += 1;
  }

  return std::make_pair(pbegin, pend);
}

} // namespace Numina

#endif // PYEMIR_OPERATIONS_H
