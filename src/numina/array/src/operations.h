/*
 * Copyright 2008-2024 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_OPERATIONS_H
#define NU_OPERATIONS_H

#include <iterator>
#include <algorithm>
#include <numeric>
#include <functional>

#include "zip_iterator.h"

namespace Numina {

template<typename T>
inline T iround(double x) { return static_cast<T>(round(x));}

template<> inline double iround(double x) { return x;}
template<> inline float iround(double x) { return (float)x;}
template<> inline long double iround(double x) { return (long double)x;}


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
inline T weighted_population_variance(Iterator1 begin, Iterator1 end,
    Iterator2 wbegin, T mean) {

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
inline T weighted_population_variance_unit(Iterator1 begin, Iterator1 end,
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
  const T v = weighted_population_variance(begin, end, weights, m);
  return std::make_pair(m, v);
}

template<typename Iterator1, typename Iterator2>
inline std::pair<typename std::iterator_traits<Iterator1>::value_type,
    typename std::iterator_traits<Iterator1>::value_type> sum_central_tendency(
    Iterator1 begin, Iterator1 end, Iterator2 weights) {
  typedef typename std::iterator_traits<Iterator1>::value_type T;
  if (begin == end)
    return std::make_pair(T(0.0), T(0.0));

  const T mu = weighted_mean(begin, end, weights);
  const T m = mu * std::distance(begin, end);
  const T v = weighted_population_variance(begin, end, weights, mu) *
    std::distance(begin, end) * std::distance(begin, end);
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
  const T avg = mean(begin, end);
  // Using this http://mathworld.wolfram.com/StatisticalMedian.html
  // as a first approach
  const T v = weighted_population_variance(begin, end, weights, avg) / 0.637;
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


template<typename Iterator1, typename Iterator2>
std::pair<double, double>
average_central_tendency_clip(Iterator1 begin, Iterator1 end, Iterator2 weights,
    size_t low, size_t high) {
  typedef std::pair<Iterator1, Iterator2> _IterPair;
  typedef ZipIterator<_IterPair> _ZIter;
  typedef std::pair<_ZIter, _ZIter> _ZIterPair;
  typedef typename _ZIter::value_type ValueType;

  size_t n_elem = std::distance(begin, end);

  if ((low + high) >= n_elem)
    return std::make_pair(0.0, 0.0);

  _ZIter beg = make_zip_iterator(begin, weights);
  _ZIter ned = make_zip_iterator(end, weights + n_elem);

  _ZIterPair result = reject_min_max(beg, ned, low, high,
      [](const ValueType& em1, const ValueType& em2) {
         return em1.first < em2.first;
      }
   );

  _IterPair itp_beg = result.first.get_iterator_pair();
  _IterPair itp_end = result.second.get_iterator_pair();

  return average_central_tendency(itp_beg.first, itp_end.first, itp_beg.second);
}



} // namespace Numina

#endif // NU_OPERATIONS_H
