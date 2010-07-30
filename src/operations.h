/*
 * Copyright 2008-2010 Sergio Pascual
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

namespace Numina
{

namespace detail // implementation details
{

template<class T>
class CuadSum
{
public:
  CuadSum(T mean) :
    m_mean(mean)
  {
  }
  T operator()(T sum, T val) const
  {
    const T inter = val - m_mean;
    return sum + inter * inter;
  }
private:
  T m_mean;
};

} // namespace detail

double mean(double* data, size_t size);

double variance(double* data, size_t size, int dof, double mean);

double stdev(double* data, size_t size, int dof, double mean);

double median(double* data, size_t size);

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type mean(Iterator begin,
    Iterator end)
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  return std::accumulate(begin, end, value_type(0)) / std::distance(begin, end);
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type variance(Iterator begin,
    Iterator end, unsigned int dof,
    typename std::iterator_traits<Iterator>::value_type mean)
{

  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return std::accumulate(begin, end, value_type(0),
      detail::CuadSum<value_type>(mean)) / std::distance(begin, end);
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type weighted_mean(Iterator begin,
    Iterator end, Iterator wbegin)
{
  typedef typename std::iterator_traits<Iterator>::value_type T;

  const T allw = std::accumulate(wbegin, wbegin + (end - begin), T(0));
  return std::inner_product(begin, end, wbegin, T(0)) / allw;
}

// A weighted_mean where the weights add up to one
template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type weighted_mean_unit(Iterator begin,
    Iterator end, Iterator wbegin)
{
  typedef typename std::iterator_traits<Iterator>::value_type T;

  return std::inner_product(begin, end, wbegin, T(0));
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type weighted_variance(Iterator begin,
    Iterator end, Iterator wbegin, typename std::iterator_traits<Iterator>::value_type mean)
{
  typedef typename std::iterator_traits<Iterator>::value_type T;

  T v1 = T(0);
  T v2 = T(0);
  T sum = T(0);

  while(begin != end) {
    v1 += *wbegin;
    v2 += (*wbegin) * (*wbegin);
    const T val = *begin - mean;
    sum += *wbegin * val * val;

    ++begin;
    ++wbegin;

  }

  return v1 / (v1 * v1 - v2) * sum;
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type weighted_variance_unit(Iterator begin,
    Iterator end, Iterator wbegin, typename std::iterator_traits<Iterator>::value_type mean)
{
  typedef typename std::iterator_traits<Iterator>::value_type T;

  T v2 = T(0);
  T sum = T(0);

  while(begin != end) {
    v2 += (*wbegin) * (*wbegin);
    const T val = *begin - mean;
    sum += *wbegin * val * val;

    ++begin;
    ++wbegin;

  }

  return 1 / (1 - v2) * sum;
}


template<typename Iterator>
std::pair<Iterator, Iterator> reject_min_max(Iterator begin, Iterator end,
    int nmin, int nmax)
{
  Iterator pbegin = begin;

  if (nmin >= 1)
  {
    pbegin = begin + nmin - 1;
    std::nth_element(begin, pbegin, end);
    pbegin += 1;
  }

  Iterator pend = end;

  if (nmax >= 1)
  {
    pend = end - nmax - 1;
    std::nth_element(begin, pend, end);
    pend += 1;
  }

  return std::make_pair(pbegin, pend);
}

template<typename Iterator, typename StrictWeakOrdering>
std::pair<Iterator, Iterator> reject_min_max(Iterator begin, Iterator end,
    int nmin, int nmax, StrictWeakOrdering comp)
{
  Iterator pbegin = begin;

  if (nmin >= 1)
  {
    pbegin = begin + nmin - 1;
    std::nth_element(begin, pbegin, end, comp);
    pbegin += 1;
  }

  Iterator pend = end;

  if (nmax >= 1)
  {
    pend = end - nmax - 1;
    std::nth_element(begin, pend, end, comp);
    pend += 1;
  }

  return std::make_pair(pbegin, pend);
}


} // namespace Numina

#endif // PYEMIR_OPERATIONS_H
