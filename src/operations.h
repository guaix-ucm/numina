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
typename std::iterator_traits<Iterator>::value_type imean(Iterator begin,
    Iterator end)
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;

  return std::accumulate(begin, end, value_type(0)) / std::distance(begin, end);
}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type ivariance(Iterator begin,
    Iterator end, unsigned int dof,
    typename std::iterator_traits<Iterator>::value_type mean)
{

  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  return std::accumulate(begin, end, value_type(0),
      detail::CuadSum<value_type>(mean)) / std::distance(begin, end);
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

} // namespace Numina

#endif // PYEMIR_OPERATIONS_H
