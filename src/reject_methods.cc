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

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <functional>
#include <ext/functional>

#include "operations.h"
#include "method_exception.h"
#include "reject_methods.h"
#include "zip_iterator.h"

namespace Numina
{

// Compares two std::pair objects. Returns true
// if the first component of the first is less than the first component
// of the second std::pair
template<typename Iterator1, typename Iterator2>
struct LessFirst: public std::binary_function<typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, bool>
{
  bool operator()(
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& a,
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& b) const
  {
    return a.first < b.first;
  }
};

// Checks if first component of a std::pair
// is inside the range (low, high)
template<typename Iterator1, typename Iterator2>
class InRangeFirst: public std::unary_function<typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, bool>
{
public:
  InRangeFirst(double low, double high) :
    m_lowc(low), m_highc(high)
  {
  }
  bool operator()(
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& x) const
  {
    return (x.first < m_highc) && (x.first > m_lowc);
  }
private:
  double m_lowc;
  double m_highc;
};

NoneReject::NoneReject(auto_ptr<CombineMethod> combine) :
  RejectMethod(combine)
{
}

NoneReject::~NoneReject()
{
}

void NoneReject::combine(double* data, double* weights, size_t size,
    double* results[3]) const
{
  central_tendency(data, weights, size, results[0], results[1]);
  *results[2] = size;
}

MinMax::MinMax(auto_ptr<CombineMethod> combine, unsigned int nmin,
    unsigned int nmax) :
  RejectMethod(combine), m_nmin(nmin), m_nmax(nmax)
{
}

MinMax::~MinMax()
{
}

void MinMax::combine(double* data, double* weights, size_t size,
    double* results[3]) const
{
  typedef std::pair<double*, double*> IterPair;
  typedef ZipIterator<IterPair> ZIter;
  typedef std::pair<ZIter, ZIter> ZIterPair;

  ZIterPair result = reject_min_max(make_zip_iterator(data, weights),
      make_zip_iterator(data + size, weights + size), m_nmin, m_nmax,
      LessFirst<double*, double*> ());

  *results[2] = result.second - result.first;
  IterPair beg = result.first.get_iterator_pair();
  IterPair end = result.second.get_iterator_pair();
  central_tendency(beg.first, beg.second, *results[2], results[0], results[1]);
}

SigmaClipMethod::SigmaClipMethod(auto_ptr<CombineMethod> combine, double low,
    double high) :
  RejectMethod(combine), m_low(low), m_high(high)
{
}

SigmaClipMethod::~SigmaClipMethod()
{
}

void SigmaClipMethod::combine(double* data, double* weights, size_t size,
    double* results[3]) const
{
  typedef std::pair<double*, double*> IterPair;
  typedef ZipIterator<IterPair> ZIter;

  ZIter begin = make_zip_iterator(data, weights);
  ZIter end = make_zip_iterator(data + size, weights + size);

  int delta = 0;
  double c_mean = 0;
  double c_std = 0;
  size_t c_size = size;
  do
  {
    central_tendency(data, weights, c_size, &c_mean, &c_std);
    c_std = sqrt(c_std);

    end = partition(begin, end, InRangeFirst<double*, double*> (c_mean
        - c_std * m_low, c_mean + c_std * m_high));

    delta = c_size - (end - begin);
    c_size = end - begin;

  } while (delta);
  *results[0] = c_mean;
  *results[1] = c_std;
  *results[2] = c_size;
}

} // namespace Numina
