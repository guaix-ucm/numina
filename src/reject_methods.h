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

#ifndef PYEMIR_REJECT_METHODS_H
#define PYEMIR_REJECT_METHODS_H

#include <memory>

#include "method_base.h"
#include "operations.h"
#include "zip_iterator.h"

namespace Numina {

// Compares two std::pair objects. Returns true
// if the first component of the first is less than the first component
// of the second std::pair
template<typename Iterator1, typename Iterator2>
struct LessFirst: public std::binary_function<typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, bool> {
  bool operator()(
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& a,
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& b) const {
    return a.first < b.first;
  }
};

// Checks if first component of a std::pair
// is inside the range (low, high)
template<typename Iterator1, typename Iterator2>
class InRangeFirst: public std::unary_function<typename ZipIterator<std::pair<
    Iterator1, Iterator2> >::value_type, bool> {
public:
  InRangeFirst(double low, double high) :
    m_lowc(low), m_highc(high) {
  }
  bool operator()(
      const typename ZipIterator<std::pair<Iterator1, Iterator2> >::value_type& x) const {
    return (x.first < m_highc) && (x.first > m_lowc);
  }
private:
  double m_lowc;
  double m_highc;
};

template<typename CentralTendency, typename Iterator1, typename Iterator2,
    typename Result = double>
class RejectNone {
public:
  RejectNone(const CentralTendency& central) :
    m_central(central) {
  }

  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {
    std::pair<Result, Result> r = m_central(begin, end, weights);
    *results[0] = r.first;
    *results[1] = r.second;
    *results[2] = std::distance(begin, end);
  }
private:
  CentralTendency m_central;
};

template<typename CentralTendency, typename Iterator1, typename Iterator2,
    typename Result = double>
class RejectMinMax {
public:
  RejectMinMax(const CentralTendency& central, size_t nmin, size_t nmax) :
    m_central(central), m_nmin(nmin), m_nmax(nmax) {
  }

  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {
    typedef std::pair<Iterator1, Iterator2> IterPair;
    typedef ZipIterator<IterPair> ZIter;
    typedef std::pair<ZIter, ZIter> ZIterPair;

    ZIterPair result = reject_min_max(make_zip_iterator(begin, weights),
        make_zip_iterator(end, weights + (end - begin)), m_nmin, m_nmax,
        LessFirst<Iterator1, Iterator2> ());

    *results[2] = result.second - result.first;
    IterPair beg = result.first.get_iterator_pair();
    IterPair ned = result.second.get_iterator_pair();
    std::pair<double, double> r = m_central(beg.first, ned.first, beg.second);
    *results[0] = r.first;
    *results[1] = r.second;
  }
private:
  CentralTendency m_central;
  size_t m_nmin;
  size_t m_nmax;
};

template<typename CentralTendency, typename Iterator1, typename Iterator2,
    typename Result = double>
class RejectSigmaClip {
public:
  RejectSigmaClip(const CentralTendency& central, double low, double high) :
    m_central(central), m_low(low), m_high(high) {
  }

  virtual ~RejectSigmaClip() {
  }

  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {
    typedef std::pair<Iterator1, Iterator2> IterPair;
    typedef ZipIterator<IterPair> ZIter;

    ZIter beg = make_zip_iterator(begin, weights);
    ZIter ned = make_zip_iterator(end, weights + (end - begin));

    int delta = 0;
    double c_mean = 0;
    double c_std = 0;
    size_t c_size = end - begin;
    do {
      std::pair<double, double> r = m_central(begin, begin + c_size, weights);
      c_mean = r.first;
      c_std = sqrt(r.second);

      ned = partition(beg, ned, InRangeFirst<Iterator1, Iterator2> (c_mean
          - c_std * m_low, c_mean + c_std * m_high));

      delta = c_size - (ned - beg);
      c_size = end - begin;

    } while (delta);
    *results[0] = c_mean;
    *results[1] = c_std;
    *results[2] = c_size;
  }
private:
  CentralTendency m_central;
  double m_low;
  double m_high;
};

// Adaptor
template<typename Iterator1, typename Iterator2, typename Result = double>
class CTW {
public:
  CTW(std::auto_ptr<CombineMethod> cm) :
    m_cm(cm) {
  }
  CTW(const CTW& a) {
    m_cm.reset(a.m_cm->clone());
  }
  std::pair<Result, Result> operator()(Iterator1 begin, Iterator1 end,
      Iterator2 weights) const {
    return m_cm->central_tendency(begin, end, weights);
  }
private:
  std::auto_ptr<CombineMethod> m_cm;
};

typedef CTW<DataIterator, WeightsIterator, ResultType> MyCTWType;

template<typename MRNT>
class RejectHV: public RejectMethod {
public:
  RejectHV(const MRNT& rn) :
    m_rn(rn) {
  }
  virtual ~RejectHV() {
  }
  inline virtual void combine(DataIterator begin, DataIterator end,
      WeightsIterator weights, ResultType* results[3]) const {
    m_rn.combine(begin, end, weights, results);
  }
private:
  MRNT m_rn;
};

} // namespace Numina

#endif // PYEMIR_REJECT_METHODS_H
