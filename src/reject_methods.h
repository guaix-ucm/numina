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

#ifndef PYEMIR_REJECT_METHODS_H
#define PYEMIR_REJECT_METHODS_H

#include <memory>
#include <ext/functional>

#include "functional.h"
#include "method_base.h"
#include "operations.h"
#include "zip_iterator.h"

namespace Numina {

template<typename CentralTendency>
class RejectNone {
public:
  RejectNone(const CentralTendency& central) :
    m_central(central) {
  }
  template<typename Iterator1, typename Iterator2, typename Result>
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

template<typename CentralTendency>
class RejectMinMax {
public:
  RejectMinMax(const CentralTendency& central, size_t nmin, size_t nmax) :
    m_central(central), m_nmin(nmin), m_nmax(nmax) {
  }
  template<typename Iterator1, typename Iterator2, typename Result>
  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {
    typedef std::pair<Iterator1, Iterator2> IterPair;
    typedef ZipIterator<IterPair> ZIter;
    typedef std::pair<ZIter, ZIter> ZIterPair;

    ZIterPair result = reject_min_max(make_zip_iterator(begin, weights),
        make_zip_iterator(end, weights + (end - begin)), m_nmin, m_nmax,
        // Compares two std::pair objects. Returns true
        // if the first component of the first is less than the first component
        // of the second std::pair
        compose(std::less<typename Iterator1::value_type>(), __gnu_cxx::select1st<
            typename ZIter::value_type>(), __gnu_cxx::select1st<
            typename ZIter::value_type>()));

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

template<typename CentralTendency>
class RejectSigmaClip {
public:
  RejectSigmaClip(const CentralTendency& central, double low, double high) :
    m_central(central), m_low(low), m_high(high) {
  }

  virtual ~RejectSigmaClip() {
  }
  template<typename Iterator1, typename Iterator2, typename Result>
  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {
    typedef std::pair<Iterator1, Iterator2> IterPair;
    typedef ZipIterator<IterPair> ZIter;

    size_t c_size = std::distance(begin, end);
    ZIter beg = make_zip_iterator(begin, weights);
    ZIter ned = make_zip_iterator(end, weights + c_size);

    double c_mean = 0.0;
    double c_std = 0.0;
    size_t nc_size = c_size;

    do {
      std::pair<double, double> r = m_central(begin, begin + nc_size, weights);

      c_mean = r.first;
      c_std = sqrt(r.second);
      c_size = std::distance(beg, ned);

      const double low = c_mean - c_std * m_low;
      const double high = c_mean + c_std * m_high;
      ned = partition(beg, ned,
          // Checks if first component of a std::pair
          // is inside the range (low, high)
          // equivalent to return (low < x.first) && (high > x.first);
          __gnu_cxx::compose2(std::logical_and<bool>(),
              __gnu_cxx::compose1(
                  std::bind1st(std::less<double>(), low),
                  __gnu_cxx::select1st<typename ZIter::value_type>()),
              __gnu_cxx::compose1(
                  std::bind1st(std::greater<double>(), high),
                  __gnu_cxx::select1st<typename ZIter::value_type>())
            )
          );

      nc_size = std::distance(beg, ned);
      // We stop when std == 0, all the points would be reject otherwise
      // or when no points are removed in the iteration
    } while (c_std > 0 && (nc_size != c_size));

    *results[0] = c_mean;
    *results[1] = c_std;
    *results[2] = c_size;
  }
private:
  CentralTendency m_central;
  double m_low;
  double m_high;
};

template<typename CentralTendency>
class RejectQuantileClip {
public:
  RejectQuantileClip(const CentralTendency& central, double fclip) :
    m_central(central), m_fclip(fclip) {
  }

  virtual ~RejectQuantileClip() {
  }

  template<typename Iterator1, typename Iterator2, typename Result>
  void combine(Iterator1 begin, Iterator1 end, Iterator2 weights,
      Result* results[3]) const {


    size_t n_elem = std::distance(begin, end);
    double n_elem_to_clip = n_elem * m_fclip;
    size_t nclip = static_cast<size_t>(floor(n_elem_to_clip));
    size_t mclip = static_cast<size_t>(ceil(n_elem_to_clip));

    if (nclip == mclip) {
      // No interpolation
      std::pair<double, double> r = central(begin, end, weights, nclip);
      *results[0] = r.first;
      *results[1] = r.second;
      *results[2] = n_elem - nclip;
    }
    else {
      // Interpolation
      std::pair<double, double> r1 = central(begin, end, weights, nclip);
      std::pair<double, double> r2 = central(begin, end, weights, mclip);
      *results[0] = r1.first + (n_elem_to_clip - nclip) * (r2.first - r1.first);
      *results[1] = r1.second + (n_elem_to_clip - nclip) * (r2.second - r1.second);
      *results[2] = n_elem - n_elem_to_clip;
    }
  }

  template<typename Iterator1, typename Iterator2>
  std::pair<double, double> central(Iterator1 begin, Iterator1 end, Iterator2 weights, size_t nclip) const {
    typedef std::pair<Iterator1, Iterator2> IterPair;
    typedef ZipIterator<IterPair> ZIter;
    typedef std::pair<ZIter, ZIter> ZIterPair;

    ZIter beg = make_zip_iterator(begin, weights);
    size_t n_elem = std::distance(begin, end);
    ZIter ned = make_zip_iterator(end, weights + n_elem);

    ZIterPair result = reject_min_max(beg, ned, nclip, nclip,
        // Compares two std::pair objects. Returns true
        // if the first component of the first is less than the first component
        // of the second std::pair
        compose(std::less<typename Iterator1::value_type>(), __gnu_cxx::select1st<
            typename ZIter::value_type>(), __gnu_cxx::select1st<
            typename ZIter::value_type>()));

    IterPair itp_beg = result.first.get_iterator_pair();
    IterPair itp_end = result.second.get_iterator_pair();

    return m_central(itp_beg.first, itp_end.first, itp_beg.second);
  }


private:
  CentralTendency m_central;
  double m_fclip;
};

// Adaptor
class CTW {
public:
  CTW(std::auto_ptr<CombineMethod> cm) :
    m_cm(cm) {
  }
  CTW(const CTW& a) :
    m_cm(a.m_cm) {
  }
  template<typename Iterator1, typename Iterator2>
  std::pair<ResultType, ResultType> operator()(Iterator1 begin, Iterator1 end,
      Iterator2 weights) const {
    return m_cm->central_tendency(begin, end, weights);
  }
private:
  mutable std::auto_ptr<CombineMethod> m_cm;
};

template<typename MRNT>
class RejectMethodAdaptor: public RejectMethod {
public:
  RejectMethodAdaptor(const MRNT& rn) :
    m_rn(rn) {
  }
  virtual ~RejectMethodAdaptor() {
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
