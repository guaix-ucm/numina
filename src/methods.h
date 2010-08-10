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

#ifndef PYEMIR_METHODS_H
#define PYEMIR_METHODS_H

#include "method_base.h"
#include "operations.h"

namespace Numina {

template<typename MCNT>
class CombineHV: public CombineMethod {
public:
  CombineHV(const MCNT& cn = MCNT()) :
    m_cn(cn) {
  }
  virtual ~CombineHV() {
  }
  inline virtual CombineHV<MCNT>* clone() const {
    return new CombineHV<MCNT>(*this);
  }
  inline virtual std::pair<double, double> central_tendency(DataIterator begin,
      DataIterator end, WeightsIterator weights) const {
    return m_cn(begin, end, weights);
  }
private:
  MCNT m_cn;
};


struct MethodAverage {
  template<typename Iterator1, typename Iterator2>
  inline std::pair<double, double> operator()(Iterator1 begin, Iterator1 end,
      Iterator2 weights) const {
    return average_central_tendency(begin, end, weights);
  }
};


struct MethodMedian {
  template<typename Iterator1, typename Iterator2>
  inline std::pair<double, double> operator()(Iterator1 begin, Iterator1 end,
      Iterator2 weights) const {
    return median_central_tendency(begin, end, weights);
  }
};

} // namespace Numina

#endif // PYEMIR_METHODS_H
