/*
 * Copyright 2010-2014 Universidad Complutense de Madrid
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

#ifndef NU_FUNCTIONAL_H
#define NU_FUNCTIONAL_H

#include <functional>

namespace Numina {

template<class F1, class F2, class F3>
class Compose: public std::binary_function<typename F2::argument_type,
    typename F3::argument_type, typename F1::result_type> {
public:
  Compose(const F1& f1, const F2& f2, const F3& f3) :
    m_f1(f1), m_f2(f2), m_f3(f3) {
  }

  typename Compose::result_type operator()(
      typename Compose::first_argument_type a,
      typename Compose::second_argument_type b) const {
    return m_f1(m_f2(a), m_f3(b));
  }
private:
  F1 m_f1;
  F2 m_f2;
  F3 m_f3;
};

template<class F1, class F2, class F3>
Compose<F1, F2, F3> compose(const F1& f1, const F2& f2, const F3& f3) {
  return Compose<F1, F2, F3> (f1, f2, f3);
}

} // namespace Numina

#endif // NU_FUNCTIONAL_H
