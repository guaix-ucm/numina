/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
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

#ifndef NU_BASE_CRTP_H
#define NU_BASE_CRTP_H

/**
 * @file
 * @brief Base class for the <A HREF="http://en.wikipedia.org/wiki/Curiously_Recurring_Template_Pattern">CRTP</A> (Curious Recursive Template pattern)
 *
 * The curiously recurring template pattern (CRTP) is a C++ idiom in which a class X derives from a
 * class template instantiation using X itself as template argument.
 */

namespace Numina
{

namespace Detail
{
/// Base class for the Curious Recursive Template pattern
template<typename Base, typename Derived>
struct BaseCRTP
{
  /** Transforms a reference to Base into a reference to Derived
   * @param myself A reference to Base object
   * @return A reference to Derived
   */
  static Derived& derived(Base& myself)
  {
    return *static_cast<Derived*> (&myself);
  }

  /** Transforms a const reference to Base into a const reference to Derived
   * @param myself A const reference to Base object
   * @return A const reference to Derived
   */
  static Derived const & derived(Base const& myself)
  {
    return *static_cast<Derived const*> (&myself);
  }
};
} // namespace Detail

} // namespace Numina

#endif // NU_BASE_CRTP_H
