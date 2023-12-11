/*
 * Copyright 2008-2014 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
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
