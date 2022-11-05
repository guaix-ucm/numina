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

#ifndef NU_BASEITERATOR_H
#define NU_BASEITERATOR_H

#include <cstddef>

#include "base_crtp.h"

namespace Numina
{
namespace Detail
{
/**
 * \brief Base class for the iterator adaptors
 *
 * @ingroup iter
 */
template<typename Derived, typename ValueType, typename IteratorCategory,
    typename Reference = ValueType&, typename Pointer = Reference*,
    typename Difference = std::ptrdiff_t>
class BaseIterator: public BaseCRTP<BaseIterator<Derived, ValueType,
    IteratorCategory, Reference, Pointer, Difference> , Derived>
{
public:
  //! The value type of the sequence
  typedef ValueType value_type;
  //! The iterator category of the sequence
  typedef IteratorCategory iterator_category;
  //! The reference type of the sequence
  typedef Reference reference;
  //! The difference type of the sequence
  typedef Difference difference_type;
  //! The pointer type of  of the sequence
  typedef Pointer pointer;

  typedef BaseCRTP<BaseIterator<Derived, ValueType,
      IteratorCategory, Reference, Pointer, Difference> , Derived> Base;

  /**
   * Default constructor
   */
  BaseIterator()
  {
  }

  /**
   * Implementation of operator[] const
   * @param n Index of the acessed element
   * @return A constant reference to the element
   */
  const reference operator[](difference_type n) const
  {
    return *(&Base::derived(*this) + n);
  }
  /**
   * Implementation of operator[]
   * @param n Index of the acessed element
   * @return A reference to the element
   */
  reference operator[](difference_type n)
  {
    return *(&Base::derived(*this) + n);
  }

  /**
   * Implementation of operator*()
   * @return A reference to the pointed object
   */
  reference operator*()
  {
    return Base::derived(*this).dereference();
  }
  /**
   * Implementation of operator*() const
   * @return A const reference to the pointed object
   */
  const reference operator*() const
  {
    return Base::derived(*this).dereference();
  }

  /**
   * Implementation of operator->()
   * @return A pointer to the pointed object
   */
  pointer operator->()
  {
    return &Base::derived(*this).dereference();
  }

  /**
   * Implementation of operator->() const
   * @return A const pointer to the pointed object
   */
  const pointer operator->() const
  {
    return &Base::derived(*this).dereference();
  }

  /**
   * Implementation of operator++() (preincrement)
   * @return A reference to the object pointed by the next iterator
   */
  Derived& operator++()
  {
    Base::derived(*this).increment();
    return Base::derived(*this);
  }

  /**
   * Implementation of operator++(int) (posincrement)
   * @return The object pointed by the iterator before incrementing
   */
  Derived operator++(int)
  {
    Derived tmp = Base::derived(*this);
    ++Base::derived(*this);
    return tmp;
  }

  /**
   * Implementation of operator--() (predecrement)
   * @return A reference to the object pointed by the previous iterator
   */
  Derived& operator--()
  {
    Base::derived(*this).decrement();
    return Base::derived(*this);
  }
  /**
   * Implementation of operator--(int) (postdecrement)
   * @return The object pointed by the previous iterator
   */
  Derived operator--(int)
  {
    Derived tmp = Base::derived(*this);
    --Base::derived(*this);
    return tmp;
  }
  /**
   * Implementation of operator+=()
   * @param n The iterator now points n positions forward
   * @return A reference to the new iterator
   */
  Derived& operator+=(difference_type n)
  {
    Base::derived(*this).advance(n);
    return Base::derived(*this);
  }
  /**
   * Implementation of operator-=()
   * @param n The iterator now points n positions backward
   * @return A reference to the new iterator
   */
  Derived& operator-=(difference_type n)
  {
    Base::derived(*this).advance(-n);
    return Base::derived(*this);
  }
  /**
   * Implementation of operator+()
   * @param n The iterator now points n positions forward
   * @return A new iterator pointing n positions forward
   */
  Derived operator+(difference_type n) const
  {
    Derived tmp = Base::derived(*this);
    return tmp += n;
  }
  /**
   * Implementation of operator-() (unary)
   * @param n The iterator now points n positions backward
   * @return A new iterator pointing n positions backward
   */
  Derived operator-(difference_type n) const
  {
    Derived tmp = Base::derived(*this);
    return tmp -= n;
  }

  /**
   * Implementation of operator-() (binary)
   * @param rhs A iterator belonging to the same sequence
   * @return The distance between iterators
   */
  difference_type operator-(const Derived& rhs) const
  {
    return -Base::derived(*this).distance_to(rhs);
  }

  /**
   * Equallity operator. Returns true if the iterators point to the same
   * element in a sequence, false otherwise.
   * @param rhs A iterator
   * @return True if the two iterators point to the same element, false otherwise
   */
  bool operator==(const Derived& rhs) const
  {
    return Base::derived(*this).equal(rhs);
  }
  /**
   * Inequallity operator. Returns false if the iterators point to the same
   * element in a sequence, true otherwise.
   * @param rhs A iterator
   * @return False if the two iterators point to the same element, true otherwise
   */
  bool operator!=(const Derived& rhs) const
  {
    return !Base::derived(*this).equal(rhs);
  }

  /**
   * Less than operator. Returns true if the iterator rhs is located ahead
   * in the sequence than the iterator pointed by this
   * @param rhs A iterator
   * @return True is this is closer to the begining than rhs
   */
  bool operator<(const Derived& rhs) const
  {
    return (Base::derived(*this).distance_to(rhs) > 0);
  }
  /**
   * Less equal  operator. Returns true if the iterator rhs is closer (or at the same distance)
   * to the begining of the sequence than the iterator pointed by this
   * @param rhs A iterator
   * @return True is this is closer to the begining (or the same distance) than rhs
   */
  bool operator<=(const Derived& rhs) const
  {
    return (Base::derived(*this).distance_to(rhs) >= 0);
  }
  /**
   * Greater than operator. Returns true if the iterator rhs is located before
   * in the sequence than the iterator pointed by this
   * @param rhs A iterator
   * @return True is this is farther to the begining than rhs
   */
  bool operator>(const Derived& rhs) const
  {
    return (Base::derived(*this).distance_to(rhs) < 0);
  }
  /**
   * Greater than equal operator. Returns true if the iterator rhs is located before
   * (or in the same position) in the sequence than the iterator pointed by this
   * @param rhs A iterator
   * @return True is this is farther to the begining of the sequence (or in the same position) than rhs
   */
  bool operator>=(const Derived& rhs) const
  {
    return (Base::derived(*this).distance_to(rhs) <= 0);
  }
};

} // namespace Detail

} // namespace Numina


#endif // NU_BASEITERATOR_H
