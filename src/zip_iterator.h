/*
 * Copyright 2008-2016 Universidad Complutense de Madrid
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

#ifndef NU_ZIP_ITERATOR_H
#define NU_ZIP_ITERATOR_H

#include <utility>

#include "base_iterator.h"

namespace Numina
{
namespace Detail
{
//! A proxy class to access by reference the values pointed by ZipIterator
/**
 * This class is used to access by reference the elements pointed by the individual iterators
 * inside a ZipIterator. This class is convertible to std::pair
 */
template<typename IteratorPair>
class ProxyPairRef
{
public:
  //! The type of the first iterator in IteratorPair
  typedef typename IteratorPair::first_type first_iterator_type;
  //! The type of the second iterator in IteratorPair
  typedef typename IteratorPair::second_type second_iterator_type;
  //! The value_type of the first iterator in IteratorPair
  typedef typename std::iterator_traits<first_iterator_type>::value_type
      first_value_type;
  //! The value_type of the second iterator in IteratorPair
  typedef typename std::iterator_traits<second_iterator_type>::value_type
      second_value_type;
  //! Type of the first member
  typedef first_value_type T1;
  //! Type of the second member
  typedef second_value_type T2;
  //! Type of the std::pair with the same types in the first and second member
  typedef std::pair<T1, T2> value_type;

  //! Empty constructor
  ProxyPairRef() :
    m_first(), m_second()
  {
  }

  //! Constructor from two references
  ProxyPairRef(T1& a, T2& b) :
    m_first(&a), m_second(&b)
  {
  }



  //! Assignment operator
  ProxyPairRef& operator=(const ProxyPairRef& b)
  {
    if (this == &b)
      return *this;
    *m_first = *b.m_first;
    *m_second = *b.m_second;
    return *this;
  }

  //! Asignament operator
  ProxyPairRef& operator=(const value_type& b)
  {
    *m_first = b.first;
    *m_second = b.second;
    return *this;
  }

  //! Cast operator to value_type, i.e. std::pair
  operator value_type() const
  {
    return std::make_pair(*m_first, *m_second);
  }
  //! This operator checks the equality between two objects of the class
  bool operator==(const ProxyPairRef& b) const
  {
    return ((*m_first == *b.m_first) && (*m_second == *b.m_second));
  }

  //! This operator checks the inequality between two objects of the class
  bool operator!=(const ProxyPairRef& b) const
  {
    return ((*m_first != *b.m_first) || (*m_second != *b.m_second));
  }

  //! Constant accessor to first member
  const T1& first() const
  {
    return *m_first;
  }
  /**
   * A member setter. It updates the value of the first member of ProxyPairRef
   * @param[in] a New value for the first member of ProxyPairRef
   */
  void set_first(T1& a)
  {
    m_first = &a;
  }
  //! Accessor to first member
  T1& first()
  {
    return *m_first;
  }
  //! Constant accessor to second member
  const T2& second() const
  {
    return *m_second;
  }
  //! Accessor to second member
  T2& second()
  {
    return *m_second;
  }
  /**
   * A member setter. It updates the value of the second member of ProxyPairRef
   * @param[in] b New value for the second member of ProxyPairRef
   */
  void set_second(T2& b)
  {
    m_second = &b;
  }

  /**
   * Internal swap method. It is used by the swap function in the namespace
   */
  void swap(ProxyPairRef& b) {
    std::swap(*m_first, *b.m_first);
    std::swap(*m_second, *b.m_second);
  }
private:

  T1* m_first;
  T2* m_second;
};

/**
 * Swap two values. It is required by nth_element (at least in gcc >= 6 and LLVM)
 */
template<typename IteratorPair>
void swap(ProxyPairRef<IteratorPair>& ppr1, ProxyPairRef<IteratorPair>& ppr2) {
    ppr1.swap(ppr2);
}

/**
 * \brief Types used to help build ZipIterator
 */
template<typename IteratorPair>
struct ZipIteratorTypes
{
  //! The type of the first iterator in IteratorPair
  typedef typename IteratorPair::first_type first_iterator_type;
  //! The type of the second iterator in IteratorPair
  typedef typename IteratorPair::second_type second_iterator_type;
  //! The value_type of the first iterator in IteratorPair
  typedef typename std::iterator_traits<first_iterator_type>::value_type
      first_value_type;
  //! The value_type of the second iterator in IteratorPair
  typedef typename std::iterator_traits<second_iterator_type>::value_type
      second_value_type;
  //! A proxy to the values stored in the two sequences.
  /**
   * A proxy class to access to and to modify the values stored in the two parallel sequences. It is convertible to
   * value_type
   */
  typedef Detail::ProxyPairRef<IteratorPair>
      reference_wrapper;

  //! The reference type of the sequence
  typedef reference_wrapper& reference;
  //! The value type of the sequence
  typedef std::pair<first_value_type, second_value_type> value_type;
  //! The pointer type of  of the sequence
  typedef reference_wrapper* pointer;
  //! The difference type of the sequence
  typedef typename std::iterator_traits<first_iterator_type>::difference_type
      difference_type;
  //! The iterator category of the sequence
  typedef typename std::iterator_traits<second_iterator_type>::iterator_category
      iterator_category;
};

}

/**
 * \brief Iterator adaptor to iterate over parallel sequences
 *
 *
 * This iterator adapter can be used to iterate over two parallel sequences. It is based on boost::zip_iterator, but uses
 * a different technique to return reference values, and can be used only with two sequences.
 * IteratorPair is supposed to be std::pair<Iterator1,Iterator2>.
 *
 * \b Example:
 *
 * \code
 * using Numina::make_zip_iterator;
 * std::vector<double> data;
 * std::vector<double> error;
 * for(int i = 10; i > 0; ++i
 * {
 *  data.push_back(i);
 *  error.push_back(0.001*i);
 * }
 * std::sort(make_zip_iterator(data.begin(),error.begin()),make_zip_iterator(data.end(),error.end()));
 * \endcode
 *
 * \ingroup iter
 */

template<typename IteratorPair>
class ZipIterator: public Detail::BaseIterator<ZipIterator<IteratorPair> ,
    typename Detail::ZipIteratorTypes<IteratorPair>::value_type,
    typename Detail::ZipIteratorTypes<IteratorPair>::iterator_category,
    typename Detail::ZipIteratorTypes<IteratorPair>::reference,
    typename Detail::ZipIteratorTypes<IteratorPair>::pointer>
{
public:
  //! Empty constructor
  /**
   * Note that an empty constructed iterator is not valid until it points to valid data.
   */

  ZipIterator()
  {
  }

  //! Constructor from a pair of iterators
  /**
   * This is a constructor used to build the iterator from a pair of valid iterators to parallel sequences
   * @param iterator_tuple A std::pair of iterators
   */
  ZipIterator(IteratorPair iterator_pair) :
    m_iterator_pair(iterator_pair)
  {
    update_internal_ref();
  }

  //! Copy constructor
  /**
   * The copy constructor can be used to copy any valid iterator
   * @param b A ZipIterator to be copied
   */
  ZipIterator(const ZipIterator& rhs) :
    m_iterator_pair(rhs.m_iterator_pair), m_internal_ref(rhs.m_internal_ref)
  {
    update_internal_ref();
  }

  //!Asignation operator
  ZipIterator& operator=(const ZipIterator& rhs)
  {
    if (&rhs == this)
      return *this;
    m_iterator_pair = rhs.m_iterator_pair;
    update_internal_ref();
    m_internal_ref = rhs.m_internal_ref;
    return *this;
  }

  //! Constant accessor to the internal iterator pair
  const IteratorPair get_iterator_pair() const
  {
    return m_iterator_pair;
  }

  //! Accessor to the internal iterator pair
  IteratorPair get_iterator_pair()
  {
    return m_iterator_pair;
  }

  //!Used to implement the different decrement operators in the base class
  void decrement()
  {
    --m_iterator_pair.first;
    --m_iterator_pair.second;
    update_internal_ref();
  }

  //!Used to implement the different increment operators in the base class
  void increment()
  {
    ++m_iterator_pair.first;
    ++m_iterator_pair.second;
    update_internal_ref();
  }

  //! Used to implement the different addition and subtraction operators in the base class
  //void advance(const difference_type n)
  void advance(const std::ptrdiff_t n)
  {
    std::advance(m_iterator_pair.first, n);
    std::advance(m_iterator_pair.second, n);
    update_internal_ref();
  }

  /**
   * Used to implement operator*() and operator->() in the base class
   * @return A reference to the pointed object
   */
  typename ZipIterator::reference dereference()
  {
    return m_internal_ref;
  }

  /**
   * Used to implement operator*() and operator->() in the base class
   * @return A constant reference to the pointed object
   */
  const typename ZipIterator::reference dereference() const
  {
    return m_internal_ref;
  }

  /**
   * Used to implement the difference between iterators and
   * ordering operators in the base class
   */
  typename ZipIterator::difference_type distance_to(const ZipIterator& b) const
  {
    return std::distance(m_iterator_pair.first, b.m_iterator_pair.first);
  }

  /**
   * Used to implement operator==() and operator!=() in the base class
   * @return True if the rhs iterator points to the same element than this
   */
  bool equal(const ZipIterator& b) const
  {
    return (m_iterator_pair == b.m_iterator_pair);
  }

private:
  IteratorPair m_iterator_pair;
  typedef typename Detail::ZipIteratorTypes<IteratorPair>::reference_wrapper
      ReferenceWrapper;
  ReferenceWrapper m_internal_ref;

  void update_internal_ref()
  {
    m_internal_ref.set_first(*m_iterator_pair.first);
    m_internal_ref.set_second(*m_iterator_pair.second);
  }
};

/**
 * \brief Helper function to build a ZipIterator from a std::pair of iterators
 * @param iterpair A std::pair of iterators
 * @return A constructed ZipIterator
 * \ingroup iter
 */
template<typename IteratorPair>
ZipIterator<IteratorPair> make_zip_iterator(IteratorPair iterpair)
{
  return ZipIterator<IteratorPair> (iterpair);
}
/**
 *  \brief Helper function to build a ZipIterator from two iterators
 * @param iter1 A iterator of type Iterator1
 * @param iter2 A iterator of type Iterator2
 * @return A constructed ZipIterator
 * \ingroup iter
 */
template<typename Iterator1, typename Iterator2>
ZipIterator<std::pair<Iterator1, Iterator2> > make_zip_iterator(
    const Iterator1& iter1, const Iterator2& iter2)
{
  typedef std::pair<Iterator1, Iterator2> IteratorTuple;
  return ZipIterator<IteratorTuple> (std::make_pair(iter1, iter2));
}

} // namespace Numina

#endif // NU_ZIP_ITERATOR_H
