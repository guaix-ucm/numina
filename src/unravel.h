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

/* $Id$ */

#ifndef PYEMIR_UNRAVEL_H
#define PYEMIR_UNRAVEL_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

namespace Numina {

  class UnRavel
  {
    public:
      UnRavel(npy_intp dims[], npy_intp size) :
        m_size(size), m_dims(dims, dims + size), m_help(size)
      {
        std::vector<npy_intp> vdims(m_dims.begin(), m_dims.end());
        vdims.push_back(1L);
        std::partial_sum(vdims.rbegin(), vdims.rend() - 1, vdims.rbegin(),
            std::multiplies<npy_intp>());
        std::copy(vdims.begin() + 1, vdims.end(), m_help.begin());
      }

      std::vector<npy_intp> index(npy_intp i) const
      {
        std::vector<npy_intp> buffer(m_size);
        std::transform(m_help.begin(), m_help.end(), m_dims.begin(),
            buffer.begin(), _Helper(i));

        return buffer;
      }

      void index_copy(npy_intp i, npy_intp coor[], npy_intp size) const
      {
        std::transform(m_help.begin(), m_help.end(), m_dims.begin(), coor,
            _Helper(i));
      }

    private:

      class _Helper
      {
        public:
          _Helper(npy_intp i) :
            m_i(i)
          {
          }
          inline npy_intp operator()(npy_intp x, npy_intp y)
          {
            return (m_i / x) % y;
          }
        private:
          npy_intp m_i;
      };

      npy_intp m_size;
      std::vector<npy_intp> m_dims;
      std::vector<npy_intp> m_help;
  };


} // namespace Numina

#endif // PYEMIR_UNRAVEL_H
