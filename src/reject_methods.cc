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

namespace Numina
{

NoneReject::NoneReject(auto_ptr<CombineMethod> combine) :
  RejectMethod(combine)
{
}

NoneReject::~NoneReject()
{
}

void NoneReject::run(double* data, double* weights, size_t size,
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

void MinMax::run(double* data, double* weights, size_t size, double* results[3]) const
{

  std::pair<double*, double*> result = reject_min_max(data, data + size,
      m_nmin, m_nmax);
  *results[2] = result.second - result.first;
  central_tendency(result.first, weights, *results[2], results[0], results[1]);
}

SigmaClipMethod::SigmaClipMethod(auto_ptr<CombineMethod> combine, double low,
    double high) :
  RejectMethod(combine), m_low(low), m_high(high)
{
}

SigmaClipMethod::~SigmaClipMethod()
{
}

void SigmaClipMethod::run(double* data, double* weights, size_t size,
    double* results[3]) const
{

  int delta = 0;
  double c_mean = 0;
  double c_std = 0;
  int c_size = size;
  do
  {
    central_tendency(data, weights, size, &c_mean, &c_std);
    c_std = sqrt(c_std);

    double* end = std::partition(data, data + c_size, __gnu_cxx::compose2(
        std::logical_and<bool>(), std::bind2nd(std::greater<double>(), c_mean
            - c_std * m_low), std::bind2nd(std::less<double>(), c_mean + c_std
            * m_high)));
    delta = c_size - (end - data);
    c_size = (end - data);
  } while (delta);
  *results[0] = c_mean;
  *results[1] = c_std;
  *results[2] = c_size;
}

} // namespace Numina
