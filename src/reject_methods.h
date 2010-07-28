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

namespace Numina {

class NoneReject: public RejectMethod {
public:
	NoneReject(auto_ptr<CombineMethod> combine_method);
	virtual ~NoneReject();
	virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
};

class MinMax: public RejectMethod {
public:
  MinMax(auto_ptr<CombineMethod> combine_method, unsigned int nmin, unsigned int nmax);
  virtual ~MinMax();
  virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
  unsigned int m_nmin;
  unsigned int m_nmax;
};


class SigmaClipMethod: public RejectMethod {
public:
  SigmaClipMethod(auto_ptr<CombineMethod> combine_method, double low, double high);
  virtual ~SigmaClipMethod();
  virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
  double m_low;
  double m_high;
};

}

#endif // PYEMIR_METHODS_H
