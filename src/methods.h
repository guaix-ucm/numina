/*
 * Copyright 2008-2009 Sergio Pascual
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

#ifndef PYEMIR_METHODS_H
#define PYEMIR_METHODS_H

#include <Python.h>

#include "method_base.h"

namespace Numina {

class MeanMethod: public Method {
public:
	MeanMethod(PyObject* args);
	virtual ~MeanMethod();
	virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
	int m_dof;
};


class MedianMethod: public Method {
public:
	MedianMethod();
	virtual ~MedianMethod();
	virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
	double* kth_smallest(double* data, size_t size, size_t kth) const;
};

class SigmaClipMethod: public Method {
public:
  SigmaClipMethod(PyObject* args);
  virtual ~SigmaClipMethod();
  virtual void run(double* data, double* weights, size_t size, double* results[3]) const;
private:
  double m_low;
  double m_high;
  int m_dof;
};


}

#endif // PYEMIR_METHODS_H
