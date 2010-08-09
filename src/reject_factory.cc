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


#include "method_exception.h"
#include "reject_factory.h"
#include "reject_methods.h"

namespace Numina {

auto_ptr<RejectMethod>
RejectMethodFactory::create(const std::string& name,
		PyObject* args,
		auto_ptr<CombineMethod> combine_method) {
  const MyCTWType cm(combine_method);
	if (name == "none") {
	  return auto_ptr<RejectMethod>(new
	      RejectHV<RejectNone<MyCTWType, DataIterator, WeightsIterator, ResultType> >(cm));
	}
	if (name == "minmax") {
		unsigned int nmin = 0;
		unsigned int nmax = 0;
		if (not PyArg_ParseTuple(args, "II", &nmin, &nmax))
	    	throw MethodException("problem creating MinMax");
		const RejectMinMax<MyCTWType, DataIterator, WeightsIterator, ResultType> aa(cm, nmin, nmax);
		return auto_ptr<RejectMethod>(new
        RejectHV<RejectMinMax<MyCTWType, DataIterator, WeightsIterator, ResultType> >(aa));
	}
	if (name == "sigmaclip") {
		double low = 0.0;
		double high = 0.0;
		if (not PyArg_ParseTuple(args, "dd", &low, &high))
			throw MethodException("problem creating SigmaClipMethod");
		const RejectSigmaClip<MyCTWType, DataIterator, WeightsIterator, ResultType> aa(cm, low, high);
    return auto_ptr<RejectMethod>(new
        RejectHV<RejectSigmaClip<MyCTWType, DataIterator, WeightsIterator, ResultType> >(aa));
	}
	return auto_ptr<RejectMethod>();
}


} // namespace Numina

