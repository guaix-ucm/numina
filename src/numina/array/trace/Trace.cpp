/*
 * Copyright 2015 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#include <vector>
#include <cstddef>
#include <algorithm>

#include "Trace.h"

#include "fitter.h"

namespace Numina {

  InternalTrace::InternalTrace() 
  {}
  
  void InternalTrace::push_back(double x, double y, double p) {
    xtrace.push_back(x);
    ytrace.push_back(y);
    ptrace.push_back(p);
  }

  void InternalTrace::reverse() {
    std::reverse(xtrace.begin(), xtrace.end());
    std::reverse(ytrace.begin(), ytrace.end());
    std::reverse(ptrace.begin(), ptrace.end());
  }

  double InternalTrace::predict(double x) const {

    size_t n = std::min<size_t>(5, xtrace.size());
    Numina::LinearFit mm = Numina::linear_fitter(xtrace.end() - n, xtrace.end(), ytrace.end() - n, ytrace.end());
    return mm.slope * x + mm.intercept;
  }

} // namespace numina
