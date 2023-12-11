/*
 * Copyright 2015 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_TRACE_H
#define NU_TRACE_H


#include <vector>

namespace Numina {

class InternalTrace {
  public:
    InternalTrace();
    double predict(double x) const;
    void push_back(double x, double y, double p);
    void reverse();
    std::vector<double> xtrace;
    std::vector<double> ytrace;
    std::vector<double> ptrace;
  };

} // namespace numina

#endif // NU_TRACE_H
