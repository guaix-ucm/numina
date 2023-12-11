/*
 * Copyright 2008-2018 Universidad Complutense de Madrid
 *
 * This file is part of Numina
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * License-Filename: LICENSE.txt
 *
 */

#ifndef NU_COMBINE_H
#define NU_COMBINE_H

#include "nu_combine_defs.h"

int NU_generic_combine(PyObject** images, PyObject** masks, size_t size,
    PyObject* out[NU_COMBINE_OUTDIM],
    CombineFunc function,
    void* vdata,
    const double* zeros,
    const double* scales,
    const double* weights);

bool NU_combine_image_check(PyObject* exception,
     PyObject* image,
     PyObject* ref,
     PyObject* typeref,
     const char* name,
     size_t index);

#endif // NU_COMBINE_H
