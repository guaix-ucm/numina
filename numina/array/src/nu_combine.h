/*
 * Copyright 2008-2018 Universidad Complutense de Madrid
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
