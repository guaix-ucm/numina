/*
 * Copyright 2008 Sergio Pascual
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

typedef void (*GenericMethodPtr)(double data[], int size, double* c, double* var, long* number, void* params);

typedef struct {
  const char* name;
  GenericMethodPtr function;
} MethodStruct;

void method_mean(double data[], int size, double* c, double* var, long* number, void* params);
void method_median(double data[], int size, double* c, double* var, long* number, void* params);


static MethodStruct methods[] = {
    {"mean",method_mean},
    {"median",method_median},
    {NULL, NULL}
};

