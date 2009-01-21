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

void method_mean(double data[], int size, double* c, double* var, long* number, void* params)
{
  if (size == 0)
  {
    *c = *var = 0.0;
    *number = 0;
    return;
  }

  if (size == 1)
  {
    *c = data[0];
    *var = 0.0;
    *number = 1;
    return;
  }

  double sum = 0.0;
  double sum2 = 0.0;
  int i;
  for (i = 0; i < size; ++i)
  {
    sum += data[i];
    sum2 += data[i] * data[i];
  }

  *c = sum / size;
  *number = size;
  *var = sum2 / (size - 1) - (sum * sum) / (size * (size - 1));
}

/*
 * Algorithm from N. Wirth's book, implementation by N. Devillard.
 * This code in public domain.
 * http://ndevilla.free.fr/median/median/index.html
 */
typedef double elem_type ;

#define WIRTH_ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }

elem_type kth_smallest(elem_type a[], int n, int k)
{
    register int i,j,l,m ;
    register elem_type x ;

    l=0 ; m=n-1 ;
    while (l<m) {
        x=a[k] ;
        i=l ;
        j=m ;
        do {
            while (a[i]<x) i++ ;
            while (x<a[j]) j-- ;
            if (i<=j) {
                WIRTH_ELEM_SWAP(a[i],a[j]) ;
                i++ ; j-- ;
            }
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
    }
    return a[k] ;
}

void method_median(double data[], int size, double* c, double* var, long* number, void* params) {
  *c = kth_smallest(data, size, ((size & 1) ? (size / 2) : ((size / 2) - 1)));
  *var = 0.0;
  *number = size;
}

