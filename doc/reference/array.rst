==========================================
:mod:`numina.array` --- Array manipulation
==========================================

.. automodule:: numina.array
   :synopsis: Array manipulation
   :members:

.. py:function:: process_ramp(inp[, out=None, axis=2, ron=0.0, gain=1.0, nsig=4.0, dt=1.0, saturation=65631])

   .. versionadded:: 0.8.2

   Compute the result 2d array computing slopes in a 3d array or ramp.

   :param inp: input array
   :param out: output array
   :param axis: unused
   :param ron: readout noise of the detector
   :param gain: gain of the detector
   :param nsig: rejection level to detect glitched and cosmic rays
   :param dt: time interval between exposures
   :param saturation: saturation level
   :return: a 2d array




:mod:`numina.array.background` --- Background estimation
========================================================

.. automodule:: numina.array.background
   :members:
   
:mod:`numina.array.blocks` --- Generation of blocks
====================================================

.. automodule:: numina.array.blocks
   :members:  
 
:mod:`numina.array.bpm` --- Bad Pixel Mask interpolation
========================================================

.. automodule:: numina.array.bpm
   :members:


:mod:`numina.array.combine` --- Array combination
=================================================

.. automodule:: numina.array.combine
   :synopsis: Array combination
   :members:
   
   
Combination methods in :mod:`numina.array.combine`
==================================================
All these functions return a :class:`PyCapsule`, that 
can be passed to :func:`generic_combine`
   
   
.. py:function:: mean_method()

   Mean method

.. py:function:: median_method()

   Median method

.. py:function:: sigmaclip_method([low=0.0[, high=0.0]])

   Sigmaclip method
   
   :param low: Number of sigmas to reject under the mean
   :param high: Number of sigmas to reject over the mean
   :raises: :class:`ValueError` if **low** or **high** are negative
 
.. py:function:: quantileclip_method([fclip=0.0])

   Quantile clip method
   
   :param fclip: Fraction of points to reject on both ends
   :raises: :class:`ValueError` if **fclip** is negative or greater than 0.4
 
.. py:function:: minmax_method([nmin=0[, nmax=0]])

   Min-max method

   :param nmin: Number of minimum points to reject
   :param nmax: Number of maximum points to reject
   :raises: :class:`ValueError` if **nmin** or **nmax** are negative
   
   
Extending :func:`generic_combine`
=================================
 
New combination methods can be implemented and used by :func:`generic_combine`
The combine function expects a :class:`PyCapsule` object containing a pointer
to a C function implementing the combination method.

.. c:function:: int combine(double *data, double *weights, size_t size, double *out[3], void *func_data)

   Operate on two arrays, containing **data** and **weights**. The result, its variance and the number of points
   used in the calculation (useful when there is some kind of rejection) are stored in **out[0]**, 
   **out[1]**  and **out[2]**.

   :param data: a pointer to an array containing the data 
   :param weights: a pointer to an array containing weights
   :param size: the size of data and weights
   :param out: an array of pointers to the pixels in the result arrays
   :param func_data: additional parameters of the function encoded as a void pointer
   :return: 1 if operation succeeded, 0 in case of error.
   
 
If the function uses dynamically allocated data stored in *func_data*, we must also
implement a function that deallocates the data once it is used. 
 
.. c:function:: void destructor_function(PyObject* cobject)

   :param cobject: the object owning dynamically allocated data
   
 
Simple combine method
---------------------
 
As an example, I'm going to implement a combination method that returns the minimum
of the input arrays. Let's call the method `min_method`

First, we implement the C function. I'm going to use some C++ here (it makes the code
very simple).

.. code-block:: c++

   int min_combine(double *data, double *weights, size_t size, double *out[3], 
            void *func_data) {
                        
       double* res = std::min_element(data, data + size);

       *out[0] = *res; 
       // I'm not going to compute the variance for the minimum
       // but it should go here
       *out[1] = 0.0;
       *out[2] = size;

       return 1;   
   }

A destructor function is not needed in this case as we are not using *func_data*. 

The next step is to build a Python extension. First we need to create a function
returning the :class:`PyCapsule` in C code like this:


.. code-block:: c
   
   static PyObject *
   py_method_min(PyObject *obj, PyObject *args) {
     if (not PyArg_ParseTuple(args, "")) {
       PyErr_SetString(PyExc_RuntimeError, "invalid parameters");
       return NULL;
     }
     return PyCapsule_New((void*)min_function, "numina.cmethod", NULL);
   }

The string ``"numina.cmethod"`` is the name of the :class:`PyCapsule`. It cannot be loadded
unless it is the name expected by the C code.

The code to load it in a module is like this:

.. code-block:: c

   static PyMethodDef mymod_methods[] = {
    {"min_combine", (PyCFunction) py_method_min, METH_VARARGS, "Minimum method."},
    ...,
    { NULL, NULL, 0, NULL } /* sentinel */
   };

   PyMODINIT_FUNC 
   init_mymodule(void)
   {
     PyObject *m;
     m = Py_InitModule("_mymodule", mymod_methods);
   }

When compiled, this code created a file `_mymodule.so` that can be loaded by the 
Python interpreter. This module will contain, among others, a `min_combine` function.

    >>> from _mymodule import min_combine
    >>> method = min_combine()
    ...
    >>> o = generic_combine(method, arrays)


A combine method with parameters
--------------------------------
A combine method with parameters follow a similar approach. Let's say we want
to implement a sigma-clipping method. We need to pass the function a *low* and
a *high* rejection limits. Both numbers are real numbers greater than zero.

First, the Python function. I'm skipping error checking code hre.

.. code-block:: c

   static PyObject *
   py_method_sigmaclip(PyObject *obj, PyObject *args) {
      double low = 0.0;
      double high = 0.0;
      PyObject *cap = NULL;

      if (!PyArg_ParseTuple(args, "dd", &low, &high)) {
         PyErr_SetString(PyExc_RuntimeError, "invalid parameters");
         return NULL;
      }

      cap = PyCapsule_New((void*) my_sigmaclip_function, "numina.cmethod", my_destructor);
         
      /* Allocating space for the two parameters */
      /* We use Python memory allocator */
      double *funcdata = (double*)PyMem_Malloc(2 * sizeof(double));

      funcdata[0] = low;
      funcdata[1] = high;
      PyCapsule_SetContext(cap, funcdata);
      return cap;
   }

Notice that in this case we construct the :class:`PyCObject` using the same function
than in the previouis case. The aditional data is stored as *Context*.

The deallocator is simply:

.. code-block:: c

   void my_destructor_function(PyObject* cap) {
      void* cdata = PyCapsule_GetContext(cap);
      PyMem_Free(cdata);
   }
   
and the combine function is:

.. code-block:: c

   int my_sigmaclip_function(double *data, double *weights, size_t size, double *out[3], 
            void *func_data) {
       
       double* fdata = (double*) func_data;
       double slow = *fdata;
       double shigh = *(fdata + 1);                 
    
       /* Operations go here */
    
       return 1;    
    }

Once the module is created and loaded, a sample session would be:

    >>> from _mymodule import min_combine
    >>> method = sigmaclip_combine(3.0, 3.0)
    ...
    >>> o = generic_combine(method, arrays)
    
    
    
:mod:`numina.array.cosmetics` --- Array cosmetics
===================================================

.. automodule:: numina.array.cosmetics
   :synopsis: Array cosmetics
   :members:


:mod:`numina.array.fwhm` --- FWHM
===================================================

.. automodule:: numina.array.fwhm
   :members:
    
:mod:`numina.array.imsurfit` --- Image surface fitting
======================================================

.. automodule:: numina.array.imsurfit
   :synopsis: Image surface fitting
   :members:

:mod:`numina.array.interpolation` --- Interpolation
======================================================

.. automodule:: numina.array.interpolation
   :members:

:mod:`numina.array.mode` --- Mode
======================================================

.. automodule:: numina.array.mode
   :members:


:mod:`numina.array.nirproc` --- nIR preprocessing
======================================================

.. automodule:: numina.array.nirproc
   :synopsis: nIR preprocessing
   :members:


:mod:`numina.array.offrot` --- Offset and Rotation
======================================================

.. automodule:: numina.array.offrot
   :members:

:mod:`numina.array.peaks` --- Peak finding
======================================================

.. automodule:: numina.array.peaks.peakdet
   :members:

.. automodule:: numina.array.peaks.detrend
   :members:


:mod:`numina.array.recenter` --- Recenter
======================================================

.. automodule:: numina.array.recenter
   :members:

:mod:`numina.array.robusfit` --- Robust fits
======================================================

.. automodule:: numina.array.robustfit
   :members:


:mod:`numina.array.stats` --- Statistical utilities
======================================================

.. automodule:: numina.array.stats
   :members:

:mod:`numina.array.trace` --- Spectrum tracing
======================================================

.. automodule:: numina.array.trace.traces
   :members:

.. automodule:: numina.array.trace.extract
   :members:


:mod:`numina.array.wavecalib` --- Wavelength calibration
=========================================================

.. automodule:: numina.array.wavecalib.arccalibration
   :members:

.. automodule:: numina.array.wavecalib.peaks_spectrum
   :members:

.. automodule:: numina.array.wavecalib.solutionarc
   :members:

:mod:`numina.array.utils` ---
======================================================

.. automodule:: numina.array.utils
   :members: