
:mod:`numina.array` --- Array manipulation
==========================================

.. automodule:: numina.array
   :synopsis: Array manipulation
   :members:
   
   
:mod:`numina.array.combine` --- Array combination
=================================================

.. automodule:: numina.array.combine
   :synopsis: Array combination
   :members:
   
   
Combination methods in :mod:`numina.array.combine`
==================================================
All these functions return a :class:`PyCObject`, that 
can be passed to :func:`generic_combine`
   
   
.. py:function:: mean_method()

   Mean method

.. py:function:: median_method()

   Median method

.. py:function:: sigmaclip_method([low=0.0[, high=0.0]])

   Sigmaclip method
   
   :param low: Number of sigmas to reject under the mean
   :param high: Number of sigmas to reject over the mean
 
.. py:function:: quantileclip_method([fclip=0.0])

   Quantile clip method
   
   :param fclip: Fraction of points to reject on both ends
   :raises: :class:`ValueError` if fclip is negative or greater than 0.4 Here is a link :func:`time.time`.
 
.. py:function:: minmax_method([nmin=0[, nmax=0]])

   Min-max method

   :param nmin: Number of minimum points to reject
   :param nmax: Number of maximum points to reject
   
   