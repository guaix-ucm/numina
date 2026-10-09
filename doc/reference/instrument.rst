===================================================
:mod:`numina.instrument` --- Instrument description
===================================================

This module contains clases and functions used to create
a model of the instrument during the reduction process.
The model of the instrument is assembled from a set
of JSON configuration files provided by each particular
instrument package, usually in the submodule `packagedrp.instrument.configs`.

The functions in this module build the instrument representation
as an object of a class derived from `numina.instrument.generic.InstrumentGeneric`.





.. automodule:: numina.instrument.assembly
   :members:

.. automodule:: numina.instrument.configorigin
   :members:

.. automodule:: numina.instrument.device
   :members:

.. automodule:: numina.instrument.generic
   :members:

.. automodule:: numina.instrument.hwdevice
   :members:

Components
==========

.. automodule:: numina.instrument.components.wheel
   :members:

.. automodule:: numina.instrument.components.detector
   :members:

Simulation
==========

.. automodule:: numina.instrument.simulation.actions
   :members:

.. automodule:: numina.instrument.simulation.atmosphere
   :members:

.. automodule:: numina.instrument.simulation.efficiency
   :members:

.. automodule:: numina.instrument.simulation.factory
   :members:

.. automodule:: numina.instrument.simulation.optics
   :members:
