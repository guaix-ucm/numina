==================================================
:mod:`numina.types` --- Data types
==================================================

.. automodule:: numina.types
   :synopsis: Types of the requirements and the results of the recipes
   :members:

Types and values
----------------

Each requirement and each result of a recipe has a type, given when it is
declared::

    master_bias = Requirement(MasterBias, "Master bias image")
    reduced_image = Result(ProcessedImage)

The type is an object of a subclass of :class:`~numina.types.datatype.DataType`,
created by the requirement if a class is given. It is not the value that the
recipe receives or returns, but its description:

- :meth:`~numina.types.datatype.DataType.convert` and
  :meth:`~numina.types.datatype.DataType.validate` convert and check the values;
- ``_datatype_load`` and ``_datatype_dump`` load a value from its serialized
  form (usually a file) and save it, through :func:`numina.store.load` and
  :func:`numina.store.dump`;
- the tags of the type (``__tags__``) select the products that a requirement
  can use, for example a master bias with the same read mode;
- :meth:`~numina.types.datatype.DataType.isproduct` tells products (searched in
  the registry or the calibrations) from parameters (searched in the control
  file).

The types and their values are:

.. list-table::
   :header-rows: 1

   * - Type
     - Value
   * - :class:`~numina.types.frame.DataFrameType`
     - :class:`~numina.types.dataframe.DataFrame`, an image in disk or in memory
   * - :class:`~numina.types.array.ArrayType`
     - :class:`numpy.ndarray`
   * - :class:`~numina.types.datatype.PlainPythonType`
     - a Python value (``int``, ``float``, ``str``...), used by
       :class:`~numina.core.dataholders.Parameter`
   * - :class:`~numina.types.datatype.ListOfType`
     - a list of the values of another type
   * - :class:`~numina.types.multitype.MultiType`
     - a value of one of several types
   * - :class:`~numina.types.obsresult.ObservationResultType`
     - :class:`~numina.core.oresult.ObservationResult`

The structured calibrations, subclasses of
:class:`~numina.types.structured.BaseStructuredCalibration`, are the exception:
the same class is the type and the value, as a trace map is described by the
class ``TraceMap`` of megaradrp and stored as an object of that class.

A DRP defines its own types as subclasses of these, usually to add tags, a
data model or a validation of its products.


.. automodule:: numina.types.base
   :synopsis: TBD
   :members:


.. automodule:: numina.types.array
   :synopsis: TBD
   :members:

.. automodule:: numina.types.dataframe
   :synopsis: TBD
   :members:

.. automodule:: numina.types.datatype
   :synopsis: TBD
   :members:

.. automodule:: numina.types.frame
   :synopsis: TBD
   :members:

.. automodule:: numina.types.linescatalog
   :synopsis: TBD
   :members:


.. automodule:: numina.types.multitype
   :synopsis: TBD
   :members:

.. automodule:: numina.types.product
   :synopsis: Data products
   :members:

.. automodule:: numina.types.qc
   :synopsis: Quality Control for Numina
   :members:

QA Levels
---------

The numeric values of the QC levels are given in this table.

+--------------+---------------+
| Level        | Numeric value |
+==============+===============+
| ``GOOD``     | 100           |
+--------------+---------------+
| ``FAIR``     | 90            |
+--------------+---------------+
| ``BAD``      | 70            |
+--------------+---------------+


.. automodule:: numina.types.structured
   :synopsis: TBD
   :members:

.. automodule:: numina.types.typedialect
   :synopsis: Description of the data types for other systems
   :members:
