**************
DRP Data Types
**************

Custom data types can be used as Requirements and Products by Recipes.
New data types can be derived as follows.

Create a new DataType
#####################

New Data Types must derive from `numina.core.DataType` or
one of its subclasses. In the constructor, we must declare the base type
of the objects ot this Data Product.

For example, a `MasterBias` Data Product is an image, so its base type
is a `DataFrame`. A table of 2D coordinates will have a `numpy.ndarray`
base type.

In general, we are interested in defining new DataTypes for objects that
will contain information that will be used as inputs in different recipes.
In this case, we must derive from `numina.core.DataProductType`.

As an example, we create a DataType that will store information about the
trace of a spectrum. The information will be stored in Python `dict`.

.. code-block:: python

    class TraceMap(DataProductType):
        def __init__(self, default=None):
            super(TraceMap, self).__init__(dict, default)


Construction of objects
#######################

The input of a recipe is created by inspecting the Recipe Requirements.
The Recipe Loader is in charge of finding an appropriated value for each
requirement. The value is passed to `Requirement.convert`, that in turn
calls `DataType.convert`. The default implementation just returns in
input object unchanged.


Loading and Storage with the command line Recipe Loader
#######################################################
Each Recipe Loader can implement its own mechanism to store and load Data
Products. The Command Line Recipe Loader uses text files in YAML format.

To define how a particular DataProduct is stored under the default Recipe Loader,
two functions must be defined, a store function and a load function. Then
thse two functions must be registered with the global
functions `numina.store.dump` and `numina.store.load`.


.. code-block:: python

    from numina.store import dump, load

    from .products import TraceMap

    @dump.register(TraceMap)
    def dump_tracemap(tag, obj, where):

        filename = where.destination + '.yaml'

        with open(filename, 'w') as fd:
            yaml.dump(obj, fd)

        return filename

    @load.register(TraceMap)
    def load_tracemap(tag, obj):

        with open(obj, 'r') as fd:
            traces = yaml.load(fd)

        return traces


In this example, `tag` is an argument of type `TraceMap` and `obj` is of
type `dict`.


Formats of the files
####################
A type can declare the formats of its files that numina reads in the class
attribute ``__numina_formats__``: the number of each format and the name of
the method that reads a file in that format. Files without envelope are
format 1 (see :mod:`numina.store.envelope`).

.. code-block:: python

    from numina.types.product import DataProductType

    class LinesCatalog(DataProductType):
        __numina_formats__ = {1: "_datatype_load", 2: "_load_ecsv_table"}

When the type is loaded from a file, `numina.store.load` reads the envelope
of the file, checks that its type is the name of the type and that its
format is declared, and reads the file with the method of the format. A
file in a format that is not declared is not read, the error names the
file, the formats read and the package to update.
