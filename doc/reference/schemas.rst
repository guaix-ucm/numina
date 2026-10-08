=======================================
:mod:`numina.schemas` --- JSON schemas
=======================================

The schemas of the files read by numina, in JSON Schema (draft 2020-12),
and the function to validate them.

``control-schema.json``
   Control file of :program:`numina run` (format 1).

``drp-schema.json``
   Description of a DRP (``drp.yaml``), validated when the DRP is loaded.

``component-schema.json``
   Elements of the instrument configurations (instruments, components,
   setups and properties), validated when they are loaded.

``oblock-schema.json``
   An observing block, each document of the observation files.

.. automodule:: numina.schemas
   :synopsis: JSON schemas of the files read by numina
   :members:
