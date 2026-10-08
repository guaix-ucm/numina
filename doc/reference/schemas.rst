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

The schemas are published with this documentation, at the URL of their
``$id``, for example
``https://numina.readthedocs.io/en/stable/drp-schema.json``. Editors
that use `yaml-language-server
<https://github.com/redhat-developer/yaml-language-server>`_, such as
Visual Studio Code with the YAML extension, check a file against a
schema while it is edited, and complete its keys, if the file begins
with a comment like::

   # yaml-language-server: $schema=https://numina.readthedocs.io/en/stable/drp-schema.json

The files can also be checked from the command line, for example with
`check-jsonschema <https://check-jsonschema.readthedocs.io>`_::

   check-jsonschema --schemafile https://numina.readthedocs.io/en/stable/drp-schema.json drp.yaml

check-jsonschema does not read YAML files with several documents, such as
most observation files; these are validated by numina when they are
loaded.

.. automodule:: numina.schemas
   :synopsis: JSON schemas of the files read by numina
   :members:
