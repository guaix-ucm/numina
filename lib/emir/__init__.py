# $Id$

EMIR_VERSION_STRING = '0.1.0'

import logging

try:
    from logging import NullHandler
except ImportError:
    from logger import NullHandler

# Top level NullHandler
logging.getLogger("emir").addHandler(NullHandler())