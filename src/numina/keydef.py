#
# Copyright 2008-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io.fits import HDUList, Header


class QueryAttribute:
    def __init__(self, name, tipo, description=""):
        self.name = name
        self.type = tipo
        self.description = description


class KeyDefinition:
    def __init__(
        self, key: str, ext: int | str | None = None, default=None, convert=None
    ):
        self.key = key
        self.ext = 0 if ext is None else ext
        self.default = default
        # convert must be callable or None
        self.convert = convert
        # if selector is None:
        #    def selector(x): return x  # noqa: E731

    def _get_header(self, head: HDUList | Header) -> Header:
        match head:
            case HDUList():
                return head[self.ext].header
            case Header():
                return head
            case _:
                raise ValueError("head is not HDUList nor Header")

    def get_value(self, head: HDUList | Header):
        hdr = self._get_header(head)
        value = hdr.get(self.key, self.default)
        if self.convert:
            return self.convert(value)
        return value

    def set_value(self, head: HDUList | Header, value=None):
        hdr = self._get_header(head)

        if value is None:
            true_value = self.default
        else:
            if self.convert:
                true_value = self.convert(value)
            else:
                true_value = value

        hdr[self.key] = true_value

    def __call__(self, head: HDUList | Header):
        return self.get_value(head)


class FITSKeyExtractor:
    """Extract values from FITS images"""

    def __init__(self, values):
        self.map = {}
        for key, entry in values.items():
            if isinstance(entry, KeyDefinition):
                newval = entry
            elif isinstance(entry, tuple):
                if len(entry) == 3:
                    keyname = entry[0]
                    hduname = entry[1]
                    convert = entry[2]
                    default = None
                elif len(entry) == 2:
                    keyname = entry[0]
                    default = entry[1]
                    hduname = 0
                    convert = None
                else:
                    raise ValueError(
                        f"a tuple in FITSKeyExtractor must have 2-3 fields, has {len(entry)} instead"
                    )

                newval = KeyDefinition(
                    keyname, ext=hduname, convert=convert, default=default
                )
            elif isinstance(entry, str):
                newval = KeyDefinition(entry)
            else:
                newval = entry

            self.map[key] = newval

    def extract(self, value, hdulist):
        extractor = self.map[value]
        return extractor(hdulist)


def extract(header, conf, path, key_def):
    """Extracts values from a config dictionary and stores them in a header."""
    m = conf
    try:
        for part in path:
            m = m[part]
        key_def.set_value(header, m)
    except KeyError:
        # Keyword missing
        key_def.set_value(header)
    return header
