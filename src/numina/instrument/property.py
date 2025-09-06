#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


class PropertyBase:
    def __init__(self, is_readonly=True, mapping=None, depends=None):
        self.is_readonly = is_readonly
        self.mapping = mapping
        self.name = "undefined"
        #
        if depends is None:
            self.depends = []
        else:
            self.depends = depends

    def __set_name__(self, owner, name):
        self.name = name


class PropertyEntry(PropertyBase):
    """Class representing a property entry"""

    def __init__(self, values, depends):
        super().__init__(depends=depends)
        self.values = values

    def get(self, **kwds):
        result = self.values
        for dep in self.depends:
            key = kwds[dep]
            result = result[key]
        return result

    def __get__(self, instance, owner=None):
        state = {dep: None for dep in self.depends}
        obj = instance
        while obj is not None:
            early_exit = True
            for dep in self.depends:
                if state[dep] is None:
                    if hasattr(obj, dep):
                        state[dep] = getattr(obj, dep)
                    else:
                        early_exit = False
            if early_exit:
                # we have filled all Nones
                break
            # Find top level
            obj = obj.parent

        result = self.values
        for dep in self.depends:
            key = state[dep]
            result = result[key]
        return result

    def __set__(self, obj, value):
        pass

    def defaults(self):
        val = {}
        result = self.values
        for dep in self.depends:
            # First key, sorted needed in Python < 3.6
            key = sorted(result.keys())[0]
            val[dep] = key
            result = result[key]
        return val


class PropertyProxy(PropertyBase):
    """Class representing a property proxy"""

    def __init__(self, ref_name, is_readonly=True):
        # FIXME: depends should be the depends of the ref_name
        super().__init__(is_readonly=is_readonly, mapping=None, depends=None)
        self.ref_name = ref_name

    def defaults(self):
        # FIXME: defaults should be the defaults of the ref_name
        val = {}
        return val

    def __get__(self, instance, owner=None):
        res = instance.get_property(self.ref_name)
        return res

    def __set__(self, obj, value):
        dev_path, prop_name = obj.split_path(self.ref_name)
        device = obj.get_device_seq(dev_path)
        setattr(device, prop_name, value)


class _PropertyModBase(PropertyBase):
    """Class representing a property"""

    def __init__(self, default=0, mapping=None):
        super().__init__(is_readonly=False, mapping=mapping, depends=None)
        self.default = default

    def defaults(self):
        # FIXME
        val = {}
        return val

    def value_check(self, value):  # pragma: no cover
        return True

    def __get__(self, instance, owner=None):
        intl = instance._internal_state  # noqa
        if self.name not in intl:
            intl[self.name] = self.default
        return intl[self.name]

    def __set__(self, obj, value):
        if self.value_check(value):
            obj._internal_state[self.name] = value  # noqa
        else:
            raise ValueError(f"value {value} is not allowed")


class PropertyModOneOf(_PropertyModBase):
    """Class representing a property"""

    def __init__(self, one_of, default=0, mapping=None):
        self.one_of = one_of
        super().__init__(default, mapping)

    def value_check(self, value):
        if value in self.one_of:
            return True
        else:
            return False


class PropertyModLimits(_PropertyModBase):
    """Class representing a property in a range"""

    def __init__(self, limits, default=0, mapping=None):
        match limits:
            case []:
                self.limit_1 = None
                self.limit_2 = None
            case [a, b]:
                self.limit_1 = a
                self.limit_2 = b
            case _:
                raise ValueError(f"invalid range {limits}")

        super().__init__(default, mapping)

    def value_check(self, value):
        if self.limit_1 is not None and value < self.limit_1:
            return False
        if self.limit_2 is not None and value > self.limit_2:
            return False

        return True
