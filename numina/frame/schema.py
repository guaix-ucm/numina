#
# Copyright 2014-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""FITS header schema and validation.

This module is a simplification of the FITS Schema defined
by Erik Bray here:
http://embray.github.io/PyFITS/schema/users_guide/users_schema.html

If this schema implementation reaches pyfits/astropy stable,
we will use it instead of ours, with schema definitions
being the same.

"""


class SchemaValidationError(Exception):
    """Exception raised when a Schema does not validate a FITS header."""
    pass


class SchemaDefinitionError(Exception):
    """Exception raised when a FITS Schema definition is not valid."""
    pass


def _from_ipt(value):
    if isinstance(value, bool):
        return (value, bool)
    if isinstance(value, str):
        return (value, str)
    elif isinstance(value, int):
        return (value, int)
    elif isinstance(value, float):
        return (value, float)
    elif isinstance(value, complex):
        return (value, complex)
    elif value in [str, int, float, complex, bool]:
        return (None, value)
    elif isinstance(value, list):
        if value:
            _, type_ = _from_ipt(value[0])
            return value, type_
        else:
            raise SchemaDefinitionError(value)
    else:
        raise SchemaDefinitionError(value)


class SchemaKeyword(object):
    """A keyword in the schema"""
    def __init__(self, name, mandatory=False, valid=True,
                 value=None):
        self.name = name
        self.mandatory = mandatory
        self.valid = valid
        if self.mandatory and not self.valid:
            raise SchemaDefinitionError(
                "keyword 'cannot be 'mandatory' and "
                "'not valid'"
                )
        self.choose = False
        self.valcheck = False
        self.value = None
        self.type_ = None
        if value is not None:
            self.value, self.type_ = _from_ipt(value)
            if self.value is not None:
                self.valcheck = True
                if isinstance(self.value, list):
                    self.choose = True

    def validate(self, header):
        sname = 'schema'
        # check the keyword is defined
        val = header.get(self.name)

        if val is None:
            if self.mandatory:
                raise SchemaValidationError(
                    sname, f'mandatory keyword {self.name!r} missing from header')

            # In the rest of cases
            return True
        else:
            if not self.valid:
                raise SchemaValidationError(
                    sname, f'invalid keyword {self.name!r} present in header')

        # Cases here
        # val is not None and key id mandatory or valid

        if not self.type_:
            # We dont have type information
            # Nothing more to do
            return True
        else:
            if not isinstance(val, self.type_):
                raise SchemaValidationError(
                    sname, 'keyword %r is required to have a value of type %r'
                    '; got a value of type %r instead' %
                    (self.name, self.type_.__name__, type(val).__name__))
            # Check value
            if self.choose:
                if val not in self.value:
                    raise SchemaValidationError(
                        sname,
                        'keyword %r is required to have one of the values %r; '
                        'got %r instead' %
                        (self.name, self.value, val))
                else:
                    return True
            elif self.valcheck:
                if val != self.value:
                    raise SchemaValidationError(
                        sname,
                        'keyword %r is required to have the value %r; got '
                        '%r instead' % (self.name, self.value, val))
            else:
                pass

        return True


class Schema(object):
    """A FITS schema"""
    def __init__(self, sc):
        self.kwl = []
        self.extend(sc)

    def validate(self, header):
        for ll in self.kwl:
            ll.validate(header)

    def extend(self, sc):
        kw = sc.get('keywords', {})
        for k, v in kw.items():
            mandatory = v.get('mandatory', False)
            valid = v.get('valid', True)
            value = v.get('value', None)
            sk = SchemaKeyword(
                k, mandatory=mandatory,
                valid=valid, value=value
                )
            self.kwl.append(sk)


types_table = {
    'string': str,
    'number': float,
    'integer': int,
    'bool': bool
}

class SchemaNode(object):
    def __init__(self, *args, **kwargs):
        self.title = kwargs.get('title', 'undefined')
        self.required = kwargs.get('required', True)
        #
        self.group = kwargs.get('group', None)
        self.matched = kwargs.get('matched', None)
        #
        self.children_obj = {}

    def validate_req(self, value, state):
        if self.group is not None:
            print('this node has group', self.group)
        # value must be hdulist
        pass

    def validate(self, value):
        pass


class SchemaExtension(SchemaNode):
    def __init__(self, *args, **kwargs):
        super(SchemaExtension, self).__init__(*args, **kwargs)

    def validate(self, value):
        # value must be hdulist
        pass

    def validate_req(self, value, state):
        # super(SchemaExtension, self).validate_req(value, state)

        if self.group is not None:
            print('this node has group', self.group)
            print('matched', self.matched)
            if self.group in state['groups']:
                # both must match
                if state['groups'][self.group] != self.matched:
                    msg = 'matched group {} differs from state group {}'
                    raise ValueError(msg.format(self.matched, state['groups'][self.group]))

        for key, val in self.children_obj.items():
            val.validate_req(value, state)
            print(state)


class SchemaKeywordII(SchemaNode):
    def __init__(self, *args, **kwargs):
        super(SchemaKeywordII, self).__init__(*args, **kwargs)

        self.v_type = kwargs.get('type', None)
        self.enum_t = kwargs.get('enum', None)
        self.value_t = kwargs.get('value', None)

        # if enum_t is a list, and group is set
        # matched must be a list of the same length


    def validate_req(self, value, state):
        print('validate keyword', self.title)

        hvalue = value.get(self.title)
        matched_val = None
        print('hvalue', hvalue)
        sname = 'test'
        key = 0
        if self.required:
            print('keyword required')
            # if not defined, fail
            if hvalue is None:
                raise SchemaValidationError(
                    sname, f'required keyword {key!r} missing from header')
        else:
            print('keyword not required')
        # check type
        if self.v_type:
            ptype = types_table[self.v_type]
            if not isinstance(hvalue, ptype):
                raise SchemaValidationError(
                    sname, 'keyword %r is required to have a value of type %r'
                           '; got a value of type %r instead' %
                           (key, ptype.__name__, type(hvalue).__name__))

        if self.enum_t:
            try:
                match = self.enum_t.index(hvalue)
                matched_val = self.matched[match]


            except ValueError:
                raise SchemaValidationError(
                    sname,
                    'keyword %r is required to have one of the values %r; '
                    'got %r instead' %
                    (self.title, self.enum_t, hvalue))


        if self.value_t:
            matched_val = self.matched
            if self.value_t != hvalue:
                raise SchemaValidationError(
                    sname,
                    'keyword %r is required to have value %r; '
                    'got %r instead' %
                    (key, self.value_t, hvalue))


        if self.group is not None:
            print('this node has group', self.group)
            print('matched', self.matched)
            if self.group in state['groups']:
                # both must match
                if state['groups'][self.group] != matched_val:
                    msg = 'matched group {} differs from state group {}'
                    raise ValueError(msg.format(matched_val, state['groups'][self.group]))
                else:
                    msg = 'matched group {} = value from state group {}'
                    print(msg.format(matched_val, state['groups'][self.group]))
            else:
                state['groups'][self.group] = matched_val


def create_keyword_node(node, key=None):
    if 'title' not in node:
        node['title'] = key
    return SchemaKeywordII(**node)


def create_extension_node(node):
    ext = SchemaExtension(**node)
    keywords = node.get('keywords', {})
    for k, knode in keywords.items():
        ext.children_obj[k] = create_keyword_node(knode, key=k)
    return ext


class SchemeKeyword(object):
    def __init__(self, *args, **kwargs):
        types_table = {
            'string': str,
            'number': float,
            'integer': int,
            'bool': bool
        }

        self.key = "key"
        self.required = kwargs.get('required', True)
        self.enum = kwargs.get('enum', None)
        self.value = kwargs.get('value', None)
        self.type = kwargs.get('type', None)
        self.type_p = types_table.get(self.type, None)

    def validate(self, value):
        sname = 'titlw'
        key = "ky"
        if self.required:
            if value is None:
                msg = f'required keyword {key!r} missing from header'
                raise SchemaValidationError(sname, msg)
        if self.type:
            if not isinstance(value, self.type_p):
                raise SchemaValidationError(
                    sname, 'keyword %r is required to have a value of type %r'
                           '; got a value of type %r instead' %
                           (key, self.type_p.__name__, type(value).__name__))


class SchemeKeywordString(SchemeKeyword):
    def __init__(self, *args, **kwargs):
        super(SchemeKeywordString, self).__init__(*args, **kwargs)





def validate(header, schema):
    sname = schema.get('title', 'schema')
    schema_keys = schema.get('keywords', {})
    for key, desc in schema_keys.items():
        required = desc.get('required', True)
        vtype = desc.get('type', None)
        enumt = desc.get('enum', None)
        valuet = desc.get('value', None)
        hvalue = header.get(key, None)
        if required:
            # if not defined, fail
            if hvalue is None:
                raise SchemaValidationError(
                    sname, f'required keyword {key!r} missing from header')
        # check type
        if vtype:
            ptype = types_table[vtype]
            if not isinstance(hvalue, ptype):
                raise SchemaValidationError(
                    sname, 'keyword %r is required to have a value of type %r'
                           '; got a value of type %r instead' %
                           (key, ptype.__name__, type(hvalue).__name__))

        if enumt:
            if hvalue not in enumt:
                raise SchemaValidationError(
                    sname,
                    'keyword %r is required to have one of the values %r; '
                    'got %r instead' %
                    (hvalue, enumt, hvalue))

        if valuet:
            if valuet != hvalue:
                raise SchemaValidationError(
                    sname,
                    'keyword %r is required to have value %r; '
                    'got %r instead' %
                    (key, valuet, hvalue))


def validate_image(image, schema):
    print('validate image')
    state = {}
    state['groups'] = {}

    for extname, extnode in schema['extensions'].items():

        node = create_extension_node(extnode)
        if extname in image:
            node.validate_req(image[extname].header, state)
        else:
            if node.required:
                print('required ext', extname, 'not present')
            else:
                print('not required ext', extname, 'not present')

