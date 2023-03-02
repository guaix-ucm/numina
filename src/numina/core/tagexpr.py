#
# Copyright 2018-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Classes to express tags constraints in recipe requirements

Each type has a series of tag attributes inside the .tags member. Operating
with tags creates tag expressions.

>>> tags1 = Tagged(['vph', 'insmode'])
>>> tags1.tags.insmode == "LCB"

When processed, this represents that the `insmode` key of a particular calibration must be equal to "LCB".

Most of the time, you don't need a fixed value, but a value that comes from the observed data

>>> tags1.tags.insmode == tags1.tags.p_("insmode")

In this case, the `insmode` tag of the calibration must be equal to the value of insmode extracted from
the observed data.

Expresions can be nested and make use of the typical logical conectors (AND, OR, NOT), like:

>>> (tags1.tags.insmode == tags1.tags.p_("insmode")) & (tags1.tags.vph == tags1.tags.p_("vph"))

Where & represents AND, | represents OR and ~ represents NOT

"""

import operator

try:
    from functools import reduce
except ImportError:
    pass


def map_tree(visitor, tree):
    """Apply function to nodes"""
    newn = [map_tree(visitor, node) for node in tree.nodes]
    return visitor(tree, newn)


def filter_tree(condition, tree):
    """Return parts of the tree that fulfill condition"""
    if condition(tree):
        for node in tree.nodes:
            # this works in python > 3.3
            # yield from filter_tree(condition, node)
            for n in filter_tree(condition, node):
                yield n
    yield tree


class Expression(object):
    """Base class for expressions"""
    def __init__(self, *args):

        self.nodes = []
        self.metadata = {}
        for arg in args:
            if isinstance(arg, Expression):
                self.nodes.append(arg)
            else:
                self.nodes.append(ConstExpr(arg))

        self._fields = set()
        self._places = set()

        for node in self.nodes:
            self._fields.update(node.fields())
            self._places.update(node.places())

    def fields(self):
        return self._fields

    def tags(self):
        return self._places

    def places(self):
        return self._places

    def is_terminal(self):
        "True for leaf nodes"
        return len(self.nodes) == 0

    def copy(self):

        def copy_node(node, children):
            return node.clone(children)

        return map_tree(copy_node, self)

    def __eq__(self, other):
        return PredEq(self, other)

    def __gt__(self, other):
        return PredGt(self, other)

    def __ge__(self, other):
        return PredGe(self, other)

    def __le__(self, other):
        return PredLe(self, other)

    def __lt__(self, other):
        return PredLt(self, other)

    def __ne__(self, other):
        return PredNe(self, other)

    def __or__(self, other):
        if isinstance(other, CompoundExpr):
            return PredOr(self, other)

        if isinstance(other, bool):
            if other:
                return True
            else:
                return other

        return NotImplemented

    def __ror__(self, other):
        return self.__or__(other)

    def __and__(self, other):
        if isinstance(other, CompoundExpr):
            return PredAnd(self, other)

        if isinstance(other, bool):
            if other:
                return self
            else:
                return False

        return NotImplemented

    def __rand__(self, other):
        return self.__and__(other)

    def __invert__(self):
        return PredNot(self)

    @staticmethod
    def map_tree(visitor, tree):
        return map_tree(visitor, tree)

    def fill_placeholders(self, **kwargs):
        """Substitute Placeholder nodes by its value in tags"""
        def change_p_node_tags(node, children):
            if isinstance(node, Placeholder) and (node.name in kwargs):
                return ConstExpr(kwargs[node.name])
            else:
                return node.clone(children)

        return map_tree(change_p_node_tags, self)

    fill_tags = fill_placeholders

    def eval(self, **kwargs):

        def eval_nodes(node, children):
            if isinstance(node, BinaryExpr):
                return node.operator(children[0], children[1])
            if isinstance(node, UnaryExpr):
                return node.operator(children[0])
            else:
                return node.eval(**kwargs)

        return map_tree(eval_nodes, self)

    def clone(self, children):
        raise NotImplementedError


class AtomicExpr(Expression):
    """"Atomic expression"""
    def __init__(self, name, value):
        super(AtomicExpr, self).__init__()
        self.name = name
        self.value = value


class TagRepr(AtomicExpr):
    "A representation of a Tag"
    def __init__(self, name, metadata=None):
        super(TagRepr, self).__init__(name, name)
        self.metadata = metadata or {}
        self._fields.add(name)

    def clone(self, nodes):
        return self

    def eval(self, **kwargs):
        if self.value in kwargs:
            return kwargs[self.value]
        else:
            return self

    def __repr__(self):
        return f"TagRepr({self.name})"


class Placeholder(AtomicExpr):
    """A representation of a value expected to be substituted"""
    def __init__(self, name):
        super(Placeholder, self).__init__(name, name)
        self._places.add(name)

    def __repr__(self):
        return f"Placeholder({self.name})"

    def clone(self, nodes):
        return self


class ConstExpr(AtomicExpr):
    """A representation of a constant value"""
    def __init__(self, value):
        super(ConstExpr, self).__init__("ConstExpr", value)

    def clone(self, nodes):
        return self

    def __repr__(self):
        return f"ConstExpr({self.value})"

    def eval(self, **kwargs):
        return self.value


ConstExprTrue = ConstExpr(True)
ConstExprFalse = ConstExpr(False)


class CompoundExpr(Expression):
    """Compound expression"""
    pass


class UnaryExpr(CompoundExpr):
    def __init__(self, pred, oper):
        super(UnaryExpr, self).__init__(pred)
        self.pred = self.nodes[0]
        self.operator = oper

    def clone(self, nodes):
        new = self.__class__(nodes[0], self.operator)
        return new


class BinaryExpr(CompoundExpr):
    def __init__(self, lhs, rhs, oper, op_rep):
        super(BinaryExpr, self).__init__(lhs, rhs)
        self.lhs = self.nodes[0]
        self.rhs = self.nodes[1]
        self.operator = oper
        self.op_rep = op_rep

    def clone(self, nodes):
        new = self.__class__(nodes[0], nodes[1])
        return new

    def __str__(self):
        return f"({str(self.lhs)} {self.op_rep} {str(self.rhs)})"


class PredAnd(BinaryExpr):
    def __init__(self, lhs, rhs):
        super(PredAnd, self).__init__(lhs, rhs, operator.and_, 'AND')


class PredOr(BinaryExpr):
    def __init__(self, lhs, rhs):
        super(PredOr, self).__init__(lhs, rhs, operator.or_, 'OR')


class PredNot(UnaryExpr):
    def __init__(self, pred):
        super(PredNot, self).__init__(pred, operator.not_)


class PredEq(BinaryExpr):
    def __init__(self, lhs, rhs):
        super(PredEq, self).__init__(lhs, rhs, operator.eq, '==')


class PredGt(BinaryExpr):
    def __init__(self, key, value):
        super(PredGt, self).__init__(key, value, operator.gt, '>')


class PredGe(BinaryExpr):
    def __init__(self, key, value):
        super(PredGe, self).__init__(key, value, operator.ge, '>=')


class PredLe(BinaryExpr):
    def __init__(self, key, value):
        super(PredLe, self).__init__(key, value, operator.le, '<=')


class PredLt(BinaryExpr):
    def __init__(self, key, value):
        super(PredLt, self).__init__(key, value, operator.lt, '<')


class PredNe(BinaryExpr):
    def __init__(self, key, value):
        super(PredNe, self).__init__(key, value, operator.ne, '!=')


class ConstraintAdapter(object):
    """For GTC"""
    def __init__(self, key, value, oper):
        self.key = key
        self.value = value
        self.type = type(value)
        self.oper = oper

    def __repr__(self):
        return f"ConstraintAdapter(key={self.key}, value={self.value}, {self.oper}, type={self.type})"


def condition_terminal(tree):
    if tree.nodes:
        term = all(node.is_terminal() for node in tree.nodes)
        return not term
    else:
        return True


def adapter(tree):
    if tree.nodes and all(node.is_terminal() for node in tree.nodes):
        return ConstraintAdapter(tree.lhs.name, tree.rhs.value, tree.operator)


def query_expr_from_attr(attrs):
    # Create a query descriptor from a sequence for fields
    if len(attrs) == 0:
        return ConstExprTrue
    exprs = []
    #for name, dtype in descs:
    for attr in attrs:
        metadata = {'type': attr.type, 'description': attr.description}
        lhs = TagRepr(attr.name, metadata=metadata)
        rhs = Placeholder(attr.name)
        expr = (lhs == rhs)
        exprs.append(expr)
    return reduce(operator.and_, exprs)
