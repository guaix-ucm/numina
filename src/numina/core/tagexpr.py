#
# Copyright 2018-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
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


class Expression:
    """Base class for expressions

    Parameters
    ----------
    *args
        The child nodes. Arguments that are not an ``Expression`` are
        wrapped in a ``ConstExpr``.
    """

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
        """Return the names of the tags used in the expression."""
        return self._fields

    def tags(self):
        """Return the names of the placeholders of the expression, as ``places``."""
        return self._places

    def places(self):
        """Return the names of the placeholders of the expression."""
        return self._places

    def is_terminal(self):
        """True for leaf nodes"""
        return len(self.nodes) == 0

    def copy(self):
        """Return a copy of the expression tree, the leaf nodes are shared."""

        def copy_node(node, children):
            return node.clone(children)

        return map_tree(copy_node, self)

    def __eq__(self, other):
        """Return the expression ``self == other``."""
        return PredEq(self, other)

    def __gt__(self, other):
        """Return the expression ``self > other``."""
        return PredGt(self, other)

    def __ge__(self, other):
        """Return the expression ``self >= other``."""
        return PredGe(self, other)

    def __le__(self, other):
        """Return the expression ``self <= other``."""
        return PredLe(self, other)

    def __lt__(self, other):
        """Return the expression ``self < other``."""
        return PredLt(self, other)

    def __ne__(self, other):
        """Return the expression ``self != other``."""
        return PredNe(self, other)

    def __or__(self, other):
        """Return the expression ``self OR other``.

        `other` can be a compound expression or a bool, ``True`` gives ``True``
        and ``False`` gives ``self``.
        """
        if isinstance(other, CompoundExpr):
            return PredOr(self, other)

        if isinstance(other, bool):
            if other:
                return True
            else:
                return self

        return NotImplemented

    def __ror__(self, other):
        """Return the expression ``other OR self``, see ``__or__``."""
        return self.__or__(other)

    def __and__(self, other):
        """Return the expression ``self AND other``.

        `other` can be a compound expression or a bool, ``True`` gives ``self``
        and ``False`` gives ``False``.
        """
        if isinstance(other, CompoundExpr):
            return PredAnd(self, other)

        if isinstance(other, bool):
            if other:
                return self
            else:
                return False

        return NotImplemented

    def __rand__(self, other):
        """Return the expression ``other AND self``, see ``__and__``."""
        return self.__and__(other)

    def __invert__(self):
        """Return the expression ``NOT self``."""
        return PredNot(self)

    @staticmethod
    def map_tree(visitor, tree):
        """Apply `visitor` to the nodes of `tree`, from the leaves to the root."""
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
        """Evaluate the expression with the values of the tags in `kwargs`.

        The tags without value and the placeholders remain unevaluated, and
        then the result can be an expression.
        """

        def eval_nodes(node, children):
            if isinstance(node, BinaryExpr):
                return node.operator(children[0], children[1])
            if isinstance(node, UnaryExpr):
                return node.operator(children[0])
            else:
                return node.eval(**kwargs)

        return map_tree(eval_nodes, self)

    def clone(self, children):
        """Return a node like this one, with the child nodes `children`."""
        raise NotImplementedError


class AtomicExpr(Expression):
    """Atomic expression

    Parameters
    ----------
    name : str
        Name of the node.
    value
        Value of the node.
    """

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value


class TagRepr(AtomicExpr):
    """A representation of a Tag

    Parameters
    ----------
    name : str
        Name of the tag.
    metadata : dict, optional
        Description of the tag, such as its ``type`` and ``description``.
    """

    def __init__(self, name, metadata=None):
        super().__init__(name, name)
        self.metadata = metadata or {}
        self._fields.add(name)

    def clone(self, nodes):
        """Return the node itself."""
        return self

    def eval(self, **kwargs):
        """Return the value of the tag in `kwargs`, or the node itself if it is not there."""
        if self.value in kwargs:
            return kwargs[self.value]
        else:
            return self

    def __repr__(self):
        return f"TagRepr({self.name})"


class Placeholder(AtomicExpr):
    """A representation of a value expected to be substituted

    Parameters
    ----------
    name : str
        Name of the placeholder.
    """

    def __init__(self, name):
        super().__init__(name, name)
        self._places.add(name)

    def __repr__(self):
        return f"Placeholder({self.name})"

    def clone(self, nodes):
        """Return the node itself."""
        return self

    def eval(self, **kwargs):
        """Return the node itself, the placeholder was not substituted."""
        # Not substituted, the expression is incomplete
        return self


class ConstExpr(AtomicExpr):
    """A representation of a constant value

    Parameters
    ----------
    value
        The constant value.
    """

    def __init__(self, value):
        super().__init__("ConstExpr", value)

    def clone(self, nodes):
        """Return the node itself."""
        return self

    def __repr__(self):
        return f"ConstExpr({self.value})"

    def eval(self, **kwargs):
        """Return the constant value."""
        return self.value


ConstExprTrue = ConstExpr(True)
ConstExprFalse = ConstExpr(False)


class CompoundExpr(Expression):
    """Compound expression"""

    pass


class UnaryExpr(CompoundExpr):
    """Expression with an operator and one operand.

    Parameters
    ----------
    pred : Expression or value
        The operand, a value that is not an expression is a ``ConstExpr``.
    oper : callable
        The operator, applied to the value of the operand by ``eval``.
    """

    def __init__(self, pred, oper):
        super().__init__(pred)
        self.pred = self.nodes[0]
        self.operator = oper

    def clone(self, nodes):
        """Return a new expression of the same class, with the operand ``nodes[0]``."""
        # The subclasses take only the operand, as in BinaryExpr
        new = self.__class__(nodes[0])
        return new


class BinaryExpr(CompoundExpr):
    """Expression with an operator and two operands.

    Parameters
    ----------
    lhs, rhs : Expression or value
        The operands, a value that is not an expression is a ``ConstExpr``.
    oper : callable
        The operator, applied to the values of the operands by ``eval``.
    op_rep : str
        Representation of the operator, used by ``str``.
    """

    def __init__(self, lhs, rhs, oper, op_rep):
        super().__init__(lhs, rhs)
        self.lhs = self.nodes[0]
        self.rhs = self.nodes[1]
        self.operator = oper
        self.op_rep = op_rep

    def clone(self, nodes):
        """Return a new expression of the same class, with the operands ``nodes[0]`` and ``nodes[1]``."""
        new = self.__class__(nodes[0], nodes[1])
        return new

    def __str__(self):
        return f"({str(self.lhs)} {self.op_rep} {str(self.rhs)})"


class PredAnd(BinaryExpr):
    """Expression ``lhs AND rhs``."""

    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs, operator.and_, "AND")


class PredOr(BinaryExpr):
    """Expression ``lhs OR rhs``."""

    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs, operator.or_, "OR")


class PredNot(UnaryExpr):
    """Expression ``NOT pred``."""

    def __init__(self, pred):
        super().__init__(pred, operator.not_)


class PredEq(BinaryExpr):
    """Expression ``lhs == rhs``."""

    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs, operator.eq, "==")


class PredGt(BinaryExpr):
    """Expression ``key > value``."""

    def __init__(self, key, value):
        super().__init__(key, value, operator.gt, ">")


class PredGe(BinaryExpr):
    """Expression ``key >= value``."""

    def __init__(self, key, value):
        super().__init__(key, value, operator.ge, ">=")


class PredLe(BinaryExpr):
    """Expression ``key <= value``."""

    def __init__(self, key, value):
        super().__init__(key, value, operator.le, "<=")


class PredLt(BinaryExpr):
    """Expression ``key < value``."""

    def __init__(self, key, value):
        super().__init__(key, value, operator.lt, "<")


class PredNe(BinaryExpr):
    """Expression ``key != value``."""

    def __init__(self, key, value):
        super().__init__(key, value, operator.ne, "!=")


class ConstraintAdapter:
    """For GTC

    Parameters
    ----------
    key : str
        Name of the tag.
    value
        Value compared with the tag.
    oper : callable
        Comparison operator, from the ``operator`` module.
    """

    def __init__(self, key, value, oper):
        self.key = key
        self.value = value
        self.type = type(value)
        self.oper = oper

    def __repr__(self):
        return f"ConstraintAdapter(key={self.key}, value={self.value}, {self.oper}, type={self.type})"


def adapter(tree):
    """Return a ``ConstraintAdapter`` for a node whose children are all leaves, otherwise None.

    The key is the name of the left operand and the value the value of the
    right operand.
    """
    if tree.nodes and all(node.is_terminal() for node in tree.nodes):
        return ConstraintAdapter(tree.lhs.name, tree.rhs.value, tree.operator)


def query_expr_from_attr(attrs):
    """Build the query expression of the tag descriptions `attrs`.

    Each tag must be equal to the placeholder of the same name, and the
    conditions are joined with AND. Without tags, ``ConstExprTrue``.
    The elements of `attrs` have the attributes ``name``, ``type`` and
    ``description``.
    """
    # Create a query descriptor from a sequence for fields
    if len(attrs) == 0:
        return ConstExprTrue
    exprs = []
    # for name, dtype in descs:
    for attr in attrs:
        metadata = {"type": attr.type, "description": attr.description}
        lhs = TagRepr(attr.name, metadata=metadata)
        rhs = Placeholder(attr.name)
        expr = lhs == rhs
        exprs.append(expr)
    return reduce(operator.and_, exprs)
