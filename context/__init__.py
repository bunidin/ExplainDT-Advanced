from enum import Enum
from typing import Optional
from context.model import Tree, FeatNode, Leaf, Node  # NOQA


RESERVER_VAR_NAME = 'reserved_var'


class Context:

    def __init__(
        self,
        dimension: int,
        tree: Tree,
        vars: Optional[dict] = None,
        topv: Optional[int] = None
    ):
        self.V = vars if vars is not None else dict()
        self.TOPV = topv if topv is not None else 0
        self.DIM: int = dimension
        self.TREE: Tree = tree

    def add_var(self, var_key):
        if var_key in self.V.keys():
            return
        self.V[var_key] = self.TOPV + 1
        self.TOPV += 1


def contextualize(component, context):
    component.set_context(context)
    for child in component.get_children():
        contextualize(child, context)


class Symbol(Enum):
    ZERO = 0
    ONE = 1
    BOT = 2

    def __repr__(self):
        return '_' if self._value_ == 2 else str(self._value_)

    def __str__(self):
        return '_' if self._value_ == 2 else str(self._value_)

    def __int__(self):
        return self._value_
