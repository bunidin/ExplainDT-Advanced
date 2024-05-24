from components.cnf import CNF
from functools import reduce
from components.base import TRUE, FALSE
from components.predicates.base import UnaryPredicate
from components.instances import Var, Constant
from context import Symbol, Context


class Full(UnaryPredicate):

    @staticmethod
    def _var_encode(var: Var, context: Context) -> CNF:
        no_bottoms = [
            [-context.V[(var.name, i, Symbol.BOT)]]
            for i in range(context.DIM)
        ]
        return CNF(from_clauses=no_bottoms)

    @staticmethod
    def _constant_encode(constant: Constant, context: Context) -> CNF:
        no_bottoms = reduce(
            lambda x, y: x and y,
            map(lambda x: x != Symbol.BOT, constant.value)
        )
        if not no_bottoms:
            return CNF(from_clauses=[[]])
        return CNF()

    def simplify(self):
        if isinstance(self.instance, Var):
            return self
        no_bottoms = reduce(
            lambda x, y: x and y,
            map(lambda x: x != Symbol.BOT, self.instance.value)
        )
        if no_bottoms:
            return TRUE()
        return FALSE()
