from components.cnf import CNF
from components.predicates.base import BinaryPredicate
from components.instances import Var, Constant
from components.predicates.full import FALSE, TRUE
from context import Symbol, Context


class Cons(BinaryPredicate):

    @staticmethod
    def _double_var_encode(
        var_one: Var,
        var_two: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        for i in range(context.DIM):
            cnf.extend([
                [
                    -context.V[(var_one.name, i, Symbol.ONE)],
                    -context.V[(var_two.name, i, Symbol.ZERO)]
                ],
                [
                    -context.V[(var_one.name, i, Symbol.ZERO)],
                    -context.V[(var_two.name, i, Symbol.ONE)]
                ]
            ])
        return cnf

    @staticmethod
    def _var_constant_encode(
        var: Var,
        constant: Constant,
        context: Context
    ) -> CNF:
        clauses = []
        for i, symbol in enumerate(constant.value):
            if symbol == Symbol.ONE:
                clauses.append([-context.V[(var.name, i, Symbol.ZERO)]])
            elif symbol == Symbol.ZERO:
                clauses.append([-context.V[(var.name, i, Symbol.ONE)]])
        if len(clauses) == 0:
            return CNF()
        return CNF(from_clauses=clauses)

    @staticmethod
    def _constant_var_encode(
        constant: Constant,
        var: Var,
        context: Context
    ) -> CNF:
        return Cons._var_constant_encode(var, constant, context)

    @staticmethod
    def _double_constant_encode(
        constant_one: Constant,
        constant_two: Constant,
        context: Context
    ) -> CNF:
        for i in range(context.DIM):
            if (
                constant_one.value[i] != Symbol.BOT and
                constant_two.value[i] != Symbol.BOT and
                constant_one.value[i] != constant_two.value[i]
            ):
                return CNF(from_clauses=[[]])
        return CNF()

    def simplify(self):
        if (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant)
        ):
            for i in range(self.context.DIM):
                if (
                    self.instance_one.value[i] != Symbol.BOT and
                    self.instance_two.value[i] != Symbol.BOT and
                    self.instance_one.value[i] != self.instance_two.value[i]
                ):
                    return FALSE()
            return TRUE()
        return self
