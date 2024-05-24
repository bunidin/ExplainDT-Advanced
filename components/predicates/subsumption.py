from components.cnf import CNF
from components.predicates.base import BinaryPredicate
from components.instances import Var, Constant
from components.base import FALSE, TRUE
from context import Symbol, Context


class Subsumption(BinaryPredicate):

    @staticmethod
    def _double_var_encode(
        var_one: Var,
        var_two: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        for i in range(context.DIM):
            cnf.append([
                -context.V[(var_one.name, i, Symbol.ONE)],
                context.V[(var_two.name, i, Symbol.ONE)]
            ])
            cnf.append([
                -context.V[(var_one.name, i, Symbol.ZERO)],
                context.V[(var_two.name, i, Symbol.ZERO)]
            ])
        return cnf

    @staticmethod
    def _var_constant_encode(
        var: Var,
        constant: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        for i, symbol in enumerate(constant.value):
            if symbol == Symbol.ONE:
                cnf.append([-context.V[(var.name, i, Symbol.ZERO)]])
            elif symbol == Symbol.ZERO:
                cnf.append([-context.V[(var.name, i, Symbol.ONE)]])
            else:
                cnf.extend([
                    [-context.V[(var.name, i, Symbol.ONE)]],
                    [-context.V[(var.name, i, Symbol.ZERO)]],
                ])
        return cnf

    @staticmethod
    def _constant_var_encode(
        constant: Constant,
        var: Var,
        context: Context
    ) -> CNF:
        clauses = []
        for i, symbol in enumerate(constant.value):
            if symbol != Symbol.BOT:
                clauses.append([context.V[(var.name, i, symbol)]])
        return CNF(from_clauses=clauses)

    @staticmethod
    def _double_constant_encode(
        constant_one: Constant,
        constant_two: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        for s_one, s_two in zip(constant_one.value, constant_two.value):
            if s_one != Symbol.BOT and s_one != s_two:
                cnf = cnf.negate()
                return cnf
        return cnf

    def simplify(self):
        if (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant)
        ):
            for s_one, s_two in zip(
                self.instance_one.value,
                self.instance_two.value
            ):
                if s_one != Symbol.BOT and s_one != s_two:
                    return FALSE()
            return TRUE()
        return self
