from components.cnf import CNF
from components.predicates.base import BinaryPredicate
from components.instances import Var, Constant
from components.predicates.full import FALSE, TRUE
from context import Symbol, Context


class LEL(BinaryPredicate):

    @staticmethod
    def _generate_variables(var: Var, context: Context):
        for i in range(context.DIM):
            for j in range(context.DIM + 1):
                context.add_var(('c', var.name, i, j))

    @staticmethod
    def _generate_counting_clauses(
        var: Var,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        for i in range(1, context.DIM):
            # c_x_i_0 <-> (c_x_i-1_0 ^ x_i != bot)
            clauses.extend([
                [
                    -context.V[('c', var.name, i, 0)],
                    -context.V[(var.name, i, Symbol.BOT)]
                ],
                [
                    -context.V[('c', var.name, i, 0)],
                    context.V[('c', var.name, i - 1, 0)]
                ],
                [
                    -context.V[('c', var.name, i - 1, 0)],
                    context.V[(var.name, i, Symbol.BOT)],
                    context.V[('c', var.name, i, 0)]
                ],
            ])
            for j in range(1, i + 2):
                # c_x_i_j <-> (c_x_i-1_j-1 ^ x_i = b) v (c_x_i-1_j ^ x_i != b)
                clauses.extend([
                    [
                        -context.V[('c', var.name, i, j)],
                        context.V[('c', var.name, i - 1, j - 1)],
                        -context.V[(var.name, i, Symbol.BOT)]
                    ],
                    [
                        context.V[('c', var.name, i, j)],
                        -context.V[('c', var.name, i - 1, j - 1)],
                        -context.V[(var.name, i, Symbol.BOT)]
                    ],
                    [
                        -context.V[('c', var.name, i, j)],
                        context.V[('c', var.name, i - 1, j)],
                        context.V[(var.name, i, Symbol.BOT)]
                    ],
                    [
                        context.V[('c', var.name, i, j)],
                        -context.V[('c', var.name, i - 1, j)],
                        context.V[(var.name, i, Symbol.BOT)]
                    ],
                ])
        # -c_x_i_j if j > i + 1
        clauses.extend([
            [-context.V[('c', var.name, i, j)]]
            for i in range(context.DIM)
            for j in range(i + 2, context.DIM + 1)
        ])
        clauses.extend([
            # c_x_1_1 <-> x_1 = bot
            [
                -context.V[('c', var.name, 0, 1)],
                context.V[(var.name, 0, Symbol.BOT)]
            ],
            [
                -context.V[(var.name, 0, Symbol.BOT)],
                context.V[('c', var.name, 0, 1)]
            ],
            # c_x_1_0 <-> x_1 != bot
            [
                -context.V[('c', var.name, 0, 0)],
                -context.V[(var.name, 0, Symbol.BOT)]
            ],
            [
                context.V[(var.name, 0, Symbol.BOT)],
                context.V[('c', var.name, 0, 0)]
            ]
        ])
        return clauses

    @staticmethod
    def _double_var_encode(
        var_one: Var,
        var_two: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        LEL._generate_variables(var_one, context)
        LEL._generate_variables(var_two, context)
        cnf.extend(
            LEL._generate_counting_clauses(var_one, context),
            in_consistency=True
        )
        cnf.extend(
            LEL._generate_counting_clauses(var_two, context),
            in_consistency=True
        )
        # If we see a number of bots in x then we must see less or equal on y
        for i in range(context.DIM):
            less_or_equal_bots_in_two_as_in_one = [
                -context.V[('c', var_one.name, context.DIM - 1, i)]
            ]
            less_or_equal_bots_in_two_as_in_one.extend([
                context.V[('c', var_two.name, context.DIM - 1, j)]
                for j in range(i + 1)
            ])
            cnf.extend([less_or_equal_bots_in_two_as_in_one])
        return cnf

    @staticmethod
    def _var_constant_encode(
        var: Var,
        constant: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        LEL._generate_variables(var, context)
        cnf.extend(
            LEL._generate_counting_clauses(var, context),
            in_consistency=True
        )
        bots_in_constant = sum(
            map(lambda x: int(x == Symbol.BOT), constant.value)
        )
        cnf.extend([
            [-context.V[('c', var.name, context.DIM - 1, j)]]
            for j in range(bots_in_constant)
        ])
        return cnf

    @staticmethod
    def _constant_var_encode(
        constant: Constant,
        var: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        LEL._generate_variables(var, context)
        cnf.extend(
            LEL._generate_counting_clauses(var, context),
            in_consistency=True
        )
        bots_in_constant = sum(
            map(lambda x: int(x == Symbol.BOT), constant.value)
        )
        cnf.extend([
            [-context.V[('c', var.name, context.DIM - 1, j)]]
            for j in range(bots_in_constant + 1, context.DIM + 1)
        ])
        return cnf

    @staticmethod
    def _double_constant_encode(
        constant_one: Constant,
        constant_two: Constant,
        context: Context
    ) -> CNF:
        constant_one_level = sum(
            map(lambda x: int(x == Symbol.BOT), constant_one.value)
        )
        constant_two_level = sum(
            map(lambda x: int(x == Symbol.BOT), constant_two.value)
        )
        if constant_one_level < constant_two_level:
            return CNF(from_clauses=[[]])
        return CNF()

    def simplify(self):
        if (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant)
        ):
            constant_one_level = sum(
                map(lambda x: int(x == Symbol.BOT), self.instance_one.value)
            )
            constant_two_level = sum(
                map(lambda x: int(x == Symbol.BOT), self.instance_two.value)
            )
            if constant_one_level < constant_two_level:
                return FALSE()
            return TRUE()
        return self
