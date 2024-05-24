from components.cnf import CNF
from context import Symbol, Context
from typing import Optional


class Var:

    def __init__(self, name: str):
        self.name = name

    def generate_variables(self, context: Context) -> None:
        for i in range(context.DIM):
            for symbol in Symbol:
                context.V[(self.name, i, symbol)] = context.TOPV + 1
                context.TOPV += 1

    def encode_instance(self, context: Context) -> CNF:
        if (self.name, 0, Symbol.BOT) in context.V.keys():
            return CNF()
        self.generate_variables(context)

        cnf = CNF()
        at_least_one_value_per_feature = [
            [
                context.V[(self.name, i, symbol)]
                for symbol in Symbol
            ]
            for i in range(context.DIM)
        ]
        # For each feature add sanity constrains
        for i in range(context.DIM):
            sanity_clauses = [
                [
                    -context.V[(self.name, i, Symbol.ZERO)],
                    -context.V[(self.name, i, Symbol.ONE)]
                ],
                [
                    -context.V[(self.name, i, Symbol.ZERO)],
                    -context.V[(self.name, i, Symbol.BOT)]
                ],
                [
                    -context.V[(self.name, i, Symbol.ONE)],
                    -context.V[(self.name, i, Symbol.ZERO)]
                ],
                [
                    -context.V[(self.name, i, Symbol.ONE)],
                    -context.V[(self.name, i, Symbol.BOT)]
                ],
                [
                    -context.V[(self.name, i, Symbol.BOT)],
                    -context.V[(self.name, i, Symbol.ZERO)]
                ],
                [
                    -context.V[(self.name, i, Symbol.BOT)],
                    -context.V[(self.name, i, Symbol.ONE)]
                ],
            ]
            cnf.extend(sanity_clauses, in_consistency=True)
        cnf.extend(at_least_one_value_per_feature, in_consistency=True)
        return cnf


class Constant:

    def __init__(self, value: tuple):
        self.value = value

    def __repr__(self):
        return ''.join(map(lambda x: str(x), self.value))

    def __str__(self):
        return ''.join(map(lambda x: str(x), self.value))


class GuardedVar(Constant):

    def __init__(self, name: str):
        self.name: str = name
        self.value: Optional[tuple] = None
