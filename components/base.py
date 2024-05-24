from components.cnf import CNF
from context import Symbol, Context, contextualize


class Component:

    def encode(self) -> CNF:
        return CNF()

    def set_context(self, context):
        self.context = context

    def get_children(self):
        return []

    def simplify(self):
        return self

    @staticmethod
    def generate_variables(var_name: str, context: Context) -> None:
        for i in range(context.DIM):
            for symbol in Symbol:
                context.V[(var_name, i, symbol)] = context.TOPV + 1
                context.TOPV += 1


class TRUE(Component):

    def encode(self) -> CNF:
        return CNF()


class FALSE(Component):

    def encode(self) -> CNF:
        return CNF(from_clauses=[[]])


def generate_cnf(
    formula: Component,
    context: Context
) -> CNF:
    contextualize(formula, context)
    cnf = formula.encode()
    # cnf.extend(context.CNF.clauses)
    return cnf


def generate_simplified_cnf(
    formula: Component,
    context: Context
) -> CNF:
    contextualize(formula, context)
    formula = formula.simplify()
    cnf = formula.encode()
    # cnf.extend(context.CNF.clauses)
    return cnf
