from components.cnf import CNF
from components.base import Component, TRUE, FALSE
from components.instances import Var


class Not(Component):

    def __init__(self, component: Component):
        self.child = component

    def encode(self) -> CNF:
        negation = self.child.encode().negate(topv=self.context.TOPV)
        self.context.TOPV = max(self.context.TOPV, negation.nv)
        return negation

    def simplify(self):
        simple_child = self.child.simplify()
        if isinstance(simple_child, TRUE):
            return FALSE()
        elif isinstance(simple_child, FALSE):
            return TRUE()
        elif isinstance(simple_child, Not):
            return simple_child.child
        else:
            simplified_self = Not(simple_child)
            simplified_self.context = self.context
            return simplified_self

    def get_children(self):
        return [self.child]


class Exists(Component):

    def __init__(self, instance: Var, component: Component):
        self.instance = instance
        self.child = component

    def encode(self) -> CNF:
        cnf = CNF()
        cnf = cnf.conjunction(self.instance.encode_instance(self.context))
        cnf = cnf.conjunction(self.child.encode())
        # self.context.CNF.extend(
        #     self.instance.encode_instance(self.context).clauses
        # )
        # cnf.extend(self.child.encode().clauses)
        return cnf

    def simplify(self):
        simple_child = self.child.simplify()
        if isinstance(self.child, TRUE) or isinstance(self.child, FALSE):
            return self.child
        simplified_self = Exists(self.instance, simple_child)
        simplified_self.context = self.context
        return simplified_self

    def get_children(self):
        return [self.child]


class And(Component):

    def __init__(
        self,
        first_component: Component,
        second_component: Component
    ):
        self.first_child = first_component
        self.second_child = second_component

    def encode(self) -> CNF:
        cnf = self.first_child.encode()
        cnf = cnf.conjunction(self.second_child.encode())
        return cnf

    def simplify(self) -> Component:
        simple_child_one = self.first_child.simplify()
        simple_child_two = self.second_child.simplify()
        child_one_true = isinstance(simple_child_one, TRUE)
        child_two_true = isinstance(simple_child_two, TRUE)
        child_one_false = isinstance(simple_child_one, FALSE)
        child_two_false = isinstance(simple_child_two, FALSE)
        if child_one_true:
            return simple_child_two
        elif child_two_true:
            return simple_child_one
        elif child_one_false or child_two_false:
            return FALSE()
        simplified_self = And(simple_child_one, simple_child_two)
        simplified_self.context = self.context
        return simplified_self

    def get_children(self):
        return [self.first_child, self.second_child]


class Or(Component):

    def __init__(
        self,
        first_component: Component,
        second_component: Component
    ):
        self.first_child = first_component
        self.second_child = second_component

    def encode(self) -> CNF:
        # # Negate children
        # first_child_cnf = self.first_child.encode().negate(
        #     topv=self.context.TOPV
        # )
        # self.context.TOPV = max(self.context.TOPV, first_child_cnf.nv)
        # second_child_cnf = self.second_child.encode().negate(
        #     topv=self.context.TOPV
        # )
        # self.context.TOPV = max(self.context.TOPV, second_child_cnf.nv)
        # # Apply and
        # cnf.extend(first_child_cnf.clauses)
        # cnf.extend(second_child_cnf.clauses)
        # # Negate
        # cnf = cnf.negate(topv=self.context.TOPV)
        # self.context.TOPV = max(self.context.TOPV, cnf.nv)

        # Negate children
        first_child_cnf = self.first_child.encode().negate(
            topv=self.context.TOPV
        )
        self.context.TOPV = max(self.context.TOPV, first_child_cnf.nv)
        second_child_cnf = self.second_child.encode().negate(
            topv=self.context.TOPV
        )
        self.context.TOPV = max(self.context.TOPV, second_child_cnf.nv)
        # Apply and
        cnf = first_child_cnf.conjunction(second_child_cnf)
        # Negate result
        cnf = cnf.negate(topv=self.context.TOPV)
        self.context.TOPV = max(self.context.TOPV, cnf.nv)
        return cnf

    def simplify(self) -> Component:
        simple_child_one = self.first_child.simplify()
        simple_child_two = self.second_child.simplify()
        child_one_true = isinstance(simple_child_one, TRUE)
        child_two_true = isinstance(simple_child_two, TRUE)
        child_one_false = isinstance(simple_child_one, FALSE)
        child_two_false = isinstance(simple_child_two, FALSE)
        if child_one_true:
            return simple_child_one
        elif child_two_true:
            return simple_child_two
        elif child_one_false:
            return simple_child_two
        elif child_two_false:
            return simple_child_one
        simplified_self = Or(simple_child_one, simple_child_two)
        simplified_self.context = self.context
        return simplified_self

    def get_children(self):
        return [self.first_child, self.second_child]


# class Constant(Component):

#     def __init__(self, var_name: str, value: tuple):
#         self.var_name = var_name
#         self.value = value

#     def encode(self):
#         self.generate_variables(self.var_name, self.context)
#         # Initialize clauses as the assignment of attributes
#         clauses = [
#             [self.context.V[(self.var_name, i, self.value[i])]]
#             for i in range(self.context.DIM)
#         ]
#         # For each dimention add sanity constrains
#         for i in range(self.context.DIM):
#             sanity_clauses = [
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.ZERO)],
#                     -self.context.V[(self.var_name, i, Symbol.ONE)]
#                 ],
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.ZERO)],
#                     -self.context.V[(self.var_name, i, Symbol.BOT)]
#                 ],
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.ONE)],
#                     -self.context.V[(self.var_name, i, Symbol.ZERO)]
#                 ],
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.ONE)],
#                     -self.context.V[(self.var_name, i, Symbol.BOT)]
#                 ],
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.BOT)],
#                     -self.context.V[(self.var_name, i, Symbol.ZERO)]
#                 ],
#                 [
#                     -self.context.V[(self.var_name, i, Symbol.BOT)],
#                     -self.context.V[(self.var_name, i, Symbol.ONE)]
#                 ],
#             ]
#             clauses.extend(sanity_clauses)
#         return CNF(from_clauses=clauses)
