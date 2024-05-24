from components.cnf import CNF
from components.predicates.base import UnaryPredicate
from components.instances import Var, Constant
from context.model import FeatNode
from context import Symbol, Context


class IsNode(UnaryPredicate):

    @staticmethod
    def _generate_reachability_variables(var: Var, context: Context) -> None:
        for node in context.TREE.nodes():
            context.add_var(('r', var.name, node.id))

    @staticmethod
    def _var_encode(var: Var, context: Context) -> CNF:
        cnf = CNF()
        # Generate variables
        IsNode._generate_reachability_variables(var, context)
        # Every feature that is not BOT must be decided on
        # x_i != bot -> r_x_n ^ n.label = i
        label_clauses = [
            [context.V[(var.name, i, Symbol.BOT)]]
            for i in range(context.DIM)
        ]
        for node in context.TREE.nodes():
            if not isinstance(node, FeatNode):
                continue
            label_clauses[node.label].append(
                context.V[('r', var.name, node.id)]
            )
        cnf.extend(label_clauses)
        # Root is always reachable
        cnf.append(
            [
                context.V[(
                    'r',
                    var.name,
                    context.TREE.root.id,
                )]
            ],
            in_consistency=True
        )
        # Add propagation clauses
        for node in context.TREE.nodes():
            if not isinstance(node, FeatNode):
                continue
            # r_x_i ^ x_i = 0 -> r_x_i.zero
            cnf.append(
                [
                    -context.V[(var.name, node.label, Symbol.ZERO)],
                    -context.V[('r', var.name, node.id)],
                    context.V[('r', var.name, node.child_zero.id)]
                ],
                in_consistency=True
            )
            # r_x_i.zero -> r_x_i ^ x_i = zero
            cnf.append(
                [
                    -context.V[('r', var.name, node.child_zero.id)],
                    context.V[('r', var.name, node.id)]
                ],
                in_consistency=True
            )
            cnf.append(
                [
                    -context.V[('r', var.name, node.child_zero.id)],
                    context.V[(var.name, node.label, Symbol.ZERO)]
                ],
                in_consistency=True
            )
            # r_x_i ^ x_i = 1 -> r_x_i.one
            cnf.append(
                [
                    -context.V[(var.name, node.label, Symbol.ONE)],
                    -context.V[('r', var.name, node.id)],
                    context.V[('r', var.name, node.child_one.id)]
                ],
                in_consistency=True
            )
            # r_x_i.one -> r_x_i ^ x_i = one
            cnf.append(
                [
                    -context.V[('r', var.name, node.child_one.id)],
                    context.V[('r', var.name, node.id)]
                ],
                in_consistency=True
            )
            cnf.append(
                [
                    -context.V[('r', var.name, node.child_one.id)],
                    context.V[(var.name, node.label, Symbol.ONE)]
                ],
                in_consistency=True
            )
        return cnf

    @staticmethod
    def _constant_encode(constant: Constant, context: Context) -> CNF:
        cnf = CNF()
        raise Exception('Not implemented yet')
        return cnf
