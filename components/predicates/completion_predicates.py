from components.cnf import CNF
from components.base import Component, TRUE, FALSE
from components.instances import Var, Constant
from context import Symbol, Context
from context.model import FeatNode, Leaf


class CompletionPredicate(Component):

    def __init__(self, instance: Var | Constant, leaf_truth_value: bool):
        self.instance = instance
        self.leaf_truth_value = leaf_truth_value

    @staticmethod
    def _generate_reachability_variables(var: Var, context: Context) -> None:
        for node in context.TREE.nodes():
            context.add_var(('r', var.name, node.id))

    @staticmethod
    def _var_encode(var: Var, leaf_truth_value: bool, context: Context) -> CNF:
        # Generate cnf
        cnf = CNF()
        # Generate tree reachability clasuses
        CompletionPredicate._generate_reachability_variables(var, context)
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
            if node.is_leaf and node.is_poss_leaf != leaf_truth_value:
                cnf.append(
                    [-context.V[('r', var.name, node.id)]]
                )
            elif isinstance(node, FeatNode):
                # r_x_i ^ x_i = 0 -> r_x_i.zero
                cnf.append(
                    [
                        -context.V[(var.name, node.label, Symbol.ZERO)],
                        -context.V[('r', var.name, node.id)],
                        context.V[('r', var.name, node.child_zero.id)]
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
                # r_x_i ^ x_i = bot -> r_x_i.one ^ r_x_i.zero
                cnf.append(
                    [
                        -context.V[(var.name, node.label, Symbol.BOT)],
                        -context.V[('r', var.name, node.id)],
                        context.V[('r', var.name, node.child_zero.id)]
                    ],
                    in_consistency=True
                )
                cnf.append(
                    [
                        -context.V[(var.name, node.label, Symbol.BOT)],
                        -context.V[('r', var.name, node.id)],
                        context.V[('r', var.name, node.child_one.id)]
                    ],
                    in_consistency=True
                )
                # r_x_i.one -> r_x_i ^ (x_i = bot v x_i = 1)
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
                        context.V[(var.name, node.label, Symbol.ONE)],
                        context.V[(var.name, node.label, Symbol.BOT)]
                    ],
                    in_consistency=True
                )
                # r_x_i.zero -> r_x_i ^ (x_i = bot v x_i = 0)
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
                        context.V[(var.name, node.label, Symbol.ZERO)],
                        context.V[(var.name, node.label, Symbol.BOT)]
                    ],
                    in_consistency=True
                )
        return cnf

    @staticmethod
    def _constant_encode(
        constant: Constant,
        leaf_truth_value: bool,
        context: Context
    ) -> CNF:
        node_stack = [context.TREE.root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            if isinstance(node, Leaf):
                if node.truth_value != leaf_truth_value:
                    return CNF(from_clauses=[[]])
            elif isinstance(node, FeatNode):
                if constant.value[node.label] == Symbol.BOT:
                    node_stack.extend([node.child_zero, node.child_one])
                elif constant.value[node.label] == Symbol.ONE:
                    node_stack.append(node.child_one)
                else:
                    node_stack.append(node.child_zero)
        return CNF()

    def simplify(self):
        if isinstance(self.instance, Var):
            return self
        node_stack = [self.context.TREE.root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            if isinstance(node, Leaf):
                if node.truth_value != self.leaf_truth_value:
                    return FALSE()
            elif isinstance(node, FeatNode):
                if self.instance.value[node.label] == Symbol.BOT:
                    node_stack.extend([node.child_zero, node.child_one])
                elif self.instance.value[node.label] == Symbol.ONE:
                    node_stack.append(node.child_one)
                else:
                    node_stack.append(node.child_zero)
        return TRUE()

    def encode(self) -> CNF:
        if isinstance(self.instance, Var):
            cnf = self._var_encode(
                self.instance,
                self.leaf_truth_value,
                self.context
            )
        else:
            cnf = self._constant_encode(
                self.instance,
                self.leaf_truth_value,
                self.context
            )
        return cnf


class AllPoss(CompletionPredicate):

    def __init__(self, instance: Var | Constant):
        super().__init__(instance, True)


class AllNeg(CompletionPredicate):

    def __init__(self, instance: Var | Constant):
        super().__init__(instance, False)
