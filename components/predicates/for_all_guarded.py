from components.cnf import CNF
from components.base import Component
from components.instances import GuardedVar
from context.model import FeatNode, Node
from context import Symbol


class ForAllGuarded(Component):

    def __init__(self, instance: GuardedVar, component: Component):
        self.instance = instance
        self.child = component

    @staticmethod
    def traverse_nodes(node: Node, value: list):
        yield value
        if not isinstance(node, FeatNode):
            return
        value[node.label] = Symbol.ZERO
        yield from ForAllGuarded.traverse_nodes(node.child_zero, value)
        value[node.label] = Symbol.ONE
        yield from ForAllGuarded.traverse_nodes(node.child_one, value)
        value[node.label] = Symbol.BOT

    @staticmethod
    def set_variable(
        component: Component,
        var_name: str,
        value: tuple
    ):
        for child in component.get_children():
            if (
                hasattr(child, 'instance') and
                isinstance(child.instance, GuardedVar) and
                child.instance.name == var_name
            ):
                child.instance.value = value
            elif (
                hasattr(child, 'instance_one') and
                isinstance(child.instance_one, GuardedVar) and
                child.instance_one.name == var_name
            ):
                child.instance_one.value = value
            elif (
                hasattr(child, 'instance_two') and
                isinstance(child.instance_two, GuardedVar) and
                child.instance_two.name == var_name
            ):
                child.instance_two.value = value
            ForAllGuarded.set_variable(child, var_name, value)

    def encode(self) -> CNF:
        cnf = CNF()
        node_value = [Symbol.BOT for _ in range(self.context.DIM)]
        for value in self.traverse_nodes(
            self.context.TREE.root,
            node_value
        ):
            self.set_variable(self.child, self.instance.name, tuple(value))
            cnf = cnf.conjunction(self.child.encode())
        return cnf

    def get_children(self) -> list[Component]:
        return [self.child]
