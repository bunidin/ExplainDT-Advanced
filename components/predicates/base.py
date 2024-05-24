from components.cnf import CNF
from components.base import Component
from components.instances import Var, Constant
from context import Context


class UnaryPredicate(Component):

    def __init__(self, instance: Var | Constant):
        self.instance = instance

    @staticmethod
    def _var_encode(var: Var, context: Context) -> CNF:
        raise Exception('Not yet implemented')

    @staticmethod
    def _constant_encode(constant: Constant, context: Context) -> CNF:
        raise Exception('Not yet implemented')

    def encode(self):
        if isinstance(self.instance, Var):
            cnf = self._var_encode(self.instance, self.context)
        else:
            cnf = self._constant_encode(self.instance, self.context)
        return cnf


class BinaryPredicate(Component):

    def __init__(
        self,
        instance_one: Var | Constant,
        instance_two: Var | Constant
    ):
        self.instance_one = instance_one
        self.instance_two = instance_two

    @staticmethod
    def _double_var_encode(
        var_one: Var,
        var_two: Var,
        context: Context
    ) -> CNF:
        raise Exception('Not yet implemented')

    @staticmethod
    def _var_constant_encode(
        var: Var,
        constant: Constant,
        context: Context
    ) -> CNF:
        raise Exception('Not yet implemented')

    @staticmethod
    def _constant_var_encode(
        constant: Constant,
        var: Var,
        context: Context
    ) -> CNF:
        raise Exception('Not yet implemented')

    @staticmethod
    def _double_constant_encode(
        constant_one: Constant,
        constant_two: Constant,
        context: Context
    ) -> CNF:
        raise Exception('Not yet implemented')

    def encode(self) -> CNF:
        cnf = CNF()
        if (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Var)
        ):
            cnf = self._double_var_encode(
                self.instance_one,
                self.instance_two,
                self.context
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Constant)
        ):
            cnf = self._var_constant_encode(
                self.instance_one,
                self.instance_two,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Var)
        ):
            cnf = self._constant_var_encode(
                self.instance_one,
                self.instance_two,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant)
        ):
            cnf = self._double_constant_encode(
                self.instance_one,
                self.instance_two,
                self.context
            )
        return cnf
