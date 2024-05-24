from components.cnf import CNF
from functools import reduce
from components.base import Component, TRUE, FALSE
from components.instances import Var, Constant
from context import Symbol, Context


class LEH(Component):

    def __init__(
        self,
        instance_one: Var | Constant,
        instance_two: Var | Constant,
        instance_three: Var | Constant
    ):
        self.instance_one = instance_one
        self.instance_two = instance_two
        self.instance_three = instance_three

    @staticmethod
    def _generate_distance_variables(name_one, name_two, context: Context):
        for i in range(context.DIM):
            for j in range(context.DIM + 1):
                context.add_var(('d', name_one, name_two, i, j))

    @staticmethod
    def _generate_var_is_full_clauses(
        var: Var,
        context: Context
    ) -> list[list[int]]:
        return [
            [-context.V[(var.name, i, Symbol.BOT)]]
            for i in range(context.DIM)
        ]

    @staticmethod
    def _constant_is_full(constant: Constant) -> bool:
        return reduce(
            lambda x, y: x and y,
            map(lambda x: x != Symbol.BOT, constant.value)
        )

    @staticmethod
    def _generate_var_constant_distance_clauses(
        var: Var,
        constant: Constant,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        constant_name = ''.join(map(lambda x: str(x), constant.value))
        # NOTE: ForAll i, v_i != bot & c_i != bot because they are both full
        clauses.extend([
            # d_v_c_1_0 <-> v_1 = c_1
            [
                -context.V[('d', var.name, constant_name, 0, 0)],
                context.V[(var.name, 0, constant.value[0])]
            ],
            [
                -context.V[(var.name, 0, constant.value[0])],
                context.V[('d', var.name, constant_name, 0, 0)]
            ],
            # d_v_c_1_1 <-> v_1 != c_1
            [
                -context.V[('d', var.name, constant_name, 0, 1)],
                -context.V[(var.name, 0, constant.value[0])]
            ],
            [
                context.V[(var.name, 0, constant.value[0])],
                context.V[('d', var.name, constant_name, 0, 1)]
            ],
        ])
        for i in range(1, context.DIM):
            # d_v_c_i_0 <-> (d_v_c_i-1_0 ^ v_i = c_i)
            clauses.extend([
                [
                    -context.V[('d', var.name, constant_name, i, 0)],
                    context.V[('d', var.name, constant_name, i - 1, 0)]
                ],
                [
                    -context.V[('d', var.name, constant_name, i, 0)],
                    context.V[(var.name, i, constant.value[i])]
                ],
                [
                    -context.V[('d', var.name, constant_name, i - 1, 0)],
                    -context.V[(var.name, i, constant.value[i])],
                    context.V[('d', var.name, constant_name, i, 0)],
                ]
            ])
            for j in range(1, i + 2):
                # d_v_c_i_j <->
                # (d_v_c_i-1_j-1 ^ v_i != c_i) v (d_v_c_i-1_j ^ v_i = c_i)

                # d_v_c_i_j = a
                # d_v_c_i-1_j-1 = b
                # d_v_c_i-1_j = c
                # v_i != c_i = d
                clauses.extend([
                    [
                        -context.V[('d', var.name, constant_name, i, j)],
                        context.V[('d', var.name, constant_name, i-1, j-1)],
                        context.V[(var.name, i, constant.value[i])]
                    ],
                    [
                        -context.V[('d', var.name, constant_name, i, j)],
                        context.V[('d', var.name, constant_name, i-1, j)],
                        -context.V[(var.name, i, constant.value[i])]
                    ],
                    [
                        context.V[('d', var.name, constant_name, i, j)],
                        -context.V[('d', var.name, constant_name, i-1, j-1)],
                        context.V[(var.name, i, constant.value[i])]
                    ],
                    [
                        context.V[('d', var.name, constant_name, i, j)],
                        -context.V[('d', var.name, constant_name, i-1, j)],
                        -context.V[(var.name, i, constant.value[i])]
                    ],
                ])
        # -d_x_i_j if j > i + 1
        clauses.extend([
            [-context.V[('d', var.name, constant_name, i, j)]]
            for i in range(context.DIM)
            for j in range(i + 2, context.DIM + 1)
        ])
        return clauses

    @staticmethod
    def _generate_var_equal_variables(
        var_one_name: str,
        var_two_name: str,
        context: Context
    ) -> None:
        for i in range(context.DIM):
            context.add_var(('eq', var_one_name, var_two_name, i))

    @staticmethod
    def _generate_var_equal_clauses(
        var_one_name: str,
        var_two_name: str,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        for i in range(context.DIM):
            clauses.extend(
                [
                    [
                        -context.V[(var_one_name, i, Symbol.ONE)],
                        -context.V[(var_two_name, i, Symbol.ONE)],
                        context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        -context.V[(var_one_name, i, Symbol.ONE)],
                        -context.V[(var_two_name, i, Symbol.ZERO)],
                        -context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        -context.V[(var_one_name, i, Symbol.ZERO)],
                        -context.V[(var_two_name, i, Symbol.ONE)],
                        -context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        -context.V[(var_one_name, i, Symbol.ZERO)],
                        -context.V[(var_two_name, i, Symbol.ZERO)],
                        context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                ]
            )
        return clauses

    @staticmethod
    def _generate_double_var_distance_clauses(
        var_one_name: str,
        var_two_name: str,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        clauses.extend([
            [
                -context.V[('d', var_one_name, var_two_name, 0, 0)],
                context.V[('eq', var_one_name, var_two_name, 0)]
            ],
            [
                -context.V[('eq', var_one_name, var_two_name, 0)],
                context.V[('d', var_one_name, var_two_name, 0, 0)],
            ],
            [
                -context.V[('d', var_one_name, var_two_name, 0, 1)],
                -context.V[('eq', var_one_name, var_two_name, 0)]
            ],
            [
                context.V[('eq', var_one_name, var_two_name, 0)],
                context.V[('d', var_one_name, var_two_name, 0, 1)],
            ],
        ])
        for i in range(1, context.DIM):
            clauses.extend([
                [
                    -context.V[('d', var_one_name, var_two_name, i, 0)],
                    context.V[('d', var_one_name, var_two_name, i - 1, 0)]
                ],
                [
                    -context.V[('d', var_one_name, var_two_name, i, 0)],
                    context.V[('eq', var_one_name, var_two_name, i)]
                ],
                [
                    -context.V[('d', var_one_name, var_two_name, i - 1, 0)],
                    -context.V[('eq', var_one_name, var_two_name, i)],
                    context.V[('d', var_one_name, var_two_name, i, 0)],
                ]
            ])
            for j in range(1, i + 2):
                clauses.extend([
                    [
                        -context.V[('d', var_one_name, var_two_name, i, j)],
                        context.V[('d', var_one_name, var_two_name, i-1, j-1)],
                        context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        -context.V[('d', var_one_name, var_two_name, i, j)],
                        context.V[('d', var_one_name, var_two_name, i-1, j)],
                        -context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        context.V[('d', var_one_name, var_two_name, i, j)],
                        -context.V[
                            ('d', var_one_name, var_two_name, i-1, j-1)
                        ],
                        context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                    [
                        context.V[('d', var_one_name, var_two_name, i, j)],
                        -context.V[('d', var_one_name, var_two_name, i-1, j)],
                        -context.V[('eq', var_one_name, var_two_name, i)]
                    ],
                ])
        clauses.extend([
            [-context.V[('d', var_one_name, var_two_name, i, j)]]
            for i in range(context.DIM)
            for j in range(i + 2, context.DIM + 1)
        ])
        return clauses

    @staticmethod
    def _triple_var_encode(
        instance_one: Var,
        instance_two: Var,
        instance_three: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = instance_one.name
        instance_two_name = instance_two.name
        instance_three_name = instance_three.name
        # Generate variables
        LEH._generate_var_equal_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_var_equal_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_one, context))
        cnf.extend(LEH._generate_var_is_full_clauses(instance_two, context))
        cnf.extend(LEH._generate_var_is_full_clauses(instance_three, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_equal_clauses(
                instance_one_name,
                instance_two_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_var_equal_clauses(
                instance_one_name,
                instance_three_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_double_var_distance_clauses(
                instance_one_name,
                instance_two_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_double_var_distance_clauses(
                instance_one_name,
                instance_three_name,
                context
            ),
            in_consistency=True
        )
        # Add restriction clausses
        for i in range(1, context.DIM + 1):
            cnf.extend([
                [
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_two_name,
                        context.DIM - 1,
                        i,
                    )],
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_three_name,
                        context.DIM - 1,
                        j,
                    )],
                ]
                for j in range(0, i)
            ])
        return cnf

    @staticmethod
    def _var_constant_constant_encode(
        instance_one: Var,
        instance_two: Constant,
        instance_three: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = instance_one.name
        instance_two_name = ''.join(map(lambda x: str(x), instance_two.value))
        instance_three_name = ''.join(
            map(lambda x: str(x), instance_three.value)
        )
        # Check for fullness in constants
        if not (
            LEH._constant_is_full(instance_two) and
            LEH._constant_is_full(instance_three)
        ):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_distance_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_one, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_one,
                instance_two,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_one,
                instance_three,
                context
            ),
            in_consistency=True
        )
        # Add restriction clausses
        for i in range(1, context.DIM + 1):
            cnf.extend([
                [
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_two_name,
                        context.DIM - 1,
                        i,
                    )],
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_three_name,
                        context.DIM - 1,
                        j,
                    )],
                ]
                for j in range(0, i)
            ])
        return cnf

    @staticmethod
    def _var_var_constant_encode(
        instance_one: Var,
        instance_two: Var,
        instance_three: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = instance_one.name
        instance_two_name = instance_two.name
        instance_three_name = ''.join(
            map(lambda x: str(x), instance_three.value)
        )
        # Check for fullness in constants
        if not LEH._constant_is_full(instance_three):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_var_equal_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_one, context))
        cnf.extend(LEH._generate_var_is_full_clauses(instance_two, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_equal_clauses(
                instance_one_name,
                instance_two_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_double_var_distance_clauses(
                instance_one_name,
                instance_two_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_one,
                instance_three,
                context
            ),
            in_consistency=True
        )
        # Add restriction clausses
        for i in range(1, context.DIM + 1):
            cnf.extend([
                [
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_two_name,
                        context.DIM - 1,
                        i,
                    )],
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_three_name,
                        context.DIM - 1,
                        j,
                    )],
                ]
                for j in range(0, i)
            ])
        return cnf

    @staticmethod
    def _var_constant_var_encode(
        instance_one: Var,
        instance_two: Constant,
        instance_three: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = instance_one.name
        instance_two_name = ''.join(
            map(lambda x: str(x), instance_two.value)
        )
        instance_three_name = instance_three.name
        # Check for fullness in constants
        if not LEH._constant_is_full(instance_two):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_var_equal_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_two_name,
            context
        )
        LEH._generate_distance_variables(
            instance_one_name,
            instance_three_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_one, context))
        cnf.extend(LEH._generate_var_is_full_clauses(instance_three, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_equal_clauses(
                instance_one_name,
                instance_three_name,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_one,
                instance_two,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_double_var_distance_clauses(
                instance_one_name,
                instance_three_name,
                context
            ),
            in_consistency=True
        )
        # Add restriction clausses
        for i in range(1, context.DIM + 1):
            cnf.extend([
                [
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_two_name,
                        context.DIM - 1,
                        i,
                    )],
                    -context.V[(
                        'd',
                        instance_one_name,
                        instance_three_name,
                        context.DIM - 1,
                        j,
                    )],
                ]
                for j in range(0, i)
            ])
        return cnf

    @staticmethod
    def _constant_var_var_encode(
        instance_one: Constant,
        instance_two: Var,
        instance_three: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = ''.join(map(lambda x: str(x), instance_one.value))
        instance_two_name = instance_two.name
        instance_three_name = instance_three.name
        # Check for fullness in constants
        if not LEH._constant_is_full(instance_one):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_distance_variables(
            instance_two_name,
            instance_one_name,
            context
        )
        LEH._generate_distance_variables(
            instance_three_name,
            instance_one_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_two, context))
        cnf.extend(LEH._generate_var_is_full_clauses(instance_three, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_two,
                instance_one,
                context
            ),
            in_consistency=True
        )
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_three,
                instance_one,
                context
            ),
            in_consistency=True
        )
        # Add restriction clausses
        for i in range(1, context.DIM + 1):
            cnf.extend([
                [
                    -context.V[(
                        'd',
                        instance_two_name,
                        instance_one_name,
                        context.DIM - 1,
                        i,
                    )],
                    -context.V[(
                        'd',
                        instance_three_name,
                        instance_one_name,
                        context.DIM - 1,
                        j,
                    )],
                ]
                for j in range(0, i)
            ])
        return cnf

    @staticmethod
    def _constant_constant_var_encode(
        instance_one: Constant,
        instance_two: Constant,
        instance_three: Var,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = ''.join(map(lambda x: str(x), instance_one.value))
        instance_three_name = instance_three.name
        # Check for fullness in constants
        if not (
            LEH._constant_is_full(instance_one) and
            LEH._constant_is_full(instance_two)
        ):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_distance_variables(
            instance_three_name,
            instance_one_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_three, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_three,
                instance_one,
                context
            ),
            in_consistency=True
        )
        # Determine hamming distance between instance_one and instance_two
        one_two_hamming_distance = sum(
            map(
                lambda i: int(instance_one.value[i] != instance_two.value[i]),
                (i for i in range(context.DIM))
            )
        )
        # Add restriction clausses
        if one_two_hamming_distance == 0:
            return CNF()
        cnf.extend([
            [-context.V[(
                'd',
                instance_three.name,
                instance_one_name,
                context.DIM - 1,
                i,
            )]]
            for i in range(one_two_hamming_distance)
        ])
        return cnf

    @staticmethod
    def _constant_var_constant_encode(
        instance_one: Constant,
        instance_two: Var,
        instance_three: Constant,
        context: Context
    ) -> CNF:
        cnf = CNF()
        instance_one_name = ''.join(map(lambda x: str(x), instance_one.value))
        instance_two_name = instance_two.name
        # Check for fullness in constants
        if not (
            LEH._constant_is_full(instance_one) and
            LEH._constant_is_full(instance_three)
        ):
            return CNF(from_clauses=[[]])
        # Generate variables
        LEH._generate_distance_variables(
            instance_two_name,
            instance_one_name,
            context
        )
        # Force fullness of variable
        cnf.extend(LEH._generate_var_is_full_clauses(instance_two, context))
        # Generate counting clauses
        cnf.extend(
            LEH._generate_var_constant_distance_clauses(
                instance_two,
                instance_one,
                context
            ),
            in_consistency=True
        )
        # Determine hamming distance between instance_one and instance_two
        one_three_hamming_distance = sum(
            map(
                lambda i: int(
                    instance_one.value[i] != instance_three.value[i]
                ),
                (i for i in range(context.DIM))
            )
        )
        # Add restriction clausses
        if one_three_hamming_distance == 0:
            return CNF()
        cnf.extend([
            [-context.V[(
                'd',
                instance_two.name,
                instance_one_name,
                context.DIM - 1,
                i,
            )]]
            for i in range(one_three_hamming_distance + 1, context.DIM + 1)
        ])
        return cnf

    @staticmethod
    def _triple_constant_encode(
        instance_one: Constant,
        instance_two: Constant,
        instance_three: Constant,
        context: Context
    ) -> CNF:
        # Check for fullness in constants
        if not (
            LEH._constant_is_full(instance_one) and
            LEH._constant_is_full(instance_two) and
            LEH._constant_is_full(instance_three)
        ):
            return CNF(from_clauses=[[]])
        # Check hamming distance
        one_two_hamming_distance = sum(
            map(
                lambda i: int(instance_one.value[i] != instance_two.value[i]),
                (i for i in range(context.DIM))
            )
        )
        one_three_hamming_distance = sum(
            map(
                lambda i: int(
                    instance_one.value[i] != instance_three.value[i]
                ),
                (i for i in range(context.DIM))
            )
        )
        if one_two_hamming_distance > one_three_hamming_distance:
            return CNF(from_clauses=[[]])
        return CNF()

    @staticmethod
    def _simplify_constant_constant_var(
        constant_one: Constant,
        constant_two: Constant,
        context: Context,
        instance: 'LEH'
    ) -> Component:
        if not (
            LEH._constant_is_full(constant_one) and
            LEH._constant_is_full(constant_two)
        ):
            return FALSE()
        one_two_hamming_distance = sum(
            map(
                lambda i: int(
                    constant_one.value[i] != constant_two.value[i]
                ),
                (i for i in range(context.DIM))
            )
        )
        if one_two_hamming_distance == 0:
            return TRUE()
        return instance

    @staticmethod
    def _simplify_constant_var_constant(
        constant_one: Constant,
        constant_three: Constant,
        context: Context,
        instance: 'LEH'
    ) -> Component:
        if not (
            LEH._constant_is_full(constant_one) and
            LEH._constant_is_full(constant_three)
        ):
            return FALSE()
        one_three_hamming_distance = sum(
            map(
                lambda i: int(
                    constant_one.value[i] != constant_three.value[i]
                ),
                (i for i in range(context.DIM))
            )
        )
        if one_three_hamming_distance == context.DIM:
            return TRUE()
        return instance

    @staticmethod
    def _simplify_var_constant_constant(
        constant_two: Constant,
        constant_three: Constant,
        context: Context,
        instance: 'LEH'
    ) -> Component:
        if not (
            LEH._constant_is_full(constant_two) and
            LEH._constant_is_full(constant_three)
        ):
            return FALSE()
        return instance

    @staticmethod
    def _simplify_single_constant(
        constant: Constant,
        instance: 'LEH'
    ) -> Component:
        if not LEH._constant_is_full(constant):
            return FALSE()
        return instance

    @staticmethod
    def _simplify_triple_constant(
        constant_one: Constant,
        constant_two: Constant,
        constant_three: Constant,
        context: Context,
        instance: 'LEH'
    ) -> Component:
        # Check for fullness in constants
        if not (
            LEH._constant_is_full(constant_one) and
            LEH._constant_is_full(constant_two) and
            LEH._constant_is_full(constant_three)
        ):
            return FALSE()
        # Check hamming distance
        one_two_hamming_distance = sum(
            map(
                lambda i: int(constant_one.value[i] != constant_two.value[i]),
                (i for i in range(context.DIM))
            )
        )
        one_three_hamming_distance = sum(
            map(
                lambda i: int(
                    constant_one.value[i] != constant_three.value[i]
                ),
                (i for i in range(context.DIM))
            )
        )
        if one_two_hamming_distance > one_three_hamming_distance:
            return FALSE()
        return TRUE()

    def simplify(self):
        if (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Var)
        ):
            return LEH._simplify_single_constant(
                self.instance_one,
                self
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Var)
        ):
            return LEH._simplify_constant_constant_var(
                self.instance_one,
                self.instance_two,
                self.context,
                self
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Constant)
        ):
            return LEH._simplify_constant_constant_var(
                self.instance_one,
                self.instance_three,
                self.context,
                self
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Constant)
        ):
            return LEH._simplify_single_constant(
                self.instance_three,
                self
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Constant)
        ):
            return LEH._simplify_var_constant_constant(
                self.instance_two,
                self.instance_three,
                self.context,
                self
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Var)
        ):
            return LEH._simplify_single_constant(
                self.instance_two,
                self
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Constant)
        ):
            return LEH._simplify_triple_constant(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context,
                self
            )
        return self

    def encode(self) -> CNF:
        cnf = CNF()
        if (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Var)
        ):
            cnf = self._triple_var_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Constant)
        ):
            cnf = self._var_var_constant_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Var)
        ):
            cnf = self._var_constant_var_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Var)
        ):
            cnf = self._constant_var_var_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Var)
        ):
            cnf = self._constant_constant_var_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Var) and
            isinstance(self.instance_three, Constant)
        ):
            cnf = self._constant_var_constant_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Var) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Constant)
        ):
            cnf = self._var_constant_constant_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        elif (
            isinstance(self.instance_one, Constant) and
            isinstance(self.instance_two, Constant) and
            isinstance(self.instance_three, Constant)
        ):
            cnf = self._triple_constant_encode(
                self.instance_one,
                self.instance_two,
                self.instance_three,
                self.context
            )
        return cnf
