from random import sample
from components.instances import Constant, Var
from components.operators import And, Component, Not, Or, Exists
from components.predicates import (
    LEL,
    RFS,
    LEH,
    AllNeg,
    Full,
    Subsumption,
    AllPoss,
)
from context import Symbol


def SR(x: Constant, y: Constant) -> Component:
    return And(
        Full(x),
        And(
            Subsumption(y, x),
            And(
                Or(
                    Not(AllPoss(x)),
                    AllPoss(y)
                ),
                Or(
                    Not(AllNeg(x)),
                    AllNeg(y)
                )
            )
        )
    )


def minimal_SR(x: Constant, y: Constant) -> Component:
    z = Var('z')
    SR_x_y = And(
        Full(x),
        And(
            Subsumption(y, x),
            And(
                Or(
                    Not(AllPoss(x)),
                    AllPoss(y)
                ),
                Or(
                    Not(AllNeg(x)),
                    AllNeg(y)
                )
            )
        )
    )
    SR_x_z = And(
        Full(x),
        And(
            Subsumption(z, x),
            And(
                Or(
                    Not(AllPoss(x)),
                    AllPoss(z)
                ),
                Or(
                    Not(AllNeg(x)),
                    AllNeg(z)
                )
            )
        )
    )
    return Exists(
        z,
        Or(
            Not(SR_x_y),
            And(
                SR_x_z,
                And(
                    Subsumption(z, y),
                    Not(Subsumption(y, z))
                )
            )
        )
    )


def minimum_SR(x: Constant, y: Constant) -> Component:
    z = Var('z')
    SR_x_y = And(
            Full(x),
            And(
                Subsumption(y, x),
                And(
                    Or(
                        Not(AllPoss(x)),
                        AllPoss(y)
                    ),
                    Or(
                        Not(AllNeg(x)),
                        AllNeg(y)
                    )
                )
            )
        )
    SR_x_z = And(
        Full(x),
        And(
            Subsumption(z, x),
            And(
                Or(
                    Not(AllPoss(x)),
                    AllPoss(z)
                ),
                Or(
                    Not(AllNeg(x)),
                    AllNeg(z)
                )
            )
        )
    )
    return Exists(
        z,
        Or(
            Not(SR_x_y),
            And(
                SR_x_z,
                And(
                    LEL(z, y),
                    Not(LEL(y, z))
                )
            )
        )
    )


def minimal_RFS(x: Constant) -> Component:
    y = Var('y')
    return Exists(
        y,
        Or(
            Not(RFS(x)),
            And(
                RFS(y),
                And(
                    Subsumption(y, x),
                    Not(Subsumption(x, y))
                )
            )
        )
    )


def minimum_CR(x: Constant, y: Constant) -> Component:
    z = Var('z')
    PHI_x_y = And(
        Or(
            Not(AllPoss(x)),
            AllPoss(y)
        ),
        Or(
            Not(AllPoss(y)),
            AllPoss(x)
        )
    )
    PHI_x_z = And(
        Or(
            Not(AllPoss(x)),
            AllPoss(z)
        ),
        Or(
            Not(AllPoss(z)),
            AllPoss(x)
        )
    )
    RHO = Or(
        Not(Full(z)),
        Or(
            PHI_x_z,
            LEH(x, y, z)
        )
    )
    return Exists(
        z,
        Not(
            And(
                Full(x),
                And(
                    Full(y),
                    And(
                        Not(PHI_x_y),
                        RHO
                    )
                )
            )
        )
    )


def flip_k_values(values: tuple[Symbol], k: int) -> Constant:
    indexes_to_flip = sample(range(len(values)), k)
    new_values = [symbol for symbol in values]
    for i in indexes_to_flip:
        new_values[i] = Symbol.BOT
    return Constant(tuple(new_values))


def kb_SR(x: Constant, k: int) -> Component:
    z = flip_k_values(x.value, k)
    y = Var('y')
    SR_x_y = And(
        Full(x),
        And(
            Subsumption(y, x),
            And(
                Or(
                    Not(AllPoss(x)),
                    AllPoss(y)
                ),
                Or(
                    Not(AllNeg(x)),
                    AllNeg(y)
                )
            )
        )
    )
    return Exists(
        y,
        And(
            SR_x_y,
            LEL(y, z)
        )
    )


def kb_RFS(dimension: int, k: int) -> Component:
    y = Var('y')
    z = flip_k_values(tuple(Symbol.ONE for _ in range(dimension)), k)
    return Exists(
        y,
        And(
            RFS(y),
            LEL(y, z)
        )
    )
