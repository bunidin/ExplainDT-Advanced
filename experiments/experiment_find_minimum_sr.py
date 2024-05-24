from experiments.model import Experiment
from experiments.utilities import plot_query_times, generate_trees
from components.instances import Constant, Var
from components.operators import Not, And, Or, Exists
from components.predicates import (
    LEL,
    AllNeg,
    Full,
    AllPoss,
    Subsumption
)
from context.model import Tree
from context import Context, Symbol, contextualize
from solvers import run_solver
from time import time
from random import choices


FILE_NAME = 'test.cnf'


def find_minSR(x: Constant, context: Context):
    y = Var('y')
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
    minSR_x_y = And(
        SR_x_y,
        Not(
            Exists(
                z,
                And(
                    SR_x_z,
                    And(
                        LEL(z, y),
                        Not(LEL(y, z))
                    )
                )
            )
        )
    )
    formula = Exists(y, minSR_x_y)
    contextualize(formula, context)
    formula.encode().to_file(FILE_NAME)
    output = run_solver('./solvers/solver_execs/yalsat', FILE_NAME).returncode
    return output


def run_experiment(bots_allowed: bool, graph_name: str, output_name: str):
    MIN_DIM = 10
    MAX_DIM = 40
    sizes, times, results = [], [], []
    formula_string = graph_name
    symbols = [Symbol.ONE, Symbol.ZERO]
    if bots_allowed:
        symbols.append(Symbol.BOT)
    for dim in range(MIN_DIM, MAX_DIM + 1):
        dim_start = time()
        for _ in range(5):
            tree = Tree(
                from_file="dtrees/generated_trees/d" + str(dim) + ".json",
                iterative=True
            )
            context = Context(dim, tree)
            x = Constant(tuple(choices(symbols, k=dim)))
            start = time()
            result = find_minSR(x, context)
            end = time()
            results.append(result)
            sizes.append(tree.number_of_nodes())
            times.append(round(end - start, 6))
        dim_end = time()
        dim_time = dim_end - dim_start
        print(f'Dimension: {dim} finished - Time: {dim_time}')

    plot_query_times(
        f"experiments/experiment_results/{output_name}.png",
        formula_string,
        sizes,
        times,
        results
    )


def bot_allowed():
    run_experiment(
        True,
        'Find MinimumSR (random partial instances)',
        'find_minimum_sr_bot_allowed'
    )


def not_bot_allowed():
    run_experiment(
        False,
        'mSR (random full instances)',
        'find_minimum_sr_no_bot_allowed'
    )


experiment = Experiment(
    label='find_minimumSR',
    variations={
        'bot_allowed': bot_allowed,
        'no_bot_allowed': not_bot_allowed
    },
    tree_generator=generate_trees
)
