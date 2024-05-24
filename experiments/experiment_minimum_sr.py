from experiments.model import Experiment
from experiments.utilities import plot_query_times, generate_trees
from components.instances import Constant, Var
from components.operators import Not, And, Or, Exists
from components.predicates import (
    LEL, AllNeg, Full, AllPoss, Subsumption
)
from context.model import Tree
from context import Context, Symbol, contextualize
from solvers import run_solver
from time import time
from random import choices


FILE_NAME = 'test.cnf'


def not_minSR(x: Constant, y: Constant, context: Context):
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
    # formula = And(
    #     SR_x_y,
    #     Not(
    #         Exists(
    #             z,
    #             And(
    #                 SR_x_z,
    #                 And(
    #                     LEL(z, y),
    #                     Not(LEL(y, z))
    #                 )
    #             )
    #         )
    #     )
    # )
    formula = Exists(
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
    contextualize(formula, context)
    formula.encode().to_file(FILE_NAME)
    output = run_solver('./solvers/solver_execs/yalsat', FILE_NAME).returncode
    return output


def run_experiment():
    MIN_DIM = 10
    MAX_DIM = 40
    graph_name = 'not(MinSR(x, y)) (x: full, y: partial)'
    output_name = 'not_minimum_sr'
    # generate_trees(MIN_DIM, MAX_DIM)
    sizes, times, results = [], [], []
    formula_string = graph_name
    symbols = [Symbol.ONE, Symbol.ZERO]
    symbols_with_bot = [Symbol.ONE, Symbol.ZERO, Symbol.BOT]
    for dim in range(MIN_DIM, MAX_DIM + 1):
        dim_start = time()
        for _ in range(5):
            tree = Tree(
                from_file="dtrees/generated_trees/d" + str(dim) + ".json",
                iterative=True
            )
            context = Context(dim, tree)
            x = Constant(tuple(choices(symbols, k=dim)))
            y = Constant(tuple(choices(symbols_with_bot, k=dim)))
            start = time()
            result = not_minSR(x, y, context)
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


experiment = Experiment(
    label='not_minimumSR',
    variations={
        'default': run_experiment,
    },
    tree_generator=generate_trees
)
