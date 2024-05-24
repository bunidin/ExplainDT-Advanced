from components.base import generate_cnf
from experiments.model import Experiment
from experiments.utilities import plot_query_times, generate_trees
from components.instances import Constant, Var
from components.operators import And, Exists
from components.predicates import Subsumption, RFS
from context.model import Tree
from context import Context, Symbol
from solvers import run_solver
from time import time
from random import choices


FILE_NAME = 'test.cnf'


def fRFS(x: Constant, context: Context):
    # y = Var('y')
    # z = Var('z')
    # formula = Exists(
    #     y,
    #     Exists(
    #         z,
    #         And(
    #             IsNode(y),
    #             And(
    #                 IsNode(z),
    #                 And(
    #                     AllPoss(y),
    #                     And(
    #                         AllNeg(z),
    #                         And(
    #                             Cons(x, y),
    #                             Cons(x, z)
    #                         )
    #                     )
    #                 )
    #             )
    #         )
    #     )
    # )
    y = Var('y')
    formula = Exists(
        y,
        And(
            And(
                Subsumption(x, y),
                Subsumption(y, x)
            ),
            RFS(y)
        )
    )
    # contextualize(formula, context)
    # formula.encode().to_file(FILE_NAME)
    fstart = time()
    cnf = generate_cnf(formula, context)
    fend = time()
    cnf.to_file(FILE_NAME)
    estart = time()
    output = run_solver('./solvers/solver_execs/kissat', FILE_NAME).returncode
    eend = time()
    print(
        f'Formula constructed in {fend - fstart}.\
            Evaluation finished in {eend - estart}'
    )
    return output


def run_experiment():
    MIN_DIM = 10
    MAX_DIM = 40
    graph_name = 'not(RFS) - (Random pInstances)'
    output_name = 'not_RFS_bots_allowed'
    # generate_trees(MIN_DIM, MAX_DIM)
    sizes, times, results = [], [], []
    formula_string = graph_name
    symbols = [Symbol.ONE, Symbol.ZERO, Symbol.BOT]
    # symbols = [Symbol.ONE, Symbol.ZERO]
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
            result = fRFS(x, context)
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
    label='sem',
    variations={
        'default': run_experiment,
    },
    tree_generator=generate_trees
)
