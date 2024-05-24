from experiments.model import Experiment
from experiments.utilities import plot_query_times, generate_trees
from components.instances import Constant, GuardedVar, Var
from components.operators import Not, And, Or, Exists
from components.predicates import (
    AllNeg,
    ForAllGuarded,
    AllPoss,
    IsNode,
    Subsumption,
    Cons
)
from context.model import Tree
from context import Context, Symbol, contextualize
from solvers import run_solver
from time import time
from random import choices


FILE_NAME = 'test.cnf'


def RFS(x: Constant, context: Context):
    y = Var('')
    y_1 = Var('y_1')
    z_1 = Var('z_1')
    y_2 = GuardedVar('y_2')
    z_2 = GuardedVar('z_2')
    not_sem_x = Exists(
        y_1,
        Exists(
            z_1,
            And(
                IsNode(y_1),
                And(
                    IsNode(z_1),
                    And(
                        AllNeg(y_1),
                        And(
                            AllNeg(z_1),
                            And(
                                Cons(x, y_1),
                                Cons(x, z_1)
                            )
                        )
                    )
                )
            )
        )
    )
    sem_y = ForAllGuarded(
        y_2,
        ForAllGuarded(
            z_2,
            Or(
                Not(AllPoss(y_2)),
                Or(
                    Not(AllNeg(z_2)),
                    Or(
                        Not(Cons(x, y_2)),
                        Not(Cons(x, z_2))
                    )
                )
            )
        )
    )
    formula = Exists(
        y,
        Or(
            not_sem_x,
            And(
                sem_y,
                And(
                    Subsumption(y, x),
                    Not(Subsumption(x, y))
                )
            )
        )
    )
    contextualize(formula, context)
    formula.encode().to_file(FILE_NAME)
    output = run_solver('./solvers/solver_execs/kissat', FILE_NAME).returncode
    return output


def run_experiment():
    MIN_DIM = 10
    MAX_DIM = 13
    graph_name = 'RFS'
    output_name = 'RFS (bots allowed)'
    # generate_trees(MIN_DIM, MAX_DIM)
    sizes, times, results = [], [], []
    formula_string = graph_name
    symbols = [Symbol.ONE, Symbol.ZERO, Symbol.BOT]
    for dim in range(MIN_DIM, MAX_DIM + 1):
        print(f'Dimension: {dim}')
        dim_start = time()
        for i in range(5):
            tree = Tree(
                from_file="dtrees/generated_trees/d" + str(dim) + ".json",
                iterative=True
            )
            context = Context(dim, tree)
            x = Constant(tuple(choices(symbols, k=dim)))
            start = time()
            result = RFS(x, context)
            end = time()
            print(f'Input {i}. finished in {end - start}.')
            results.append(result)
            sizes.append(tree.number_of_nodes())
            times.append(round(end - start, 6))
        dim_end = time()
        print(f'Finished in {dim_end - dim_start}.')

    plot_query_times(
        f"experiments/experiment_results/{output_name}.png",
        formula_string,
        sizes,
        times,
        results
    )


def run_heavy_dim_simple_test():
    graph_name = 'minimumRFS - (80 feat)'
    output_name = 'minimumRFS80'
    sizes, times, results = [], [], []
    symbols = [Symbol.ONE, Symbol.ZERO, Symbol.BOT]
    for i in range(5):
        tree = Tree(
            from_file="dtrees/generated_trees/dim_heavy_tree.json",
            iterative=True
        )
        context = Context(80, tree)
        x = Constant(tuple(choices(symbols, k=80)))
        start = time()
        result = RFS(x, context)
        end = time()
        print(f'Input {i}. finished in {end - start}.')
        results.append(result)
        sizes.append(tree.number_of_nodes())
        times.append(round(end - start, 6))
    plot_query_times(
        f"experiments/experiment_results/{output_name}.png",
        graph_name,
        sizes,
        times,
        results
    )


experiment = Experiment(
    label='minimum_sem',
    variations={
        'default': run_experiment,
        'dim_heavy_single': run_heavy_dim_simple_test
    },
    tree_generator=generate_trees
)
