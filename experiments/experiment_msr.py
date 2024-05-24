from experiments.model import Experiment
from experiments.utilities import plot_query_times, generate_trees
from components.instances import Constant
from components.operators import Not, And, Or
from components.predicates import AllNeg, Full, AllPoss, Subsumption
from context.model import Tree
from context import Context, Symbol, contextualize
from solvers import run_solver
from time import time
from random import choices


def SR(y_values: tuple, x: Constant, context: Context) -> bool:
    y = Constant(y_values)
    formula = And(
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
    file_name = 'test.cnf'
    contextualize(formula, context)
    formula.encode().to_file(file_name)
    output = run_solver('./solvers/solver_execs/yalsat', file_name).returncode
    return output == 10


def mSR(x: Constant, dim: int, tree: Tree):
    context = Context(dim, tree)
    y_values = x.value
    progress = True
    while progress:
        progress = False
        for i in range(dim):
            if y_values[i] == Symbol.BOT:
                return None
            z_values = list(attr for attr in y_values)
            z_values[i] = Symbol.BOT
            # Reset context on each run
            context = Context(dim, tree)
            if SR(tuple(z_values), x, context):
                progress = False
                y_values = z_values
                break
    return y_values


def run_experiment(bots_allowed: bool, graph_name: str, output_name: str):
    MIN_DIM = 10
    MAX_DIM = 40
    # generate_trees(MIN_DIM, MAX_DIM)
    sizes, times, results = [], [], []
    formula_string = graph_name
    symbols = [Symbol.ONE, Symbol.ZERO]
    if bots_allowed:
        symbols.append(Symbol.BOT)
    for dim in range(MIN_DIM, MAX_DIM + 1):
        print(f'Dimension: {dim}')
        for _ in range(5):
            tree = Tree(
                from_file="dtrees/generated_trees/d" + str(dim) + ".json",
                iterative=True
            )
            x = Constant(tuple(choices(symbols, k=dim)))
            start = time()
            mSR(x, dim, tree)
            end = time()
            results.append(10)
            sizes.append(tree.number_of_nodes())
            times.append(round(end - start, 6))

    plot_query_times(
        f"experiments/experiment_results/{output_name}.png",
        formula_string,
        sizes,
        times,
        results
    )


def bot_allowed():
    run_experiment(True, 'mSR (random instances)', 'msr_bot_allowed')


def not_bot_allowed():
    run_experiment(
        False,
        'mSR (no bots, random instances)',
        'msr_no_bot_allowed'
    )


experiment = Experiment(
    label='mSR',
    variations={
        'bot_allowed': bot_allowed,
        'no_bot_allowed': not_bot_allowed
    },
    tree_generator=generate_trees
)
