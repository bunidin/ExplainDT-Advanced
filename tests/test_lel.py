import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf, generate_simplified_cnf
from components.operators import Exists, And, Not
from components.predicates import LEL, Subsumption
from components.instances import Var, Constant
from solvers import extract_meaning, run_solver


# For every version of LEL:
# - var/var
# - constant/var
# - var/constant
# - constant/constant
# we create 3 tests:
# - lesser level
# - equal level
# - greater level
# with 4 possible variants each:
# - x with bots / y with bots
# - x with bots / y withouth bots
# - x without bots / y with bots
# - x withouth / y withouth bots

# In total we should have 48 tests (we will have less becouse some comination
# are not possible).


def teardown_module(module):
    file_name = 'test_file.cnf'
    if path.isfile(file_name):
        remove(file_name)


@pytest.fixture(scope='module')
def file_name():
    return 'test_file.cnf'


@pytest.fixture(scope='module')
def solver_path():
    return path.join('solvers', 'solver_execs', 'kissat')


@pytest.fixture(scope='module')
def tree():
    node_2 = Leaf(id=3, truth_value=True)
    node_1 = Leaf(id=2, truth_value=True)
    root = FeatNode(id=1, label=0, child_zero=node_1, child_one=node_2)
    return Tree(root)


def gen_double_var_cnf(constant_1: Constant, constant_2: Constant):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    Subsumption(x, constant_1),
                    Subsumption(constant_1, x)
                ),
                And(
                    And(
                        Subsumption(y, constant_2),
                        Subsumption(constant_2, y)
                    ),
                    LEL(x, y)
                )
            )
        )
    )
    return formula


def gen_not_double_var_cnf(constant_1: Constant, constant_2: Constant):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    Subsumption(x, constant_1),
                    Subsumption(constant_1, x)
                ),
                And(
                    And(
                        Subsumption(y, constant_2),
                        Subsumption(constant_2, y)
                    ),
                    Not(LEL(x, y))
                )
            )
        )
    )
    return formula


def gen_var_constant_cnf(constant_1: Constant, constant_2: Constant):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_1),
                Subsumption(constant_1, x)
            ),
            LEL(x, constant_2)
        )
    )
    return formula


def gen_not_var_constant_cnf(constant_1: Constant, constant_2: Constant):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_1),
                Subsumption(constant_1, x)
            ),
            Not(LEL(x, constant_2))
        )
    )
    return formula


def gen_constant_var_cnf(constant_1: Constant, constant_2: Constant):
    y = Var('y')
    formula = Exists(
        y,
        And(
            And(
                Subsumption(y, constant_2),
                Subsumption(constant_2, y)
            ),
            LEL(constant_1, y)
        )
    )
    return formula


def gen_not_constant_var_cnf(constant_1: Constant, constant_2: Constant):
    y = Var('y')
    formula = Exists(
        y,
        And(
            And(
                Subsumption(y, constant_2),
                Subsumption(constant_2, y)
            ),
            Not(LEL(constant_1, y))
        )
    )
    return formula


def gen_double_constant_cnf(constant_1: Constant, constant_2: Constant):
    return LEL(constant_1, constant_2)


def gen_not_double_constant_cnf(
    constant_1: Constant,
    constant_2: Constant
):
    return Not(LEL(constant_1, constant_2))


def test_lel_double_var_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_var_lesser_level_no_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_var_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_var_greater_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_lel_double_var_greater_level_bot_no_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_var_constant_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_var_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_var_constant_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_var_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_var_constant_greater_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_var_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_constant_var_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_constant_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_constant_var_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_constant_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_constant_var_greater_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_constant_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_double_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_constant_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_lel_double_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_lel_double_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_var_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_var_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_var_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_double_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_var_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_var_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_var_constant_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_var_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_var_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_var_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_constant_var_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_constant_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_constant_var_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_constant_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_constant_var_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_constant_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_double_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_constant_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_lel_double_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_lel_double_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_var_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_var_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_var_greater_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_lel_double_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_var_constant_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_var_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_var_constant_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_var_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_var_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_var_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_constant_var_lesser_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_constant_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_constant_var_equal_level_bot_bot(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_constant_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_constant_var_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_constant_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_double_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_constant_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_lel_double_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_lel_double_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_var_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_var_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_var_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_double_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_var_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_var_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_var_constant_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_var_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_var_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_var_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_var_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_constant_var_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_constant_var_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_constant_var_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_constant_var_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_constant_var_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_constant_var_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_constant_var_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_double_constant_lesser_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_constant_lesser_level_no_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_constant_equal_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_constant_equal_level_no_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_lel_double_constant_greater_level_bot_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_lel_double_constant_greater_level_bot_no_bot(
    tree,
    file_name,
    solver_path
):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    formula = gen_not_double_constant_cnf(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10
