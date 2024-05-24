import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf, generate_simplified_cnf
from components.operators import Exists, And, Not
from components.predicates import LEH, Subsumption
from components.instances import Var, Constant
from solvers import extract_meaning, run_solver


# For every version of Subsumption:
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


def gen_constant_constant_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_3),
                Subsumption(constant_3, x)
            ),
            LEH(constant_1, constant_2, x)
        )
    )
    return formula


def gen_not_constant_constant_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_3),
                Subsumption(constant_3, x)
            ),
            Not(LEH(constant_1, constant_2, x))
        )
    )
    return formula


def gen_constant_var_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_2),
                Subsumption(constant_2, x)
            ),
            LEH(constant_1, x, constant_3)
        )
    )
    return formula


def gen_not_constant_var_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_2),
                Subsumption(constant_2, x)
            ),
            Not(LEH(constant_1, x, constant_3))
        )
    )
    return formula


def gen_var_constant_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_1),
                Subsumption(constant_1, x)
            ),
            LEH(x, constant_2, constant_3)
        )
    )
    return formula


def gen_not_var_constant_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_1),
                Subsumption(constant_1, x)
            ),
            Not(LEH(x, constant_2, constant_3))
        )
    )
    return formula


def gen_constant_var_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_2),
                        Subsumption(constant_2, x)
                    ),
                    And(
                        Subsumption(y, constant_3),
                        Subsumption(constant_3, y)
                    ),
                ),
                LEH(constant_1, x, y)
            )
        )
    )
    return formula


def gen_not_constant_var_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_2),
                        Subsumption(constant_2, x)
                    ),
                    And(
                        Subsumption(y, constant_3),
                        Subsumption(constant_3, y)
                    ),
                ),
                Not(LEH(constant_1, x, y))
            )
        )
    )
    return formula


def gen_constant_constant_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    return LEH(constant_1, constant_2, constant_3)


def gen_not_constant_constant_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    return Not(LEH(constant_1, constant_2, constant_3))


def gen_var_var_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_1),
                        Subsumption(constant_1, x)
                    ),
                    And(
                        Subsumption(y, constant_2),
                        Subsumption(constant_2, y)
                    ),
                ),
                LEH(x, y, constant_3)
            )
        )
    )
    return formula


def gen_not_var_var_constant_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_1),
                        Subsumption(constant_1, x)
                    ),
                    And(
                        Subsumption(y, constant_2),
                        Subsumption(constant_2, y)
                    ),
                ),
                Not(LEH(x, y, constant_3))
            )
        )
    )
    return formula


def gen_var_constant_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_1),
                        Subsumption(constant_1, x)
                    ),
                    And(
                        Subsumption(y, constant_3),
                        Subsumption(constant_3, y)
                    ),
                ),
                LEH(x, constant_2, y)
            )
        )
    )
    return formula


def gen_not_var_constant_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    formula = Exists(
        x,
        Exists(
            y,
            And(
                And(
                    And(
                        Subsumption(x, constant_1),
                        Subsumption(constant_1, x)
                    ),
                    And(
                        Subsumption(y, constant_3),
                        Subsumption(constant_3, y)
                    ),
                ),
                Not(LEH(x, constant_2, y))
            )
        )
    )
    return formula


def gen_var_var_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    z = Var('z')
    formula = Exists(
        x,
        Exists(
            y,
            Exists(
                z,
                And(
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
                            And(
                                Subsumption(z, constant_3),
                                Subsumption(constant_3, z)
                            ),
                        ),
                    ),
                    LEH(x, y, z)
                )
            )
        )
    )
    return formula


def gen_not_var_var_var_cnf(
    constant_1: Constant,
    constant_2: Constant,
    constant_3: Constant
):
    x = Var('x')
    y = Var('y')
    z = Var('z')
    formula = Exists(
        x,
        Exists(
            y,
            Exists(
                z,
                And(
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
                            And(
                                Subsumption(z, constant_3),
                                Subsumption(constant_3, z)
                            ),
                        ),
                    ),
                    Not(LEH(x, y, z))
                )
            )
        )
    )
    return formula


def test_leh_ccv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_ccv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_ccv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_ccv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_ccv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_ccv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_ccv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_ccv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_ccv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_cvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_cvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_cvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_cvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_cvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_cvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_cvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_cvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_ccc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_ccc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_ccc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_ccc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_ccc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_ccc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_ccc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_ccc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_ccc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_ccc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_ccc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_ccc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_constant_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_vcc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_vcc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vcc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vcc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_vcc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_vcc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vcc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vcc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_cvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_cvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_cvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_cvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_cvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_cvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_cvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_cvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_cvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_cvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_cvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_cvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_var_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_vvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_vvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_vvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_vvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvc_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvc_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvc_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvc_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvc_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vvc_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vvc_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_var_constant_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vcv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_vcv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_vcv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vcv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vcv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vcv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vcv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_vcv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_vcv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vcv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vcv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vcv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_leh_vvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_leh_vvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_leh_vvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_not_leh_vvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_not_leh_vvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_lesser_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_leh_vvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    assert result.returncode == 20


def test_s_leh_vvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_leh_vvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvv_full_full_full_lesser(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvv_full_full_full_equal(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvv_full_full_full_equal_2(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvv_full_full_full_equal_3(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 20


def test_s_not_leh_vvv_full_full_full_greater(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    z = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vvv_not_full_full_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10


def test_s_not_leh_vvv_full_full_not_full(tree, file_name, solver_path):
    context = Context(3, tree)
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    z = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    formula = gen_not_var_constant_var_cnf(x, y, z)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    assert result.returncode == 10
