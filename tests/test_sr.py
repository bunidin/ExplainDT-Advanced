import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf, generate_simplified_cnf
from components.operators import And, Exists, Or, Not
from components.predicates import AllPoss, AllNeg, Subsumption, Full, LEL
from components.instances import Var, Constant
from solvers import run_solver


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
    # In tree:
    # root: _,_,_
    #   node_1: 0,_,_
    #       node_3: 0,0,_
    #           leaf_1: 0,0,0 (False)
    #           leaf_2: 0,0,1 (False)
    #       node_4: 0,1,_
    #           leaf_3: 0,1,0 (False)
    #           leaf_4: 0,1,1 (False)
    #   node_2: 1,_,_
    #       node_5: 1,_,0
    #           leaf_5: 1,0,0 (False)
    #           leaf_6: 1,1,0 (True)
    #       node_6: 1,_,1
    #           leaf_7: 1,0,1 (True)
    #           leaf_8: 1,1,1 (True)

    # MCR(leaf_1) => leaf_7, leaf_6, leaf_4

    leaf_1 = Leaf(id=7, truth_value=False)
    leaf_2 = Leaf(id=8, truth_value=False)
    leaf_3 = Leaf(id=9, truth_value=False)
    leaf_4 = Leaf(id=10, truth_value=False)
    leaf_5 = Leaf(id=11, truth_value=False)
    leaf_6 = Leaf(id=12, truth_value=True)
    leaf_7 = Leaf(id=13, truth_value=True)
    leaf_8 = Leaf(id=14, truth_value=True)
    node_6 = FeatNode(id=6, label=1, child_zero=leaf_7, child_one=leaf_8)
    node_5 = FeatNode(id=5, label=1, child_zero=leaf_5, child_one=leaf_6)
    node_4 = FeatNode(id=4, label=2, child_zero=leaf_3, child_one=leaf_4)
    node_3 = FeatNode(id=3, label=2, child_zero=leaf_1, child_one=leaf_2)
    node_2 = FeatNode(id=2, label=2, child_zero=node_5, child_one=node_6)
    node_1 = FeatNode(id=1, label=1, child_zero=node_3, child_one=node_4)
    root = FeatNode(id=0, label=0, child_zero=node_1, child_one=node_2)
    return Tree(root)


def run_double_var_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant_1: Constant,
    constant_2: Constant,
    run_simplified=False
) -> int:
    context = Context(3, tree)
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
                    And(
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
                )
            )
        )
    )
    if run_simplified:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_constant_var_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant_1: Constant,
    constant_2: Constant,
    run_simplified=False
) -> int:
    context = Context(3, tree)
    x = constant_1
    y = Var('y')
    formula = Exists(
        y,
        And(
            And(
                Subsumption(y, constant_2),
                Subsumption(constant_2, y)
            ),
            And(
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
        )
    )
    if run_simplified:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_var_constant_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant_1: Constant,
    constant_2: Constant,
    run_simplified=False
) -> int:
    context = Context(3, tree)
    x = Var('x')
    y = constant_2
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant_1),
                Subsumption(constant_1, x)
            ),
            And(
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
        )
    )
    if run_simplified:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_double_constant_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant_1: Constant,
    constant_2: Constant,
    run_simplified=False
) -> int:
    context = Context(3, tree)
    x = constant_1
    y = constant_2
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
    if run_simplified:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def test_exists(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Var('y')
    formula = Exists(
        y,
        And(
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
    )
    context = Context(3, tree)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    assert run_solver(
        solver_path,
        file_name
    ).returncode == 10


def test_not_exists(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    y = Var('y')
    formula = Exists(
        y,
        And(
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
    )
    context = Context(3, tree)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    assert run_solver(
        solver_path,
        file_name
    ).returncode == 20


def test_not_exists_with_lel_restrict(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    y = Var('y')
    formula = Exists(
        y,
        And(
            And(
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
            ),
            LEL(y, Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT)))
        )
    )
    context = Context(3, tree)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    assert run_solver(
        solver_path,
        file_name
    ).returncode == 20


def test_is_double_var_sr_1_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 10


def test_is_double_var_sr_2_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_var_sr_3_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 10


def test_is_double_var_sr_4_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_var_sr_5_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_var_sr_6_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_double_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_var_constant_sr_1_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 10


def test_is_var_constant_sr_2_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_var_constant_sr_3_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 10


def test_is_var_constant_sr_4_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_var_constant_sr_5_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_var_constant_sr_6_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_var_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_constant_var_sr_1_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 10


def test_is_constant_var_sr_2_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_constant_var_sr_3_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 10


def test_is_constant_var_sr_4_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_constant_var_sr_5_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_constant_var_sr_6_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_constant_var_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_constant_sr_1_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 10


def test_is_double_constant_sr_2_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_constant_sr_3_true(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 10


def test_is_double_constant_sr_4_false(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_constant_sr_5_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 20


def test_is_double_constant_sr_6_false(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_double_constant_test(tree, file_name, solver_path, x, y) == 20
