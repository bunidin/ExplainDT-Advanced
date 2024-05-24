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
    #           leaf_2: 0,0,1 (True)
    #       node_4: 0,1,_
    #           leaf_3: 0,1,0 (False)
    #           leaf_4: 0,1,1 (True)
    #   node_2: 1,_,_
    #       node_5: 1,_,0
    #           leaf_5: 1,0,0 (False)
    #           leaf_6: 1,1,0 (True)
    #       node_6: 1,_,1
    #           leaf_7: 1,0,1 (True)
    #           leaf_8: 1,1,1 (True)

    leaf_1 = Leaf(id=7, truth_value=False)
    leaf_2 = Leaf(id=8, truth_value=True)
    leaf_3 = Leaf(id=9, truth_value=False)
    leaf_4 = Leaf(id=10, truth_value=True)
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


def not_minSR(x: Constant, y: Constant):
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


def run_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    x: Constant,
    y: Constant
) -> int:
    context = Context(3, tree)
    formula = not_minSR(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    print(cnf.meaning_clauses + cnf.consistency_clauses)
    return run_solver(solver_path, file_name).returncode


def run_test_simplified(
    tree: Tree,
    file_name: str,
    solver_path: str,
    x: Constant,
    y: Constant
) -> int:
    context = Context(3, tree)
    formula = not_minSR(x, y)
    # cnf = generate_cnf(formula, context)
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(file_name)
    print(cnf.meaning_clauses + cnf.consistency_clauses)
    return run_solver(solver_path, file_name).returncode


def not_minimum_SR(x: Constant, y: Constant):
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


def run_test_minimum(
    tree: Tree,
    file_name: str,
    solver_path: str,
    x: Constant,
    y: Constant
) -> int:
    context = Context(3, tree)
    formula = not_minimum_SR(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def test_is_minimal_sr_1(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, x, y) == 20


def test_not_is_minimum_sr_1(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_test_minimum(tree, file_name, solver_path, x, y) == 10


def test_is_minimal_sr_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, x, y) == 20


def test_is_minimum_sr_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_minimum(tree, file_name, solver_path, x, y) == 20


def test_not_is_minimal_sr_1(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, x, y) == 10


def test_not_is_minimal_sr_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, x, y) == 10
