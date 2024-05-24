import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf
from components.operators import And, Exists, Not
from components.predicates import IsNode, Subsumption
from components.instances import Var, Constant
from solvers import run_solver, extract_meaning


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
    #           leaf_1: 0,0,0
    #           leaf_2: 0,0,1
    #       leaf_3: 0,1,_
    #   node_2: 1,_,_
    #       leaf_4: 1,_,0
    #       node_4: 1,_,1
    #           leaf_5: 1,0,1
    #           leaf_6: 1,1,1

    # Not in tree:
    # not_node_1: 0,1,0
    # not_node_2: 0,1,1
    # not_node_3: 1,0,0
    # not_node_4: 1,1,0
    # not_node_5: _,_,0
    # not_node_6: _,_,1
    # not_node_7: _,0,0
    # not_node_8: _,1,0
    # not_node_9: _,0,1
    # not_node_10: _,1,1
    # not_node_11: 1,1,_
    # not_node_12: 1,0,_
    # not_node_13: _,0,_
    # not_node_14: _,1,_
    # not_node_15: 0,_,0
    # not_node_16: 0,_,1

    leaf_1 = Leaf(id=5, truth_value=True)
    leaf_2 = Leaf(id=6, truth_value=False)
    leaf_3 = Leaf(id=7, truth_value=True)
    leaf_4 = Leaf(id=8, truth_value=False)
    leaf_5 = Leaf(id=9, truth_value=True)
    leaf_6 = Leaf(id=10, truth_value=True)
    node_4 = FeatNode(id=4, label=1, child_zero=leaf_5, child_one=leaf_6)
    node_3 = FeatNode(id=3, label=2, child_zero=leaf_1, child_one=leaf_2)
    node_2 = FeatNode(id=2, label=2, child_zero=leaf_4, child_one=node_4)
    node_1 = FeatNode(id=1, label=1, child_zero=node_3, child_one=leaf_3)
    root = FeatNode(id=0, label=0, child_zero=node_1, child_one=node_2)
    return Tree(root)


def run_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            IsNode(x),
            And(
                Subsumption(constant, x),
                Subsumption(x, constant)
            )
        )
    )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_not(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            Not(IsNode(x)),
            And(
                Subsumption(constant, x),
                Subsumption(x, constant)
            )
        )
    )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_and_extract_meaning(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            IsNode(x),
            And(
                Subsumption(constant, x),
                Subsumption(x, constant)
            )
        )
    )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    return result.returncode


def run_test_not_and_extract_meaning(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            Not(IsNode(x)),
            And(
                Subsumption(constant, x),
                Subsumption(x, constant)
            )
        )
    )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    return result.returncode


def test_is_node_root(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_3(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_5(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_leaf_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 10


def test_is_node_not_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_is_node_not_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, y) == 20


def test_not_is_node_root(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_3(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_5(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_leaf_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 20


def test_not_is_node_not_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_test_not(tree, file_name, solver_path, y) == 10


def test_not_is_node_not_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_not(tree, file_name, solver_path, y) == 10
