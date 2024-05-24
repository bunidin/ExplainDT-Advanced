import pytest
from os import path, remove
from components.base import generate_cnf, generate_simplified_cnf
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.operators import And, Exists, Not
from components.predicates import Subsumption, AllPoss, AllNeg
from components.instances import Optional, Var, Constant
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
    #           leaf_1: 0,0,0 (True)
    #           leaf_2: 0,0,1 (True)
    #       leaf_3: 0,1,_ (True)
    #   node_2: 1,_,_
    #       leaf_4: 1,_,0 (False)
    #       node_4: 1,_,1
    #           leaf_5: 1,0,1 (True)
    #           leaf_6: 1,1,1 (False)
    leaf_1 = Leaf(id=5, truth_value=True)
    leaf_2 = Leaf(id=6, truth_value=True)
    leaf_3 = Leaf(id=7, truth_value=True)
    leaf_4 = Leaf(id=8, truth_value=False)
    leaf_5 = Leaf(id=9, truth_value=True)
    leaf_6 = Leaf(id=10, truth_value=False)
    node_4 = FeatNode(id=4, label=1, child_zero=leaf_5, child_one=leaf_6)
    node_3 = FeatNode(id=3, label=2, child_zero=leaf_1, child_one=leaf_2)
    node_2 = FeatNode(id=2, label=2, child_zero=leaf_4, child_one=node_4)
    node_1 = FeatNode(id=1, label=1, child_zero=node_3, child_one=leaf_3)
    root = FeatNode(id=0, label=0, child_zero=node_1, child_one=node_2)
    return Tree(root)


def run_all_poss_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    is_constant: Optional[bool] = False,
    simplify: Optional[bool] = False
) -> int:
    context = Context(3, tree)
    if is_constant:
        formula = AllPoss(constant)
    else:
        x = Var('x')
        formula = Exists(
            x,
            And(
                AllPoss(x),
                And(
                    Subsumption(constant, x),
                    Subsumption(x, constant)
                )
            )
        )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_not_all_poss_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    is_constant: Optional[bool] = False,
    simplify: Optional[bool] = False
) -> int:
    context = Context(3, tree)
    if is_constant:
        formula = Not(AllPoss(constant))
    else:
        x = Var('x')
        formula = Exists(
            x,
            And(
                Not(AllPoss(x)),
                And(
                    Subsumption(constant, x),
                    Subsumption(x, constant)
                )
            )
        )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_all_neg_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    is_constant: Optional[bool] = False,
    simplify: Optional[bool] = False
) -> int:
    context = Context(3, tree)
    if is_constant:
        formula = AllNeg(constant)
    else:
        x = Var('x')
        formula = Exists(
            x,
            And(
                AllNeg(x),
                And(
                    Subsumption(constant, x),
                    Subsumption(x, constant)
                )
            )
        )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_not_all_neg_test(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    is_constant: Optional[bool] = False,
    simplify: Optional[bool] = False
) -> int:
    context = Context(3, tree)
    if is_constant:
        formula = Not(AllNeg(constant))
    else:
        x = Var('x')
        formula = Exists(
            x,
            And(
                Not(AllNeg(x)),
                And(
                    Subsumption(constant, x),
                    Subsumption(x, constant)
                )
            )
        )
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


# Nodes to test:
# node_1: _,_,_  -->  AllPoss: False  |  AllNeg: False
# node_2: 0,_,_  -->  AllPoss: True  |  AllNeg: False
# node_3: 1,_,_  -->  AllPoss: False  |  AllNeg: False
# node_4: 0,0,_  -->  AllPoss: True  |  AllNeg: False
# node_5: 0,1,_  -->  AllPoss: True  |  AllNeg: False
# node_6: 1,0,_  -->  AllPoss: False  |  AllNeg: False
# node_7: 1,1,_  -->  AllPoss: False  |  AllNeg: True
# node_8: 0,0,0  -->  AllPoss: True  |  AllNeg: False
# node_9: 0,0,1  -->  AllPoss: True  |  AllNeg: False
# node_10: 0,1,0  -->  AllPoss: True  |  AllNeg: False
# node_11: 0,1,1  -->  AllPoss: True  |  AllNeg: False
# node_12: 1,0,0  -->  AllPoss: False  |  AllNeg: True
# node_13: 1,0,1  -->  AllPoss: True  |  AllNeg: False
# node_14: 1,1,0  -->  AllPoss: False  |  AllNeg: True
# node_15: 1,1,1  -->  AllPoss: False  |  AllNeg: True
# node_16: _,0,_  -->  AllPoss: False  |  AllNeg: False
# node_17: _,1,_  -->  AllPoss: False  |  AllNeg: False
# node_18: _,0,0  -->  AllPoss: False  |  AllNeg: False
# node_19: _,0,1  -->  AllPoss: True  |  AllNeg: False
# node_20: _,1,0  -->  AllPoss: False  |  AllNeg: False
# node_21: _,1,1  -->  AllPoss: False  |  AllNeg: False
# node_22: _,_,0  -->  AllPoss: False  |  AllNeg: False
# node_23: _,_,1  -->  AllPoss: False  |  AllNeg: False
# node_24: 0,_,0  -->  AllPoss: True  |  AllNeg: False
# node_25: 0,_,1  -->  AllPoss: True  |  AllNeg: False
# node_26: 1,_,0  -->  AllPoss: False  |  AllNeg: True
# node_27: 1,_,1  -->  AllPoss: False  |  AllNeg: False


def test_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_not_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_not_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_constant_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_constant_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True) == 20


def test_constant_not_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y, True) == 10


def test_constant_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True) == 10


def test_constant_not_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y, True) == 20


def test_s_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y) == 20


def test_s_not_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(tree, file_name, solver_path, y) == 10


def test_s_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y) == 10


def test_s_not_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(tree, file_name, solver_path, y) == 20


def test_s_constant_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_not_is_all_poss_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_1(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_2(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_3(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_4(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_5(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_6(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_neg_node_7(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_8(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_9(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_10(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_11(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_neg_node_12(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_13(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_neg_node_14(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_not_is_all_neg_node_15(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_not_is_all_poss_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_16(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_17(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_18(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_19(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_20(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_21(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.ONE, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_22(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_23(tree, file_name, solver_path):
    y = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_24(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_poss_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_neg_node_25(tree, file_name, solver_path):
    y = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_neg_node_26(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20


def test_s_constant_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_poss_test(tree, file_name, solver_path, y, True, True) == 20


def test_s_constant_not_is_all_poss_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_poss_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 10


def test_s_constant_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_all_neg_test(tree, file_name, solver_path, y, True, True) == 10


def test_s_constant_not_is_all_neg_node_27(tree, file_name, solver_path):
    y = Constant((Symbol.ONE, Symbol.BOT, Symbol.ZERO))
    assert run_not_all_neg_test(
        tree,
        file_name,
        solver_path,
        y,
        True,
        True
    ) == 20
