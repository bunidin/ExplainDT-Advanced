import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf, generate_simplified_cnf
from components.operators import And, Exists, Not
from components.predicates import RFS, Subsumption
from components.instances import Var, Constant
from solvers import run_solver,  extract_meaning


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
    #           leaf_3: 0,1,0 (True)
    #           leaf_4: 0,1,1 (True)
    #   node_2: 1,_,_
    #       node_5: 1,_,0
    #           leaf_5: 1,0,0 (False)
    #           leaf_6: 1,1,0 (True)
    #       node_6: 1,_,1
    #           leaf_7: 1,0,1 (False)
    #           leaf_8: 1,1,1 (True)
    # FULL, _,*,_, _,*,* and *,*,_ are RFS

    leaf_1 = Leaf(id=7, truth_value=False)
    leaf_2 = Leaf(id=8, truth_value=False)
    leaf_3 = Leaf(id=9, truth_value=True)
    leaf_4 = Leaf(id=10, truth_value=True)
    leaf_5 = Leaf(id=11, truth_value=False)
    leaf_6 = Leaf(id=12, truth_value=True)
    leaf_7 = Leaf(id=13, truth_value=False)
    leaf_8 = Leaf(id=14, truth_value=True)
    node_6 = FeatNode(id=6, label=1, child_zero=leaf_7, child_one=leaf_8)
    node_5 = FeatNode(id=5, label=1, child_zero=leaf_5, child_one=leaf_6)
    node_4 = FeatNode(id=4, label=2, child_zero=leaf_3, child_one=leaf_4)
    node_3 = FeatNode(id=3, label=2, child_zero=leaf_1, child_one=leaf_2)
    node_2 = FeatNode(id=2, label=2, child_zero=node_5, child_one=node_6)
    node_1 = FeatNode(id=1, label=1, child_zero=node_3, child_one=node_4)
    root = FeatNode(id=0, label=0, child_zero=node_1, child_one=node_2)

    node_1.parent = root
    node_2.parent = root
    node_3.parent = node_1
    node_4.parent = node_1
    node_5.parent = node_2
    node_6.parent = node_2
    leaf_1.parent = node_3
    leaf_2.parent = node_3
    leaf_3.parent = node_4
    leaf_4.parent = node_4
    leaf_5.parent = node_5
    leaf_6.parent = node_5
    leaf_7.parent = node_6
    leaf_8.parent = node_6

    tree = Tree(root)
    tree.pos_leafs = [leaf_3, leaf_4, leaf_6, leaf_8]
    tree.neg_leafs = [leaf_1, leaf_2, leaf_5, leaf_7]
    return tree


def run_test_var(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    simplify=False
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant),
                Subsumption(constant, x)
            ),
            RFS(x)
        )
    )
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_constant(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    simplify=False
) -> int:
    context = Context(3, tree)
    formula = RFS(constant)
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_not_var(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    simplify=False
) -> int:
    context = Context(3, tree)
    x = Var('x')
    formula = Exists(
        x,
        And(
            And(
                Subsumption(x, constant),
                Subsumption(constant, x)
            ),
            Not(RFS(x))
        )
    )
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_not_constant(
    tree: Tree,
    file_name: str,
    solver_path: str,
    constant: Constant,
    simplify=True
) -> int:
    context = Context(3, tree)
    formula = Not(RFS(constant))
    if simplify:
        cnf = generate_simplified_cnf(formula, context)
    else:
        cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_test_and_extract_meaning_var(
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
            And(
                Subsumption(x, constant),
                Subsumption(constant, x)
            ),
            RFS(x)
        )
    )
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    return result.returncode


def test_is_var_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x) == 10


def test_is_var_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x) == 20


def test_is_var_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x) == 20


def test_is_var_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x) == 20


def test_not_is_var_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x) == 10


def test_not_is_var_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x) == 10


def test_not_is_var_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x) == 10


def test_is_constant_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x) == 20


def test_is_constant_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x) == 20


def test_is_constant_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 20


def test_not_is_constant_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 10


def test_not_is_constant_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 10


def test_not_is_constant_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x) == 10


def test_s_is_var_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_var_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_var(tree, file_name, solver_path, x, True) == 20


def test_s_is_var_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x, True) == 20


def test_s_is_var_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_var_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 10


def test_s_not_is_var_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 10


def test_s_not_is_var_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_not_var(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 10


def test_s_is_constant_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 20


def test_s_is_constant_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 20


def test_s_is_constant_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_full_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_full_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ZERO))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_full_3(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_minimum_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ONE, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_minimum_2(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_partial_1(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.ZERO, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_sem_partial_2(tree, file_name, solver_path):
    x = Constant((Symbol.ONE, Symbol.ZERO, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 20


def test_s_not_is_constant_not_sem_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.BOT))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 10


def test_s_not_is_constant_not_sem_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.BOT, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 10


def test_s_not_is_constant_not_sem_3(tree, file_name, solver_path):
    x = Constant((Symbol.BOT, Symbol.BOT, Symbol.ONE))
    assert run_test_not_constant(tree, file_name, solver_path, x, True) == 10
