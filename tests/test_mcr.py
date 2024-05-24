import pytest
from os import path, remove
from context import FeatNode, Leaf, Tree, Context, Symbol
from components.base import generate_cnf
from components.operators import And, Exists, Or, Not
from components.predicates import AllPoss, LEH, Full
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
    #           leaf_3: 0,1,0 (False)
    #           leaf_4: 0,1,1 (True)
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


def nMCR(x: Constant, y: Constant):
    z = Var('z')
    PHI_x_y = And(
        Or(
            Not(AllPoss(x)),
            AllPoss(y)
        ),
        Or(
            Not(AllPoss(y)),
            AllPoss(x)
        )
    )
    PHI_x_z = And(
        Or(
            Not(AllPoss(x)),
            AllPoss(z)
        ),
        Or(
            Not(AllPoss(z)),
            AllPoss(x)
        )
    )
    RHO = Or(
        Not(Full(z)),
        Or(
            PHI_x_z,
            LEH(x, y, z)
        )
    )
    return Exists(
        z,
        Not(
            And(
                Full(x),
                And(
                    Full(y),
                    And(
                        Not(PHI_x_y),
                        RHO
                    )
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
    formula = nMCR(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    return run_solver(solver_path, file_name).returncode


def run_phi(
    tree: Tree,
    file_name: str,
    solver_path: str,
    x: Constant,
    y: Constant
) -> int:
    context = Context(3, tree)
    formula = And(
        Or(
            Not(AllPoss(x)),
            AllPoss(y)
        ),
        Or(
            Not(AllPoss(y)),
            AllPoss(x)
        )
    )
    # formula = Or(
    #     Not(AllPoss(y)),
    #     AllPoss(x)
    # )
    formula = Not(
        And(
            Not(Not(AllPoss(y))),
            Not(AllPoss(x))
        )
    )
    # formula = Not(
    #     And(
    #         Not(Not(Full(x))),
    #         Not(Not(Full(x)))
    #     )
    # )

    # formula = Not(AllPoss(x))
    # print(pre_cnf.clauses)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    return result.returncode


def run_test_and_extract_meaning(
    tree: Tree,
    file_name: str,
    solver_path: str,
    x: Constant,
    y: Constant
) -> int:
    context = Context(3, tree)
    formula = nMCR(x, y)
    # contextualize(formula, context)
    # formula.encode().to_file(file_name)
    cnf = generate_cnf(formula, context)
    cnf.to_file(file_name)
    result = run_solver(solver_path, file_name)
    print('\n')
    for key, value in extract_meaning(result.stdout, context).items():
        print(key, value)
    return result.returncode


def test_is_mcr_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, x, y) == 20


def test_is_mcr_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    y = Constant((Symbol.ZERO, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, x, y) == 20


def test_is_not_mcr_1(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ONE, Symbol.ONE, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, x, y) == 10


def test_is_not_mcr_2(tree, file_name, solver_path):
    x = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ZERO))
    y = Constant((Symbol.ZERO, Symbol.ZERO, Symbol.ONE))
    assert run_test(tree, file_name, solver_path, x, y) == 10
