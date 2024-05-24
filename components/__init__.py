from os import remove
from components.base import Component, generate_simplified_cnf
from solvers import run_solver
from context import Context, Tree


def eval_formula(
    formula: Component,
    dimension: int,
    tree_file_path: str,
    solver_file_path: str
) -> bool:
    aux_file = 'aux_file.cnf'
    context = Context(
        dimension,
        Tree(from_file=tree_file_path, iterative=True)
    )
    cnf = generate_simplified_cnf(formula, context)
    cnf.to_file(aux_file)
    result = run_solver(solver_file_path, aux_file)
    remove(aux_file)
    return result.returncode == 10
