from os import path, remove, makedirs
from json import load
from re import compile
from subprocess import run
from time import process_time
from typing import Callable, Any, Optional
from pyparsing import Iterable
from components.base import generate_simplified_cnf, generate_cnf
from context import Context, Tree
from solvers import run_solver


TIME_REGEX = compile(r'([\d\.]+)\s*user')


def setup(experiment_path: str, solver_paths: Iterable[str]) -> None:
    # Check if solver path exists if not raise exception
    for s_path in solver_paths:
        if not path.exists(path.expanduser(s_path)):
            raise FileNotFoundError(f'Solver in \'{s_path}\' does not exist.')
    # If experiment output dir doesnt exists create it
    if not path.exists(experiment_path):
        makedirs(experiment_path)


def timed_function(function: Callable, *args, **kwargs) -> tuple[float, Any]:
    start = process_time()
    result = function(*args, **kwargs)
    return process_time() - start, result


def timed_subprocess(*args):
    result = run(['time', *args], capture_output=True, text=True)
    search = TIME_REGEX.search(result.stderr)
    group = search.group(1)
    if group is None:
        raise Exception('Time failed to be captured.')
    return float(group), result


def remove_min_max(values: list[int | float]) -> tuple[list[int | float], int]:
    if len(values) <= 6:
        return values, len(values)
    if len(values) == 0:
        return values, 0
    max_value = max(values)
    values.remove(max_value)
    if len(values) == 0:
        return values, 0
    min_value = min(values)
    values.remove(min_value)
    return values, len(values)


def generate_exec_stats(
    times: dict[str, list[int | float]],
    nodes: int,
    accuracy: Optional[list[float]] = None
):
    global_times = [sum(x) for x in zip(times['solver'], times['formula'])]
    instances = len(global_times)
    clean_solver_times, n_solver_times = remove_min_max(times['solver'])
    clean_formula_times, n_formula_times = remove_min_max(times['formula'])
    clean_global_times, n_global_times = remove_min_max(global_times)
    n_solver_times = n_solver_times if n_solver_times != 0 else 1
    n_formula_times = n_formula_times if n_formula_times != 0 else 1
    n_global_times = n_global_times if n_global_times != 0 else 1
    result = {
        'solver': sum(clean_solver_times) / n_solver_times,
        'formula': sum(clean_formula_times) / n_formula_times,
        'global': sum(clean_global_times) / n_global_times,
        'instances': instances,
        'nodes': nodes
    }
    if accuracy is not None:
        result['accuracy'] = sum(accuracy) / len(accuracy)
    return result


def record_formula_stats(
    query: str,
    file_path: str,
    dim: int,
    n_nodes: int,
    n_variables: int,
    n_clauses: int
):
    with open(file_path, 'a') as file:
        file.write(f'{dim},{n_nodes},{query},{n_variables},{n_clauses}\n')


def min_max_rm_time_mean_driver(
    formula_generator: Callable,
    generator_params: list[Any],
    tree_paths: list[str],
    dimension: int,
    mult: int,
    nodes: int,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    parallel: bool = False
) -> dict[str, dict[str, float | int]]:
    aux_file_name = f'aux_file_{dimension}_{nodes}.cnf'
    pos_times: dict[str, list[int | float]] = {
        'solver': [],
        'formula': [],
    }
    neg_times: dict[str, list[int | float]] = {
        'solver': [],
        'formula': [],
    }
    try:
        for t_path in tree_paths:
            for _ in range(mult):
                # Generate needed objects
                tree = Tree(from_file=t_path, iterative=True)
                context = Context(dimension, tree)
                # Create formula
                formula = formula_generator(*generator_params)
                formula_time, cnf = timed_function(
                    # generate_simplified_cnf,
                    generate_cnf,
                    formula,
                    context
                )
                cnf.to_file(aux_file_name)
                if stats_info is not None:
                    number_of_clauses = len(cnf.meaning_clauses) + \
                        len(cnf.consistency_clauses)
                    record_formula_stats(
                        stats_info[0],  # query
                        stats_info[1],  # file_path
                        dimension,
                        tree.number_of_nodes(),
                        cnf.nv,
                        number_of_clauses
                    )
                # Run solver
                if parallel:
                    solver_time, result = timed_subprocess(
                        solver_path,
                        aux_file_name
                    )
                else:
                    solver_time, result = timed_function(
                        run_solver,
                        solver_path,
                        aux_file_name
                    )
                if result.returncode == 10:
                    pos_times['formula'].append(formula_time)
                    pos_times['solver'].append(solver_time)
                else:
                    neg_times['formula'].append(formula_time)
                    neg_times['solver'].append(solver_time)
    except Exception as e:
        if path.isfile(aux_file_name):
            remove(aux_file_name)
        raise e
    if path.isfile(aux_file_name):
        remove(aux_file_name)

    nodes = tree.number_of_nodes()  # type: ignore
    return {
        'pos': generate_exec_stats(pos_times, nodes),
        'neg': generate_exec_stats(neg_times, nodes),
    }


def min_max_rm_mnist_time_mean_driver(
    formula_generator: Callable,
    generator_params: list[Any],
    tree_paths: list[str],
    dimension: int,
    digit: int,
    mult: int,
    nodes: int,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    parallel: bool = False
) -> dict[str, dict[str, float | int]]:
    aux_file_name = f'aux_file_{digit}_{nodes}.cnf'
    pos_times: dict[str, list[int | float]] = {
        'solver': [],
        'formula': [],
    }
    neg_times: dict[str, list[int | float]] = {
        'solver': [],
        'formula': [],
    }
    tree_accs = []
    try:
        for tree_path in tree_paths:
            file = open(tree_path)
            tree_accs.append(load(file)['accuracy'])
            file.close()
            for _ in range(mult):
                # Generate needed objects
                tree = Tree(from_file=tree_path, iterative=True)
                context = Context(dimension, tree)
                # Create formula
                formula = formula_generator(*generator_params)
                formula_time, cnf = timed_function(
                    generate_simplified_cnf,
                    formula,
                    context
                )
                cnf.to_file(aux_file_name)
                if stats_info is not None:
                    number_of_clauses = len(cnf.meaning_clauses) + \
                        len(cnf.consistency_clauses)
                    record_formula_stats(
                        stats_info[0],  # query
                        stats_info[1],  # file_path
                        dimension,
                        tree.number_of_nodes(),
                        cnf.nv,
                        number_of_clauses
                    )
                # Run solver
                if parallel:
                    solver_time, result = timed_subprocess(
                        solver_path,
                        aux_file_name
                    )
                else:
                    solver_time, result = timed_function(
                        run_solver,
                        solver_path,
                        aux_file_name
                    )
                if result.returncode == 10:
                    pos_times['formula'].append(formula_time)
                    pos_times['solver'].append(solver_time)
                else:
                    neg_times['formula'].append(formula_time)
                    neg_times['solver'].append(solver_time)
    except Exception as e:
        if path.isfile(aux_file_name):
            remove(aux_file_name)
        raise e
    if path.isfile(aux_file_name):
        remove(aux_file_name)
    nodes = tree.number_of_nodes()  # type: ignore
    return {
        'pos': generate_exec_stats(pos_times, nodes, tree_accs),
        'neg': generate_exec_stats(neg_times, nodes, tree_accs),
    }


def output_extraction_mnist_driver(
    formula_generator: Callable,
    generator_params: list[Any],
    tree_path: str,
    dimension: int,
    digit: int,
    mult: int,
    solver_path: str,
    result_extractor=None
) -> list[tuple[list[int], list[int]]]:
    aux_file_name = f'aux_file_vis{digit}.cnf'
    digit_results = []
    try:
        for _ in range(mult):
            # Generate needed objects
            tree = Tree(from_file=tree_path, iterative=False)
            context = Context(dimension, tree)
            # Create formula
            formula, images_chosen = formula_generator(*generator_params)
            cnf = generate_simplified_cnf(formula, context)
            cnf.to_file(aux_file_name)
            # Run solver
            result = run_solver(solver_path, aux_file_name)
            if result.returncode == 10:
                # Extract resulting vector here
                if result_extractor is not None:
                    images_chosen.append(
                        result_extractor(result.stdout, context)
                    )
                digit_results.append(images_chosen)
    except Exception as e:
        if path.isfile(aux_file_name):
            remove(aux_file_name)
        raise e
    if path.isfile(aux_file_name):
        remove(aux_file_name)
    return digit_results
