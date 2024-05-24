from os import path, makedirs
from time import time
from math import ceil
from pandas import DataFrame
from typing import Optional, Callable, Any
from multiprocessing import cpu_count, Pool, Manager
from experiments.helpers.formulas.base import kb_RFS
from experiments.helpers.plots import (
    generate_multi_label_curve_plot,
    generate_result_csv
)
from experiments.model import Experiment, FLAGS
from experiments.helpers.experiment_drivers import (
    min_max_rm_time_mean_driver,
    setup
)
from experiments.helpers.dtree_generators import (
    parallel_simple_tree_generation
)
from experiments.helpers.formulas.time_mean_over_nodes import (
    RFS,
    SR,
    minimal_RFS,
    minimal_SR,
    minimum_CR,
    minimum_SR,
    kb_SR
)


DRIVER = min_max_rm_time_mean_driver
TREE_GENERATOR = parallel_simple_tree_generation
EXPERIMENT_OUTPUT_DIR = path.join(
    'experiments',
    'experiment_results',
    'time_mean_over_nodes_experiments'
)
STATS_FILE = path.join(EXPERIMENT_OUTPUT_DIR, 'stats.csv')
TREE_NAME = 'tmod'
CSV_HEADER = 'nodes, dimension, seconds, instances'
TREE_DIR = path.join('dtrees', 'generated_trees', 'time_mean_over_nodes_trees')
TREES_PER_PAIR = 1
VECTORES_PER_TREE = 15
DIM_MIN = 50
DIM_MAX = 250
DIM_STEP = 50
NODE_MIN = 100
NODE_MAX = 2500
NODE_STEP = 50
K_RATIO = 0.05


FORMULA_GENERATORS: dict[str, Callable] = {
    'RFS': RFS,
    'SR': SR,
    'minimal_RFS': minimal_RFS,
    'minimal_SR': minimal_SR,
    'minimum_CR': minimum_CR,
    'minimum_SR': minimum_SR,
    'kb_SR': kb_SR,
    'kb_RFS': kb_RFS,
}


def run_dimension(
    generator_key: str,
    generator_args: list[Any],
    solver_path: str,
    dim: int,
    pos_data: list,
    neg_data: list,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None,
    lock=None
):
    start = time()
    for nodes in range(NODE_MIN, NODE_MAX + 1, NODE_STEP):
        tree_paths = [
            path.join(TREE_DIR, f'{TREE_NAME}_d{dim}_n{nodes}_{i}.json')
            for i in range(TREES_PER_PAIR)
        ]
        result = DRIVER(
            FORMULA_GENERATORS[generator_key],
            generator_args,
            tree_paths,
            dim,
            VECTORES_PER_TREE,
            nodes,
            solver_path,
            stats_info,
            lock is not None
        )
        time_label = flag if flag is not None else 'global'
        iter_pos_data = [
            result['pos']['nodes'],
            dim,
            result['pos'][time_label],
            result['pos']['instances']
        ]
        iter_neg_data = [
            result['neg']['nodes'],
            dim,
            result['neg'][time_label],
            result['neg']['instances']
        ]
        if lock is not None:
            lock.acquire()
        pos_data.append(iter_pos_data)
        neg_data.append(iter_neg_data)
        if lock is not None:
            lock.release()
    print(f'Dimension {dim} finished in {round(time() - start, 5)}')


def generate_plot(data: list, output_file: str, plot_title: str):
    data_frame = DataFrame(
        columns=['Nodes', 'Dimension', 'Seconds', 'Instances'],
        data=data
    )
    generate_multi_label_curve_plot(
        EXPERIMENT_OUTPUT_DIR,
        output_file,
        plot_title,
        (DIM_MAX - DIM_MIN) // DIM_STEP + 1,
        'Dimension',
        data_frame
    )


def run_experiment(
    generator_key: str,
    output_file: str,
    plot_title: str,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None,
) -> None:
    pos_data = []
    neg_data = []
    for dim in range(DIM_MIN, DIM_MAX + 1, DIM_STEP):
        run_dimension(
            generator_key,
            [dim, ceil(dim * K_RATIO)],
            solver_path,
            dim,
            pos_data,
            neg_data,
            stats_info,
            flag
        )
    generate_plot(pos_data, output_file + '_pos', plot_title + ' (pos)')
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_pos.csv',
        CSV_HEADER,
        pos_data
    )
    generate_plot(neg_data, output_file + '_neg', plot_title + ' (neg)')
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_neg.csv',
        CSV_HEADER,
        neg_data
    )


def run_parallel_experiment(
    generator_key: str,
    output_file: str,
    plot_title: str,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None
) -> None:
    cpu_number = cpu_count()
    cpu_number = 6 if cpu_number > 6 else cpu_number
    manager = Manager()
    pos_data = manager.list()
    neg_data = manager.list()
    lock = manager.Lock()
    pool = Pool(cpu_number)
    for dim in range(DIM_MIN, DIM_MAX + 1, DIM_STEP):
        pool.apply_async(
            run_dimension,
            args=(
                generator_key,
                [dim, ceil(dim * K_RATIO)],
                solver_path,
                dim,
                pos_data,
                neg_data,
                stats_info,
                flag,
                lock
            )
        )
    pool.close()
    pool.join()
    generate_plot(
        list(pos_data),
        output_file + '_pos',
        plot_title + ' (pos)'
    )
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_pos.csv',
        CSV_HEADER,
        list(pos_data)
    )
    generate_plot(
        list(neg_data),
        output_file + '_neg',
        plot_title + ' (neg)'
    )
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_neg.csv',
        CSV_HEADER,
        list(neg_data)
    )


def generate_trees():
    # Ensure dtree path here and generate it if needed
    if not path.exists(TREE_DIR):
        makedirs(TREE_DIR)
    TREE_GENERATOR(
        TREE_NAME,
        TREE_DIR,
        [DIM_MIN, DIM_MAX + 1, DIM_STEP],
        [NODE_MIN, NODE_MAX + 1, NODE_STEP],
        TREES_PER_PAIR
    )


def exp_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('SR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'SR',
            'time_mean_over_nodes_sr',
            'Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'SR',
        'time_mean_over_nodes_sr',
        'Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_RFS(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('RFS', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'RFS',
            'time_mean_over_nodes_rfs',
            'Relevant Feature Set',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'RFS',
        'time_mean_over_nodes_rfs',
        'Relevant Feature Set',
        solver_path,
        stats_info,
        flag
    )


def exp_minimal_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimalSR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimal_SR',
            'time_mean_over_nodes_minimal_sr',
            'Minimal Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimal_SR',
        'time_mean_over_nodes_minimal_sr',
        'Minimal Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_minimum_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimumSR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimum_SR',
            'time_mean_over_nodes_minimum_sr',
            'Minimum Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimum_SR',
        'time_mean_over_nodes_minimum_sr',
        'Minimum Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_minimal_RFS(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimalRFS', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimal_RFS',
            'time_mean_over_nodes_minimal_rfs',
            'Minimal Relevant Feature Set',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimal_RFS',
        'time_mean_over_nodes_minimal_rfs',
        'Minimal Relevant Feature Set',
        solver_path,
        stats_info,
        flag
    )


def exp_minimum_CR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimumCR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimum_CR',
            'time_mean_over_nodes_minimum_cr',
            'Minimum Change Required',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimum_CR',
        'time_mean_over_nodes_minimum_cr',
        'Minimum Change Required',
        solver_path,
        stats_info,
        flag
    )


def exp_kb_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('kBotsSR', STATS_FILE) if stats else None
    title = r'Query: $\exists y \,$ SR$(x, y)$' + \
        r'$\wedge |y|_{\bot} \geq k$'
    if parallel:
        run_parallel_experiment(
            'kb_SR',
            'time_mean_over_nodes_sr_with_at_least_k_bottoms',
            title,
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'kb_SR',
        'time_mean_over_nodes_sr_with_at_least_k_bottoms',
        title,
        solver_path,
        stats_info,
        flag
    )


def exp_kb_RFS(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: Optional[FLAGS] = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('kBotsRFS', STATS_FILE) if stats else None
    title = r'Query: $\exists y \,$ SR$(x, y)$' + \
        r'$\wedge |y|_{\bot} \geq k$'
    if parallel:
        run_parallel_experiment(
            'kb_RFS',
            'time_mean_over_nodes_sr_with_at_least_k_bottoms',
            title,
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'kb_RFS',
        'time_mean_over_nodes_sr_with_at_least_k_bottoms',
        title,
        solver_path,
        stats_info,
        flag
    )


experiment = Experiment(
    label='time_mean_over_nodes',
    variations={
        'sr': exp_SR,
        'rfs': exp_RFS,
        'minimal_sr': exp_minimal_SR,
        'minimum_sr': exp_minimum_SR,
        'minimal_rfs': exp_minimal_RFS,
        'minimum_cr': exp_minimum_CR,
        'kb_sr': exp_kb_SR,
        'kb_rfs': exp_kb_RFS
    },
    tree_generator=generate_trees
)
