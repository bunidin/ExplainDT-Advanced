from os import path, makedirs
from time import time
from pandas import DataFrame
from typing import Optional, Callable, Any
from multiprocessing import cpu_count, Manager, Pool
from dtrees.mnist_dt import load_test_images
from experiments.helpers.plots import (
    generate_multi_label_curve_plot,
    generate_result_csv
)
from experiments.model import Experiment, FLAGS
from experiments.helpers.experiment_drivers import (
    min_max_rm_mnist_time_mean_driver,
    setup
)
from experiments.helpers.dtree_generators import (
    parallel_mnist_tree_generation
)
from experiments.helpers.formulas.mnist_time_mean_over_nodes import (
    RFS,
    SR,
    minimal_RFS,
    minimal_SR,
    minimum_CR,
    minimum_SR,
    kb_SR,
    kb_RFS,
    structure_images
)


DRIVER = min_max_rm_mnist_time_mean_driver
TREE_GENERATOR = parallel_mnist_tree_generation
EXPERIMENT_OUTPUT_DIR = path.join(
    'experiments',
    'experiment_results',
    'mnist_time_mean_over_nodes_experiments'
)
STATS_FILE = path.join(EXPERIMENT_OUTPUT_DIR, 'stats.csv')
TREE_NAME = 'mnist'
CSV_HEADER = 'nodes, digits, seconds, instances, acc'
TREE_DIR = path.join(
    'dtrees',
    'generated_trees',
    'mnist_time_mean_over_nodes_trees'
)
TREES_PER_PAIR = 1
DIMENSION = 784
VECTORS_PER_TREE = 20
# DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DIGITS = [0, 3, 4, 5, 8]
NODE_MIN = 100
NODE_MAX = 1000
NODE_STEP = 50


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
    digit: int,
    pos_data: list,
    neg_data: list,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None,
    lock=None
):
    start = time()
    print(f'Digit {digit} start.')
    for nodes in range(NODE_MIN, NODE_MAX + 1, NODE_STEP):
        tree_paths = [
            path.join(TREE_DIR, f'{TREE_NAME}_d{digit}_n{nodes}_{i}.json')
            for i in range(TREES_PER_PAIR)
        ]
        result = DRIVER(
            FORMULA_GENERATORS[generator_key],
            generator_args,
            tree_paths,
            DIMENSION,
            digit,
            VECTORS_PER_TREE,
            nodes,
            solver_path,
            stats_info,
            lock is not None
        )
        time_label = flag if flag is not None else 'global'
        iter_pos_data = [
            result['pos']['nodes'],
            digit,
            result['pos'][time_label],
            result['pos']['instances'],
            result['pos']['accuracy']
        ]
        iter_neg_data = [
            result['neg']['nodes'],
            digit,
            result['neg'][time_label],
            result['neg']['instances'],
            result['neg']['accuracy']
        ]
        if lock is not None:
            lock.acquire()
        pos_data.append(iter_pos_data)
        neg_data.append(iter_neg_data)
        if lock is not None:
            lock.release()
    print(f'Digit {digit} finished in {round(time() - start, 5)}')


def generate_plot(data: list, output_file: str, plot_title: str):
    data_frame = DataFrame(
        columns=['Nodes', 'Digits', 'Seconds', 'Instances', 'Accuracy'],
        data=data
    )
    generate_multi_label_curve_plot(
        EXPERIMENT_OUTPUT_DIR,
        output_file,
        plot_title,
        len(DIGITS),
        'Digits',
        data_frame
    )


def run_experiment(
    generator_key: str,
    generator_params: list[Any],
    output_file: str,
    plot_title: str,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None,
) -> None:
    pos_data = []
    neg_data = []
    images, labels = load_test_images()
    structured_images = structure_images(images, labels)
    for digit in DIGITS:
        run_dimension(
            generator_key,
            [structured_images[digit]] + generator_params,
            solver_path,
            digit,
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
    generator_params: list[Any],
    output_file: str,
    plot_title: str,
    solver_path: str,
    stats_info: Optional[tuple[str, str]] = None,
    flag: Optional[FLAGS] = None,
) -> None:
    cpu_number = cpu_count()
    # cpu_number = cpu_number // 2 if cpu_number > 2 else cpu_number
    cpu_number = 10 if cpu_number > 10 else cpu_number
    images, labels = load_test_images()
    structured_images = structure_images(images, labels)
    manager = Manager()
    pos_data = manager.list()
    neg_data = manager.list()
    lock = manager.Lock()
    pool = Pool(cpu_number)
    for digit in DIGITS:
        pool.apply_async(
            run_dimension,
            args=(
                generator_key,
                [structured_images[digit]] + generator_params,
                solver_path,
                digit,
                pos_data,
                neg_data,
                stats_info,
                flag,
                lock
            )
        )
    pool.close()
    pool.join()
    pos_data_list = list(pos_data)
    neg_data_list = list(neg_data)
    generate_plot(pos_data_list, output_file + '_pos', plot_title + ' (pos)')
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_pos.csv',
        CSV_HEADER,
        pos_data_list
    )
    generate_plot(neg_data_list, output_file + '_neg', plot_title + ' (neg)')
    generate_result_csv(
        EXPERIMENT_OUTPUT_DIR,
        output_file + '_neg.csv',
        CSV_HEADER,
        neg_data_list
    )


def generate_trees():
    # Ensure dtree path here and generate it if needed
    if not path.exists(TREE_DIR):
        makedirs(TREE_DIR)
    TREE_GENERATOR(
        TREE_NAME,
        TREE_DIR,
        DIGITS,
        [NODE_MIN, NODE_MAX + 1, NODE_STEP],
        TREES_PER_PAIR
    )


def exp_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('SR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'SR',
            [param],
            'mnsit_time_mean_over_nodes_sr',
            'Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'SR',
        [param],
        'mnsit_time_mean_over_nodes_sr',
        'Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_RFS(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('RFS', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'RFS',
            [param],
            'mnsit_time_mean_over_nodes_rfs',
            'Relevant Feature Set',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'RFS',
        [param],
        'mnsit_time_mean_over_nodes_rfs',
        'Relevant Feature Set',
        solver_path,
        stats_info,
        flag
    )


def exp_minimal_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimalSR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimal_SR',
            [param],
            'mnsit_time_mean_over_nodes_minimal_sr',
            'Minimal Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimal_SR',
        [param],
        'mnsit_time_mean_over_nodes_minimal_sr',
        'Minimal Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_minimum_SR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimumSR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimuma_SR',
            [param],
            'mnsit_time_mean_over_nodes_minimum_sr',
            'Minimum Sufficient Reason',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimuma_SR',
        [param],
        'mnsit_time_mean_over_nodes_minimum_sr',
        'Minimum Sufficient Reason',
        solver_path,
        stats_info,
        flag
    )


def exp_minimal_RFS(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimalRFS', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimal_RFS',
            [param],
            'mnsit_time_mean_over_nodes_minimal_rfs',
            'Minimal Relevant Feature Set',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimal_RFS',
        [param],
        'mnsit_time_mean_over_nodes_minimal_rfs',
        'Minimal Relevant Feature Set',
        solver_path,
        stats_info,
        flag
    )


def exp_minimum_CR(
    solver_path: str,
    parallel: bool = False,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('minimumCR', STATS_FILE) if stats else None
    if parallel:
        run_parallel_experiment(
            'minimum_CR',
            [param],
            'parallel_mnsit_time_mean_over_nodes_minimum_cr',
            'Minimum Change Required',
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'minimum_CR',
        [param],
        'mnsit_time_mean_over_nodes_minimum_cr',
        'Minimum Change Required',
        solver_path,
        stats_info,
        flag
    )


def exp_kb_SR(
    solver_path: str,
    parallel: bool,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('kBotsSR', STATS_FILE) if stats else None
    title = r'Query: $\exists y \,$ SR$(x, y)$' + \
        r'$\wedge |y|_{\bot} \geq' + f'{param}$'
    if parallel:
        run_parallel_experiment(
            'kb_SR',
            [param],
            f'mnsit_time_mean_over_nodes_sr_with_at_least_{param}_bottoms',
            title,
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'kb_SR',
        [param],
        f'mnsit_time_mean_over_nodes_sr_with_at_least_{param}_bottoms',
        title,
        solver_path,
        stats_info,
        flag
    )


def exp_kb_RFS(
    solver_path: str,
    parallel: bool,
    stats: Optional[bool] = False,
    flag: FLAGS = None,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    stats_info = ('kBotsRFS', STATS_FILE) if stats else None
    title = r'Query: $\exists y \,$ RFS$(x, y)$' + \
        r'$\wedge |y|_{\bot} \geq' + f'{param}$'
    if parallel:
        run_parallel_experiment(
            'kb_RFS',
            [param],
            f'mnsit_time_mean_over_nodes_rfs_with_at_least_{param}_bottoms',
            title,
            solver_path,
            stats_info,
            flag
        )
        return
    run_experiment(
        'kb_RFS',
        [param],
        f'mnsit_time_mean_over_nodes_rfs_with_at_least_{param}_bottoms',
        title,
        solver_path,
        stats_info,
        flag
    )


experiment = Experiment(
    label='mnist_time_mean_over_nodes',
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
