from os import path, makedirs
from time import time
from dtrees.mnist_dt import load_test_images
from experiments.model import Experiment
from multiprocessing import cpu_count, Manager, Pool
from experiments.helpers.experiment_drivers import (
    output_extraction_mnist_driver,
    setup
)
from experiments.helpers.dtree_generators import (
    parallel_mnist_tree_generation
)
from experiments.helpers.formulas.mnist_visual import (
    extract_image,
    no_border_SR,
    kb_SR,
    kb_RFS,
    no_bot_SR,
    no_left_SR,
    no_right_SR,
    no_top_SR,
    structure_images
)
from experiments.vis_exp import solution_to_image


DRIVER = output_extraction_mnist_driver
TREE_GENERATOR = parallel_mnist_tree_generation
EXPERIMENT_OUTPUT_DIR = path.join(
    'experiments',
    'experiment_results',
    'mnist_vis'
)
STATS_FILE = path.join(EXPERIMENT_OUTPUT_DIR, 'stats.csv')
TREE_NAME = 'mnist'
CSV_HEADER = 'nodes, digits, seconds, acc'
TREE_DIR = path.join(
    'dtrees',
    'generated_trees',
    'mnist_time_mean_over_nodes_trees'
)
DIMENSION = 784
VECTORS_PER_TREE = 1
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NODES = 500


FORMULA_GENERATORS = {
    'no_border_SR': no_border_SR,
    'kb_SR': kb_SR,
    'kb_RFS': kb_RFS,
    'no_bot_SR': no_bot_SR,
    'no_left_SR': no_left_SR,
    'no_right_SR': no_right_SR,
    'no_top_SR': no_top_SR,
}


def run_digit(
    generator_key,
    generator_params,
    solver_path: str,
    digit: int,
    results,
    result_extractor=None,
    lock=None
) -> list[tuple[list[int], list[int]]] | None:
    start = time()
    print(f'Digit {digit} start.')
    tree_path = path.join(TREE_DIR, f'{TREE_NAME}_d{digit}_n{NODES}_{0}.json')
    digit_results = DRIVER(
        FORMULA_GENERATORS[generator_key],
        generator_params,
        tree_path,
        DIMENSION,
        digit,
        VECTORS_PER_TREE,
        solver_path,
        result_extractor
    )
    print(f'Digit {digit} finished in {round(time() - start, 5)}')
    if len(digit_results) > 0:
        if lock is not None:
            lock.acquire()
        results[digit] = digit_results
        if lock is not None:
            lock.release()


def run_experiment(
    generator_key,
    generator_params,
    output_file: str,
    solver_path: str,
    result_extractor=None,
    flag=None
) -> None:
    images, labels = load_test_images()
    structured_images = structure_images(images, labels)
    results: list[list[tuple[list[int], list[int]]]] = [[] for _ in DIGITS]
    for digit in DIGITS:
        image_idx = (digit + 1) % 10 if flag == 'neg' else digit
        images = structured_images[image_idx]
        run_digit(
            generator_key,
            [images] + generator_params,
            solver_path,
            digit,
            results,
            result_extractor
        )
    for digit, image_results in enumerate(results):
        if len(image_results) == 0:
            continue
        for i, pair in enumerate(image_results):
            solution_to_image(
                pair[0],
                pair[1],
                path.join(
                    EXPERIMENT_OUTPUT_DIR,
                    f'{output_file}_vis_d{digit}_{i}.png'
                )
            )


def run_parallel_experiment(
    generator_key,
    generator_params,
    output_file: str,
    solver_path: str,
    result_extractor=None,
    flag=None
) -> None:
    cpu_number = cpu_count()
    # cpu_number = cpu_number // 2 if cpu_number > 2 else cpu_number
    cpu_number = 10 if cpu_number > 10 else cpu_number
    images, labels = load_test_images()
    structured_images = structure_images(images, labels)
    manager = Manager()
    results = manager.list([[] for _ in DIGITS])
    lock = manager.Lock()
    pool = Pool(cpu_number)
    for digit in DIGITS:
        image_idx = (digit + 1) % 10 if flag == 'neg' else digit
        images = structured_images[image_idx]
        pool.apply_async(
            run_digit,
            args=(
                generator_key,
                [images] + generator_params,
                solver_path,
                digit,
                results,
                result_extractor,
                lock
            )
        )
    pool.close()
    pool.join()
    for digit, image_results in enumerate(list(results)):
        image_results = list(image_results)
        if len(image_results) == 0:
            continue
        for i, pair in enumerate(list(image_results)):
            solution_to_image(
                list(pair[0]),
                list(pair[1]),
                path.join(
                    EXPERIMENT_OUTPUT_DIR,
                    f'{output_file}_vis_d{digit}_{i}.png'
                )
            )


def generate_trees():
    # Ensure dtree path here and generate it if needed
    if not path.exists(TREE_DIR):
        makedirs(TREE_DIR)
    TREE_GENERATOR(
        TREE_NAME,
        TREE_DIR,
        DIGITS,
        [NODES, NODES + 1, NODES],
        1
    )


def exp_no_border_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'no_border_SR',
            [param],
            'no_border_sr',
            solver_path,
        )
        return
    run_experiment(
        'no_border_SR',
        [param],
        'no_border_sr',
        solver_path,
    )


def exp_no_top_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'no_top_SR',
            [param],
            'no_top_sr',
            solver_path,
        )
        return
    run_experiment(
        'no_top_SR',
        [param],
        'no_top_sr',
        solver_path,
    )


def exp_no_bot_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'no_bot_SR',
            [param],
            'no_bot_sr',
            solver_path,
        )
        return
    run_experiment(
        'no_bot_SR',
        [param],
        'no_bot_sr',
        solver_path,
    )


def exp_no_left_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'no_left_SR',
            [param],
            'no_left_sr',
            solver_path,
        )
        return
    run_experiment(
        'no_left_SR',
        [param],
        'no_left_sr',
        solver_path,
    )


def exp_no_right_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'no_right_SR',
            [param],
            'no_right_sr',
            solver_path,
        )
        return
    run_experiment(
        'no_right_SR',
        [param],
        'no_right_sr',
        solver_path,
    )


def exp_kb_SR(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'kb_SR',
            [param],
            f"kbSR_{param}{'_neg' if flag == 'neg' else ''}",
            solver_path,
            extract_image,
            flag
        )
        return
    run_experiment(
        'kb_SR',
        [param],
        f"kbSR_{param}{'_neg' if flag == 'neg' else ''}",
        solver_path,
        extract_image,
        flag
    )


def exp_kb_RFS(
    solver_path: str,
    parallel: bool = False,
    param: int = 0,
    flag: str | None = None
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    if parallel:
        run_parallel_experiment(
            'kb_RFS',
            [param],
            f"kbRFS_{param}{'_neg' if flag == 'neg' else ''}",
            solver_path,
            extract_image,
            flag
        )
        return
    run_experiment(
        'kb_RFS',
        [param],
        f"kbRFS_{param}{'_neg' if flag == 'neg' else ''}",
        solver_path,
        extract_image,
        flag
    )


experiment = Experiment(
    label='mnist_time_mean_over_nodes',
    variations={
        'no_border_sr': exp_no_border_SR,
        'no_top_sr': exp_no_top_SR,
        'no_bot_sr': exp_no_bot_SR,
        'no_left_sr': exp_no_left_SR,
        'no_right_sr': exp_no_right_SR,
        'kb_sr': exp_kb_SR,
        'kb_rfs': exp_kb_RFS,
    },
    tree_generator=generate_trees
)
