from os import path, makedirs
from random import choice, choices
from time import time
from dtrees.mnist_dt import load_test_images
from experiments.model import Experiment
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
from experiments.vis_exp import RFS_to_image, solution_to_image


DRIVER = output_extraction_mnist_driver
TREE_GENERATOR = parallel_mnist_tree_generation
EXPERIMENT_OUTPUT_DIR = path.join(
    'experiments',
    'experiment_results',
    'mnist_rfs'
)
STATS_FILE = path.join(EXPERIMENT_OUTPUT_DIR, 'stats.csv')
TREE_NAME = 'mnist'
TREE_DIR = path.join(
    'dtrees',
    'generated_trees',
    'mnist_time_mean_over_nodes_trees'
)
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DIMENSION = 784
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
    result_extractor,
) -> list[int] | None:
    start = time()
    print(f'Digit {digit} start.')
    tree_path = path.join(TREE_DIR, f'{TREE_NAME}_d{digit}_n{NODES}_{0}.json')
    digit_result = DRIVER(
        FORMULA_GENERATORS[generator_key],
        generator_params,
        tree_path,
        DIMENSION,
        digit,
        1,
        solver_path,
        result_extractor
    )
    print(f'Digit {digit} finished in {round(time() - start, 5)}')
    if len(digit_result) > 0:
        return digit_result[0][1]


def run_experiment(
    generator_key,
    generator_params,
    digit: int,
    overlay_number: int,
    output_file: str,
    solver_path: str,
    result_extractor,
) -> None:
    images, labels = load_test_images()
    structured_images = structure_images(images, labels)
    result = run_digit(
        generator_key,
        [structured_images[digit]] + generator_params,
        solver_path,
        digit,
        result_extractor
    )
    if result is None:
        return
    if overlay_number == 8:
        RFS_to_image(
            choices(structured_images[digit], k=8),
            result,
            path.join(
                EXPERIMENT_OUTPUT_DIR,
                f'{output_file}_vis_d{digit}_special.png'
            ),
            v2=True
        )
    for i in range(overlay_number):
        solution_to_image(
            choice(structured_images[digit]),
            result,
            path.join(
                EXPERIMENT_OUTPUT_DIR,
                f'{output_file}_vis_d{digit}_i{i}.png'
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


def exp_kb_RFS(
    solver_path: str,
    digit: int,
    overlays: int,
    param: int = 0,
):
    setup(EXPERIMENT_OUTPUT_DIR, [solver_path])
    run_experiment(
        'kb_RFS',
        [param],
        digit,
        overlays,
        f'kbRFS_{param}',
        solver_path,
        extract_image
    )


experiment = Experiment(
    label='mnist_time_mean_over_nodes',
    variations={
        'default': exp_kb_RFS,
    },
    tree_generator=generate_trees
)
