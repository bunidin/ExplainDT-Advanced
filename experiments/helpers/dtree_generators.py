from os import path, makedirs
from multiprocessing import Process
from dtrees.dtree import random_dt_to_json
from dtrees.mnist_dt import train_tree, setup


def random_tree_generator(
    nodes_range_step: list[int],
    name: str,
    dir_path: str,
    dim: int,
    mult: int,
):
    for n_amount in range(*nodes_range_step):
        for m in range(mult):
            output_file = path.join(
                dir_path,
                f'{name}_d{dim}_n{n_amount}_{m}.json'
            )
            random_dt_to_json(output_file, dim, n_amount // 2 + 1, 50000)
    print(f'Dimension {dim} finished.')


def mnist_tree_generator(
    name: str,
    nodes_range_step: list[int],
    images,
    labels,
    test_images,
    test_labels,
    digit: int,
    dir_path: str,
    mult: int
):
    for n_amount in range(*nodes_range_step):
        for m in range(mult):
            output_file = path.join(
                dir_path,
                f'{name}_d{digit}_n{n_amount}_{m}.json'
            )
            train_tree(
                images,
                labels,
                test_images,
                test_labels,
                digit,
                n_amount // 2 + 1,
                output_file
            )
    print(f'Digit {digit} finished.')


def simple_tree_generation(
    name: str,
    dir_path: str,
    dimensions_range_step: list[int],
    nodes_range_step: list[int],
    mult: int
) -> None:
    print('Generating trees.')
    if not path.exists(dir_path):
        makedirs(dir_path)
    for dim in range(*dimensions_range_step):
        random_tree_generator(nodes_range_step, name, dir_path, dim, mult)
    print('Trees generated.')


def parallel_simple_tree_generation(
    name: str,
    dir_path: str,
    dimensions_range_step: list[int],
    nodes_range_step: list[int],
    mult: int
) -> None:
    print('Generating trees.')
    if not path.exists(dir_path):
        makedirs(dir_path)
    processes = []
    for dim in range(*dimensions_range_step):
        processes.append(
            Process(
                target=random_tree_generator,
                args=(
                    nodes_range_step,
                    name,
                    dir_path,
                    dim,
                    mult
                )
            )
        )
        processes[-1].start()
    for process in processes:
        process.join()
    print('Trees generated.')


def parallel_mnist_tree_generation(
    name: str,
    dir_path: str,
    digits: list[int],
    nodes_range_step: list[int],
    mult: int
):
    print('Generating trees.')
    if not path.exists(dir_path):
        makedirs(dir_path)
    processes = []
    images, labels, test_images, test_labels = setup()
    for digit in digits:
        processes.append(
            Process(
                target=mnist_tree_generator,
                args=(
                    name,
                    nodes_range_step,
                    images,
                    labels,
                    test_images,
                    test_labels,
                    digit,
                    dir_path,
                    mult
                )
            )
        )
        processes[-1].start()
    for process in processes:
        process.join()
    print('Trees generated.')
