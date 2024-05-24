from math import ceil
from random import sample, choice
from components.base import Component
from components.instances import Constant
from context import Symbol
import experiments.helpers.formulas.base as fm


MNIST_DIM = 784


def structure_images(
    images,
    labels
) -> list[list[list[int]]]:
    structured_images = [[] for _ in range(10)]
    for i, image in enumerate(images):
        structured_images[labels[i]].append(image)
    return structured_images


def image_to_constant(image: list[int]) -> Constant:
    return Constant(
        tuple(map(lambda px: Symbol.ONE if px else Symbol.ZERO, image))
    )


def image_to_partial_constant(image: list[int]) -> Constant:
    values = [Symbol.ONE if px else Symbol.ZERO for px in image]
    indexes_to_flip = sample(range(len(image)), k=ceil(len(image) * 0.3))
    for i in indexes_to_flip:
        values[i] = Symbol.BOT
    return Constant(tuple(values))


def SR(images: list[list[int]], *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_partial_constant(image)
    return fm.SR(x, y)


def RFS(images: list[list[int]], *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_partial_constant(image)
    return fm.RFS(x)


def minimal_SR(images: list[list[int]], *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_partial_constant(image)
    return fm.minimal_SR(x, y)


def minimum_SR(images: list[list[int]], *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_partial_constant(image)
    return fm.minimum_SR(x, y)


def minimal_RFS(images: list[list[int]], *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_partial_constant(image)
    return fm.minimal_RFS(x)


def minimum_CR(images: list[list[int]], *args, **kwargs) -> Component:
    x_image = choice(images)
    y_image = choice(images)
    x = image_to_constant(x_image)
    y = image_to_constant(y_image)
    return fm.minimum_CR(x, y)


def kb_SR(images: list[list[int]], k: int, *args, **kwargs) -> Component:
    image = choice(images)
    x = image_to_constant(image)
    return fm.kb_SR(x, k)


def kb_RFS(images: list[list[int]], k: int, *args, **kwargs) -> Component:
    return fm.kb_RFS(MNIST_DIM, k)
