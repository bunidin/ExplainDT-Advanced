from math import ceil
from random import choices, sample
from components.base import Component
from components.instances import Constant
from context import Symbol
import experiments.helpers.formulas.base as fm


SUBSUMPTION_DIFF_RATIO = 0.1
NO_BOT_SYMBOLS = [Symbol.ONE, Symbol.ZERO]


def random_subsumption_generator(values: tuple[Symbol], ratio: float):
    diff = ceil(len(values) * ratio)
    indexes_to_flip = sample(range(len(values)), diff)
    for i, value in enumerate(values):
        yield value if i not in indexes_to_flip else Symbol.BOT


def random_instance_n_bots(dimensions: int, n: int) -> Constant:
    values = choices(NO_BOT_SYMBOLS, k=dimensions)
    indexes_to_flip = sample(range(len(values)), n)
    for i in indexes_to_flip:
        values[i] = Symbol.BOT
    return Constant(tuple(values))


def generate_SR_constants(dimension: int) -> tuple[Constant, Constant]:
    x = Constant(tuple(choices(NO_BOT_SYMBOLS, k=dimension)))
    y = Constant(
        tuple(random_subsumption_generator(x.value, SUBSUMPTION_DIFF_RATIO))
    )
    return (x, y)


def generate_RFS_constant(dimension: int) -> Constant:
    return random_instance_n_bots(dimension, 1)


def generate_CR_constants(dimension: int) -> tuple[Constant, Constant]:
    x = Constant(tuple(choices(NO_BOT_SYMBOLS, k=dimension)))
    y = Constant(tuple(choices(NO_BOT_SYMBOLS, k=dimension)))
    return (x, y)


def SR(dimension: int, *args, **kwargs) -> Component:
    x, y = generate_SR_constants(dimension)
    return fm.SR(x, y)


def RFS(dimension: int, *args, **kwargs) -> Component:
    x = generate_RFS_constant(dimension)
    return fm.RFS(x)


def minimal_SR(dimension: int, *args, **kwargs) -> Component:
    x, y = generate_SR_constants(dimension)
    return fm.minimal_SR(x, y)


def minimum_SR(dimension: int, *args, **kwargs) -> Component:
    x, y = generate_SR_constants(dimension)
    return fm.minimum_SR(x, y)


def minimal_RFS(dimension: int, *args, **kwargs) -> Component:
    x = generate_RFS_constant(dimension)
    return fm.minimal_RFS(x)


def minimum_CR(dimension: int, *args, **kwargs) -> Component:
    x, y = generate_CR_constants(dimension)
    return fm.minimum_CR(x, y)


def kb_SR(dimension: int, k: int, *args, **kwargs) -> Component:
    x = Constant(tuple(choices(NO_BOT_SYMBOLS, k=dimension)))
    return fm.kb_SR(x, k)


def kb_RFS(dimension: int, k: int, *args, **kwargs) -> Component:
    return fm.kb_RFS(dimension, k)
