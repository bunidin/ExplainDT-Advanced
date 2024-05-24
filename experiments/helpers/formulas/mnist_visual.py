from random import choice
from components.base import Component
from components.instances import Constant
from context import Context, Symbol
import experiments.helpers.formulas.base as fm
from solvers import extract_meaning


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


def constant_to_image(constant: Constant) -> list[int | None]:
    return list(
        map(
            lambda x: int(x == Symbol.ONE) if x != Symbol.BOT else None,
            constant.value
        )
    )


def image_to_no_border_constant(image: list[int]) -> Constant:
    # Image is of dimensions: 28 * 28 = 784. Index go from 0 to 27.
    values = []
    for indx, px in enumerate(image):
        i = indx // 28
        j = indx % 28
        if min(i, j) <= 3 or max(i, j) >= 24:
            values.append(Symbol.BOT)
            continue
        values.append(Symbol.ONE if px else Symbol.ZERO)
    return Constant(tuple(values))


def image_to_no_bot_constant(image: list[int]) -> Constant:
    values = []
    mid_point = MNIST_DIM / 2
    for indx, px in enumerate(image):
        if indx >= mid_point:
            values.append(Symbol.BOT)
            continue
        values.append(Symbol.ONE if px else Symbol.ZERO)
    return Constant(tuple(values))


def image_to_no_top_constant(image: list[int]) -> Constant:
    values = []
    mid_point = MNIST_DIM / 2
    for indx, px in enumerate(image):
        if indx <= mid_point:
            values.append(Symbol.BOT)
            continue
        values.append(Symbol.ONE if px else Symbol.ZERO)
    return Constant(tuple(values))


def image_to_no_left_constant(image: list[int]) -> Constant:
    values = []
    for indx, px in enumerate(image):
        j = indx % 28
        if j <= 13:
            values.append(Symbol.BOT)
            continue
        values.append(Symbol.ONE if px else Symbol.ZERO)
    return Constant(tuple(values))


def image_to_no_right_constant(image: list[int]) -> Constant:
    values = []
    for indx, px in enumerate(image):
        j = indx % 28
        if j >= 14:
            values.append(Symbol.BOT)
            continue
        values.append(Symbol.ONE if px else Symbol.ZERO)
    return Constant(tuple(values))


def no_border_SR(
    images: list[list[int]],
    *args,
    **kwargs
) -> tuple[Component, list[list[int] | list[int | None]]]:
    # This is very ugly.. im sorry but im dying.
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_no_border_constant(image)
    return fm.SR(x, y), [image, constant_to_image(y)]


def no_bot_SR(
    images: list[list[int]],
    *args,
    **kwargs
) -> tuple[Component, list[list[int] | list[int | None]]]:
    # This is very ugly.. im sorry but im dying.
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_no_bot_constant(image)
    return fm.SR(x, y), [image, constant_to_image(y)]


def no_top_SR(
    images: list[list[int]],
    *args,
    **kwargs
) -> tuple[Component, list[list[int] | list[int | None]]]:
    # This is very ugly.. im sorry but im dying.
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_no_top_constant(image)
    return fm.SR(x, y), [image, constant_to_image(y)]


def no_left_SR(
    images: list[list[int]],
    *args,
    **kwargs
) -> tuple[Component, list[list[int] | list[int | None]]]:
    # This is very ugly.. im sorry but im dying.
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_no_left_constant(image)
    return fm.SR(x, y), [image, constant_to_image(y)]


def no_right_SR(
    images: list[list[int]],
    *args,
    **kwargs
) -> tuple[Component, list[list[int] | list[int | None]]]:
    # This is very ugly.. im sorry but im dying.
    image = choice(images)
    x = image_to_constant(image)
    y = image_to_no_right_constant(image)
    return fm.SR(x, y), [image, constant_to_image(y)]


def kb_SR(
    images: list[list[int]],
    k: int,
    *args,
    **kwargs
) -> tuple[Component, list[list[int]]]:
    image = choice(images)
    x = image_to_constant(image)
    return fm.kb_SR(x, k), [image]


def kb_RFS(
    images: list[list[int]],
    k: int,
    *args,
    **kwargs
) -> tuple[Component, list[list[int]]]:
    image = choice(images)
    return fm.kb_RFS(MNIST_DIM, k), [image]


def extract_image(output: bytes, context: Context) -> list[int]:
    variable_dict = extract_meaning(output, context)
    image = []
    for i in range(MNIST_DIM):
        one = variable_dict[('y', i, Symbol.ONE)]
        bot = variable_dict[('y', i, Symbol.BOT)]
        image.append(int(one) if not bot else None)
    return image
