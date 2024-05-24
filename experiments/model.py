from typing import Literal
from typing import Callable, Optional


FLAGS = Literal['solver', 'formula'] | None


class Experiment:

    def __init__(
        self,
        label: str,
        variations: dict[str, Callable],
        tree_generator: Optional[Callable] = None
    ):
        self.label = label
        self._variations = variations
        self._tree_generator = tree_generator

    @property
    def supports_tree_generation(self):
        return self._tree_generator is not None

    @property
    def details(self):
        variation_list = [
            f'  - {label}\n'
            for label in self._variations.keys()
        ]
        return (
            f'Experiment: {self.label} \n' +
            f'Variations: {"None" if self._variations is None else ""}\n' +
            ''.join(variation_list) +
            f'Supports tree generation: {self._tree_generator is not None}'
        )

    def run_experiments(
        self,
        run_all: Optional[bool] = False,
        variation_labels: Optional[list[str]] = None,
        generate_trees: Optional[bool] = False,
        **kwargs
    ) -> Exception | None:
        if not self._variations:
            return Exception('Experiment does not declare variations.')
        if generate_trees:
            if self._tree_generator is None:
                return Exception('Experiment does not declare tree generator.')
            self._tree_generator()
        if run_all:
            for experiment in self._variations.values():
                experiment(**kwargs)
            return
        elif variation_labels is not None:
            for label in variation_labels:
                if label not in self._variations.keys():
                    return Exception(
                        f'No variation with label {label} exists.'
                    )
            for label in variation_labels:
                self._variations[label](**kwargs)
        else:
            return Exception('No experiment variations specified.')
