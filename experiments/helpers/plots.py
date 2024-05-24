from os import path
import seaborn as sns
from random import sample
import matplotlib.pyplot as plt
from pandas import DataFrame


COLORS = [
    'brown',
    'maroon',
    'red',
    'orange',
    'olivedrab',
    'limegreen',
    'turquoise',
    'dodgerblue',
    'blue',
    'purple',
    'violet',
    'pink'
]


def generate_multi_label_curve_plot(
    dir_path: str,
    output_name: str,
    plot_title: str,
    dimensions: int,
    label: str,
    data: DataFrame
):
    if not path.exists(dir_path):
        raise FileNotFoundError(f'Dir or file \'{dir_path}\' does not exist.')
    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set_theme()
    palette = sample(COLORS, dimensions)
    plot = sns.lineplot(
        data=data,
        x='Nodes',
        y='Seconds',
        hue=label,
        palette=palette
    )
    plt.title(plot_title)
    plot.figure.savefig(path.join(dir_path, output_name), bbox_inches='tight')
    plt.clf()


def generate_result_csv(
    dir_path: str,
    output_name: str,
    header: str,
    data: list
):
    with open(path.join(dir_path, output_name), 'w') as file:
        file.write(header + '\n')
        for instance in data:
            file.write(','.join(map(lambda x: str(x), instance)) + '\n')
