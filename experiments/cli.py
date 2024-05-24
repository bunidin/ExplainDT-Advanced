from typing import Optional
from experiments.experiment_time_mean_over_nodes import experiment as tmon_exp
from experiments.experiment_mnist_time_mean_over_nodes import (
    experiment as mnist_exp
)
from experiments.experiment_mnist_visual import (
    experiment as mnist_vis_exp
)
from experiments.experiment_mnist_rfs import experiment as mnist_rfs_exp
import typer


app = typer.Typer()


@app.command()
def tmon(
    solverpath: str,
    vars: list[str],
    parallel: Optional[bool] = False,
    stats: Optional[bool] = False,
    gt: Optional[bool] = False,
    flag: Optional[str] = None
):
    """
        Run "Time Mean Over Nodes" experiment.\n
        Suports Tree generation. \n
        Variations:\n
            - sr: Sufficient reason\n
            - rfs: Relevant feature set\n
            - minimal_sr: Minimal sufficient reason\n
            - minimum_sr: Minimum sufficient reason\n
            - minimal_rfs: Minimal relevant feature set\n
            - minimum_cr: Minimum change required\n
            - kb_sr: Exists SR with K bottoms.\n
    """
    typer.secho('Running experiments.', fg=typer.colors.GREEN)
    exception_or_none = tmon_exp.run_experiments(
        run_all=False,
        variation_labels=vars,
        generate_trees=gt,
        solver_path=solverpath,
        parallel=parallel,
        stats=stats,
        flag=flag
    )
    if exception_or_none is not None:
        typer.secho(exception_or_none, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho('Experiments done.', fg=typer.colors.GREEN)


@app.command()
def mnist_tmon(
    solverpath: str,
    vars: list[str],
    parallel: Optional[bool] = False,
    param: Optional[int] = 0,
    stats: Optional[bool] = False,
    gt: Optional[bool] = False,
    flag: Optional[str] = None
):
    """
        Run "Time Mean Over Nodes" experiment on MNIST trees.\n
        Suports Tree generation. \n
        Variations:\n
            - sr: Sufficient reason\n
            - rfs: Relevant feature set\n
            - minimal_sr: Minimal sufficient reason\n
            - minimum_sr: Minimum sufficient reason\n
            - minimal_rfs: Minimal relevant feature set\n
            - minimum_cr: Minimum change required\n
            - kb_sr: Exists SR with K bottoms. (required --param k flag)\n
            - kb_rfs: Exists RFS with K bottoms. (required --param k flag)\n
    """
    typer.secho('Running experiments.', fg=typer.colors.GREEN)
    exception_or_none = mnist_exp.run_experiments(
        run_all=False,
        variation_labels=vars,
        generate_trees=gt,
        solver_path=solverpath,
        parallel=parallel,
        stats=stats,
        param=param,
        flag=flag
    )
    if exception_or_none is not None:
        typer.secho(exception_or_none, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho('Experiments done.', fg=typer.colors.GREEN)


@app.command()
def mnist_vis(
    solverpath: str,
    vars: list[str],
    parallel: Optional[bool] = False,
    param: Optional[int] = 0,
    gt: Optional[bool] = False,
    flag: Optional[str] = None
):
    """
        Run "Visualization" experiment on MNIST trees.\n
        Suports Tree generation. \n
        Variations:\n
            - no_border_sr: Is the image without a border a sufficient reason\n
            - no_top_sr: Is the image without a top a sufficient reason\n
            - no_bot_sr: Is the image without the bottom a sufficient reason\n
            - no_left_sr: Is the image without the left a sufficient reason\n
            - no_right_sr: Is the image without the right a sufficient reason\n
            - kb_sr: Exists SR with K bottoms. (required --param k flag)\n
    """
    typer.secho('Running experiments.', fg=typer.colors.GREEN)
    exception_or_none = mnist_vis_exp.run_experiments(
        run_all=False,
        variation_labels=vars,
        generate_trees=gt,
        solver_path=solverpath,
        parallel=parallel,
        param=param,
        flag=flag
    )
    if exception_or_none is not None:
        typer.secho(exception_or_none, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho('Experiments done.', fg=typer.colors.GREEN)


@app.command()
def mnist_rfs(
    solverpath: str,
    digit: int,
    param: int,
    vars: list[str],
    overlays: Optional[int] = 1,
    gt: Optional[bool] = False,
):
    """
        Run "Find RFS Visualization" experiment on MNIST trees.\n
        Does NOT Suports Tree generation. \n
        Variations:\n
            - default: Exists SR with K bottoms on digit D.
            (requires: --param k, --digit d)\n
    """
    typer.secho('Running experiments.', fg=typer.colors.GREEN)
    exception_or_none = mnist_rfs_exp.run_experiments(
        run_all=False,
        variation_labels=vars,
        generate_trees=gt,
        solver_path=solverpath,
        digit=digit,
        overlays=overlays,
        param=param,
    )
    if exception_or_none is not None:
        typer.secho(exception_or_none, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho('Experiments done.', fg=typer.colors.GREEN)
