import click


def hdim_filepath(func):
    return click.option(
        "--hdim-filepath",
        "--P",
        required=True,
        help="Path to the high-dimentional data file (.pt or text)",
        type=str,
    )(func)


def output_filepath(func):
    return click.option(
        "--output-filepath",
        help="Path to the result file",
        type=str,
    )(func)


def max_distance(func):
    return click.option(
        "--max-distance",
        "--maxd",
        help="Maximum distance to consider in pairwise distance calculations",
        type=int,
    )(func)


def n_bin(func):
    return click.option(
        "--n-bin",
        default=150,
        help="Number of bins to use in the histogram",
        type=int,
    )(func)


def weighted(func):
    return click.option(
        "--weighted/--no-weighted",
        "--w",
        default=True,
        is_flag=True,
        help="True if data is weighted",
        type=bool,
    )(func)


def compute_weights(func):
    return click.option(
        "--compute-weights",
        "--w",
        default=False,
        is_flag=True,
        help="Should compute voronoi weights",
        type=bool,
    )(func)


def numpy(func):
    return click.option(
        "--numpy",
        default=False,
        is_flag=True,
        help="If the data should be saved in numpy format",
        type=bool,
    )(func)


def high_dimension(func):
    return click.option(
        "--high-dimension",
        "--highd",
        help="Dimention to consider for the input highdim data",
        type=int,
    )(func)


def low_dimension(func):
    return click.option(
        "--low-dimension",
        "--lowd",
        help="Dimention to consider for the lowdim data",
        type=int,
    )(func)


def num(func):
    return click.option(
        "--num",
        "--n",
        default=1000,
        help="Number of data to select",
        type=int,
    )(func)


def select_mode(func):
    return click.option(
        "--select-mode",
        "--mode",
        help="Number of data to select",
        type=click.Choice(["minmax", "random"], case_sensitive=False),
        default="minmax",
    )(func)


def run_check(func):
    return click.option(
        "--run-check/--no-run-check",
        help="Run a check after the main operation",
        type=bool,
        default=False,
    )(func)


def metric(func):
    return click.option(
        "--metric",
        type=click.Choice(["euclidean", "dot", "pbc"]),
        default="euclidean",
        help="Distance metric",
    )(func)


def period(func):
    return click.option(
        "--period",
        type=float,
        default=0.0,
        help="Required for metric 'pbc'. Defines the lenght of the periodic box",
    )(func)


def sphere_period(func):
    return click.option(
        "--sphere-period",
        type=float,
        default=0.0,
        help="Required for metric 'sphere' (sperical distances). Defines the circumference of the sphere",
    )(func)


def save_indices(func):
    return click.option(
        "--save-indices/--no-save-indices",
        "--i",
        type=bool,
        is_flag=True,
        default=False,
        help="Should indices of the selected landmarks be saved as a first column",
    )(func)
