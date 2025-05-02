import click


def hdim_filepath(func):
    return click.option(
        "--hdim_filepath",
        "--P",
        required=True,
        help="Path to the high-dimentional data file (.pt or text)",
        type=str,
    )(func)


def output_filepath(func):
    return click.option(
        "--output_filepath",
        help="Path to the result file",
        type=str,
    )(func)


def max_distance(func):
    return click.option(
        "--max_distance",
        "--maxd",
        default=100,
        help="Maximum distance to consider in pairwise distance calculations",
        type=int,
    )(func)


def n_bin(func):
    return click.option(
        "--n_bin",
        default=1000,
        help="Number of bins to use in the histogram",
        type=int,
    )(func)


def weighted(func):
    return click.option(
        "--weighted",
        "--w",
        default=False,
        help="True if data is weighted",
        type=int,
    )(func)


def dimension(func):
    return click.option(
        "--dimension",
        "--d",
        help="Dimention to consider for the input data",
        type=int,
    )(func)
