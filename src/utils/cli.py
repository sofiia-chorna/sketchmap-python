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
        is_flag=True,
        help="True if data is weighted",
        type=bool,
    )(func)


def high_dimension(func):
    return click.option(
        "--high_dimension",
        "--highd",
        help="Dimention to consider for the input highdim data",
        type=int,
    )(func)


def low_dimension(func):
    return click.option(
        "--low_dimension",
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
        "--select_mode",
        "--mode",
        help="Number of data to select",
        type=click.Choice(["minmax", "random"], case_sensitive=False),
        default="minmax",
    )(func)
