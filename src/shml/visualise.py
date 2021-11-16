"""Plotting utility to conveniently see histograms for a relevant iterable."""
from __future__ import annotations

__all__ = ["histogrid"]

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import awkward as ak

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt


def histogrid(
    data: ak.Array,
    fields: list[str],
    thing_to_plot: Callable[[ak.Array, str], ak.Array],
    title: str | None = None,
    colors: Callable[..., Any] | None = None,
    bins: int | None = None,
    plots_per_row: int | None = None,
    subplot_title_pad: float | None = None,
    figsize_scale: float = 2.0,
    outfile_name: str | None = None,
) -> plt.Figure:
    """Utility function to visualize `thing_to_plot` over an iterable set of `fields`.
    Example usage:
        ```
        # import data
        all_data = ak.from_parquet(path/to/my/parquet/file)

        # grab names of background files from the EOS space holding the TTrees
        with open("filepath", "r") as f:
            filepath = f.readline().strip("\n")

        filenames = list(filter(lambda item: "root" in item, os.listdir(filepath)))
        signal_files = tuple(filter(lambda item: item[0] == "X", filenames))
        background_files = tuple(
            filter(lambda item: item not in signal_files, filenames)
        )

        # an example case to plot a chosen quantity for every background process.
        # we could also plot every quantity for one file if we wanted!
        # how to read this:
        ## for background data (category == 0):
        ## - iterate over each filename in background_files,
        ## - for the current filename, plot the "yy_phi" variable
        ## - title the plot
        ## - save to this filename
        shml.histogrid(
            data=all_data[all_data["category"] == 0],
            fields=background_files,
            thing_to_plot=lambda d, f: d[d["filename"] == f]["yy_phi"],
            title="first bjet phi for all backgrounds, post-preselection",
            outfile_name="test.png",
        )
        ```
    """
    # automatically select the number of plots per row to minimise blank plots
    if plots_per_row is None:
        if len(fields) == 1:
            plots_per_row = 1
        else:
            if len(fields) < 3:
                i = 2
            else:
                i = 3
            rem_dict = {}
            while (len(fields) % i != 0) and i < 10:
                rem_dict[i] = len(fields) % i
                i += 1
            if i < 10:
                plots_per_row = i
            else:
                plots_per_row = max(rem_dict, key=rem_dict.get)  # type: ignore
                # (https://github.com/python/mypy/issues/6692) ?

    rem = len(fields) % plots_per_row
    if rem == 0:
        rows = int(len(fields) / plots_per_row)
    else:
        rows = int(len(fields) / plots_per_row) + 1
    plt.rc(
        "figure",
        figsize=[plots_per_row * figsize_scale, rows * figsize_scale],
        facecolor="w",
    )
    fig, axes = plt.subplots(nrows=rows, ncols=plots_per_row, dpi=120)

    row = 0
    column = 0

    color1, color2 = (
        jnp.array(
            mpl.colors.to_rgb(
                "C9",
            ),
        ),
        jnp.array(mpl.colors.to_rgb("C6")),
    )
    default_colors = jax.vmap(
        partial(jnp.linspace, num=len(fields)),
    )(color1, color2).T

    def default_col(data: ak.Array, field: str, i: int) -> Any:
        return default_colors[i]

    colors = colors or default_col  # teal gang :)

    for i, field in enumerate(fields):
        if i % plots_per_row == 0 and i != 0:
            row += 1
            column -= plots_per_row

        # plotting callback (could be more general)
        axes[row, column].hist(
            thing_to_plot(
                data,
                field,
            ),
            bins=bins,
            color=colors(data, field, i),
        )
        axes[row, column].set_title(field, pad=subplot_title_pad)

        column += 1

    while column % plots_per_row != 0:
        fig.delaxes(axes[row, column])
        column += 1
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    if outfile_name is not None:
        plt.savefig(outfile_name, bbox_inches="tight")
