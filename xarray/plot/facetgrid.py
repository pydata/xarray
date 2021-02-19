import functools
import itertools
import warnings

import numpy as np

from ..core.formatting import format_item
from .utils import (
    _get_nice_quiver_magnitude,
    _infer_xy_labels,
    _process_cmap_cbar_kwargs,
    import_matplotlib_pyplot,
    label_from_attrs,
)

# Overrides axes.labelsize, xtick.major.size, ytick.major.size
# from mpl.rcParams
_FONTSIZE = "small"
# For major ticks on x, y axes
_NTICKS = 5


def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[: (maxchar - 3)] + "..."

    return title


class FacetGrid:
    """
    Initialize the matplotlib figure and FacetGrid object.

    The :class:`FacetGrid` is an object that links a xarray DataArray to
    a matplotlib figure with a particular structure.

    In particular, :class:`FacetGrid` is used to draw plots with multiple
    Axes where each Axes shows the same relationship conditioned on
    different levels of some dimension. It's possible to condition on up to
    two variables by assigning variables to the rows and columns of the
    grid.

    The general approach to plotting here is called "small multiples",
    where the same kind of plot is repeated multiple times, and the
    specific use of small multiples to display the same relationship
    conditioned on one ore more other variables is often called a "trellis
    plot".

    The basic workflow is to initialize the :class:`FacetGrid` object with
    the DataArray and the variable names that are used to structure the grid.
    Then plotting functions can be applied to each subset by calling
    :meth:`FacetGrid.map_dataarray` or :meth:`FacetGrid.map`.

    Attributes
    ----------
    axes : numpy object array
        Contains axes in corresponding position, as returned from
        plt.subplots
    col_labels : list
        list of :class:`matplotlib.text.Text` instances corresponding to column titles.
    row_labels : list
        list of :class:`matplotlib.text.Text` instances corresponding to row titles.
    fig : matplotlib.Figure
        The figure containing all the axes
    name_dicts : numpy object array
        Contains dictionaries mapping coordinate names to values. None is
        used as a sentinel value for axes which should remain empty, ie.
        sometimes the bottom right grid
    """

    def __init__(
        self,
        data,
        col=None,
        row=None,
        col_wrap=None,
        sharex=True,
        sharey=True,
        figsize=None,
        aspect=1,
        size=3,
        subplot_kws=None,
    ):
        """
        Parameters
        ----------
        data : DataArray
            xarray DataArray to be plotted
        row, col : strings
            Dimesion names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the column variable at this width, so that the column facets
        sharex : bool, optional
            If true, the facets will share x axes
        sharey : bool, optional
            If true, the facets will share y axes
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.
            If set, overrides ``size`` and ``aspect``.
        aspect : scalar, optional
            Aspect ratio of each facet, so that ``aspect * size`` gives the
            width of each facet in inches
        size : scalar, optional
            Height (in inches) of each facet. See also: ``aspect``
        subplot_kws : dict, optional
            Dictionary of keyword arguments for matplotlib subplots

        """

        plt = import_matplotlib_pyplot()

        # Handle corner case of nonunique coordinates
        rep_col = col is not None and not data[col].to_index().is_unique
        rep_row = row is not None and not data[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError(
                "Coordinates used for faceting cannot "
                "contain repeated (nonunique) values."
            )

        # single_group is the grouping variable, if there is exactly one
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn("Ignoring col_wrap since both col and row were passed")
        elif row and not col:
            single_group = row
        elif not row and col:
            single_group = col
        else:
            raise ValueError("Pass a coordinate name as an argument for row or col")

        # Compute grid shape
        if single_group:
            nfacet = len(data[single_group])
            if col:
                # idea - could add heuristic for nice shapes like 3x4
                ncol = nfacet
            if row:
                ncol = 1
            if col_wrap is not None:
                # Overrides previous settings
                ncol = col_wrap
            nrow = int(np.ceil(nfacet / ncol))

        # Set the subplot kwargs
        subplot_kws = {} if subplot_kws is None else subplot_kws

        if figsize is None:
            # Calculate the base figure size with extra horizontal space for a
            # colorbar
            cbar_space = 1
            figsize = (ncol * size * aspect + cbar_space, nrow * size)

        fig, axes = plt.subplots(
            nrow,
            ncol,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            figsize=figsize,
            subplot_kw=subplot_kws,
        )

        # Set up the lists of names for the row and column facet variables
        col_names = list(data[col].values) if col else []
        row_names = list(data[row].values) if row else []

        if single_group:
            full = [{single_group: x} for x in data[single_group].values]
            empty = [None for x in range(nrow * ncol - len(full))]
            name_dicts = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dicts = [{row: r, col: c} for r, c in rowcols]

        name_dicts = np.array(name_dicts).reshape(nrow, ncol)

        # Set up the class attributes
        # ---------------------------

        # First the public API
        self.data = data
        self.name_dicts = name_dicts
        self.fig = fig
        self.axes = axes
        self.row_names = row_names
        self.col_names = col_names

        # guides
        self.figlegend = None
        self.quiverkey = None
        self.cbar = None

        # Next the private variables
        self._single_group = single_group
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._col_wrap = col_wrap
        self.row_labels = [None] * nrow
        self.col_labels = [None] * ncol
        self._x_var = None
        self._y_var = None
        self._cmap_extend = None
        self._mappables = []
        self._finalized = False

    @property
    def _left_axes(self):
        return self.axes[:, 0]

    @property
    def _bottom_axes(self):
        return self.axes[-1, :]

    def map_dataarray(self, func, x, y, **kwargs):
        """
        Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func : callable
            A plotting function with the same signature as a 2d xarray
            plotting method such as `xarray.plot.imshow`
        x, y : string
            Names of the coordinates to plot on x, y axes
        kwargs
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """

        if kwargs.get("cbar_ax", None) is not None:
            raise ValueError("cbar_ax not supported by FacetGrid.")

        cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
            func, self.data.values, **kwargs
        )

        self._cmap_extend = cmap_params.get("extend")

        # Order is important
        func_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"cmap", "colors", "cbar_kwargs", "levels"}
        }
        func_kwargs.update(cmap_params)
        func_kwargs.update({"add_colorbar": False, "add_labels": False})

        # Get x, y labels for the first subplot
        x, y = _infer_xy_labels(
            darray=self.data.loc[self.name_dicts.flat[0]],
            x=x,
            y=y,
            imshow=func.__name__ == "imshow",
            rgb=kwargs.get("rgb", None),
        )

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # None is the sentinel value
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(
                    subset, x=x, y=y, ax=ax, **func_kwargs, _is_facetgrid=True
                )
                self._mappables.append(mappable)

        self._finalize_grid(x, y)

        if kwargs.get("add_colorbar", True):
            self.add_colorbar(**cbar_kwargs)

        return self

    def map_dataarray_line(
        self, func, x, y, hue, add_legend=True, _labels=None, **kwargs
    ):
        from .plot import _infer_line_data

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # None is the sentinel value
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(
                    subset,
                    x=x,
                    y=y,
                    ax=ax,
                    hue=hue,
                    add_legend=False,
                    _labels=False,
                    **kwargs,
                )
                self._mappables.append(mappable)

        xplt, yplt, hueplt, huelabel = _infer_line_data(
            darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y, hue=hue
        )
        xlabel = label_from_attrs(xplt)
        ylabel = label_from_attrs(yplt)

        self._hue_var = hueplt
        self._hue_label = huelabel
        self._finalize_grid(xlabel, ylabel)

        if add_legend and hueplt is not None and huelabel is not None:
            self.add_legend()

        return self

    def map_dataset(
        self, func, x=None, y=None, hue=None, hue_style=None, add_guide=None, **kwargs
    ):
        from .dataset_plot import _infer_meta_data, _parse_size

        kwargs["add_guide"] = False

        if kwargs.get("markersize", None):
            kwargs["size_mapping"] = _parse_size(
                self.data[kwargs["markersize"]], kwargs.pop("size_norm", None)
            )

        meta_data = _infer_meta_data(
            self.data, x, y, hue, hue_style, add_guide, funcname=func.__name__
        )
        kwargs["meta_data"] = meta_data

        if hue and meta_data["hue_style"] == "continuous":
            cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
                func, self.data[hue].values, **kwargs
            )
            kwargs["meta_data"]["cmap_params"] = cmap_params
            kwargs["meta_data"]["cbar_kwargs"] = cbar_kwargs

        kwargs["_is_facetgrid"] = True

        if func.__name__ == "quiver" and "scale" not in kwargs:
            raise ValueError("Please provide scale.")
            # TODO: come up with an algorithm for reasonable scale choice

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # None is the sentinel value
            if d is not None:
                subset = self.data.loc[d]
                maybe_mappable = func(
                    ds=subset, x=x, y=y, hue=hue, hue_style=hue_style, ax=ax, **kwargs
                )
                # TODO: this is needed to get legends to work.
                # but maybe_mappable is a list in that case :/
                self._mappables.append(maybe_mappable)

        self._finalize_grid(meta_data["xlabel"], meta_data["ylabel"])

        if hue:
            self._hue_label = meta_data.pop("hue_label", None)
            if meta_data["add_legend"]:
                self._hue_var = meta_data["hue"]
                self.add_legend()
            elif meta_data["add_colorbar"]:
                self.add_colorbar(label=self._hue_label, **cbar_kwargs)

        if meta_data["add_quiverkey"]:
            self.add_quiverkey(kwargs["u"], kwargs["v"])

        return self

    def _finalize_grid(self, *axlabels):
        """Finalize the annotations and layout."""
        if not self._finalized:
            self.set_axis_labels(*axlabels)
            self.set_titles()
            self.fig.tight_layout()

            for ax, namedict in zip(self.axes.flat, self.name_dicts.flat):
                if namedict is None:
                    ax.set_visible(False)

            self._finalized = True

    def _adjust_fig_for_guide(self, guide):
        # Draw the plot to set the bounding boxes correctly
        renderer = self.fig.canvas.get_renderer()
        self.fig.draw(renderer)

        # Calculate and set the new width of the figure so the legend fits
        guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
        figure_width = self.fig.get_figwidth()
        self.fig.set_figwidth(figure_width + guide_width)

        # Draw the plot again to get the new transformations
        self.fig.draw(renderer)

        # Now calculate how much space we need on the right side
        guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
        space_needed = guide_width / (figure_width + guide_width) + 0.02
        # margin = .01
        # _space_needed = margin + space_needed
        right = 1 - space_needed

        # Place the subplot axes to give space for the legend
        self.fig.subplots_adjust(right=right)

    def add_legend(self, **kwargs):
        self.figlegend = self.fig.legend(
            handles=self._mappables[-1],
            labels=list(self._hue_var.values),
            title=self._hue_label,
            loc="center right",
            **kwargs,
        )
        self._adjust_fig_for_guide(self.figlegend)

    def add_colorbar(self, **kwargs):
        """Draw a colorbar"""
        kwargs = kwargs.copy()
        if self._cmap_extend is not None:
            kwargs.setdefault("extend", self._cmap_extend)
        # dont pass extend as kwarg if it is in the mappable
        if hasattr(self._mappables[-1], "extend"):
            kwargs.pop("extend", None)
        if "label" not in kwargs:
            kwargs.setdefault("label", label_from_attrs(self.data))
        self.cbar = self.fig.colorbar(
            self._mappables[-1], ax=list(self.axes.flat), **kwargs
        )
        return self

    def add_quiverkey(self, u, v, **kwargs):
        kwargs = kwargs.copy()

        magnitude = _get_nice_quiver_magnitude(self.data[u], self.data[v])
        units = self.data[u].attrs.get("units", "")
        self.quiverkey = self.axes.flat[-1].quiverkey(
            self._mappables[-1],
            X=0.8,
            Y=0.9,
            U=magnitude,
            label=f"{magnitude}\n{units}",
            labelpos="E",
            coordinates="figure",
        )

        # TODO: does not work because self.quiverkey.get_window_extent(renderer) = 0
        # https://github.com/matplotlib/matplotlib/issues/18530
        # self._adjust_fig_for_guide(self.quiverkey.text)
        return self

    def set_axis_labels(self, x_var=None, y_var=None):
        """Set axis labels on the left column and bottom row of the grid."""
        if x_var is not None:
            if x_var in self.data.coords:
                self._x_var = x_var
                self.set_xlabels(label_from_attrs(self.data[x_var]))
            else:
                # x_var is a string
                self.set_xlabels(x_var)

        if y_var is not None:
            if y_var in self.data.coords:
                self._y_var = y_var
                self.set_ylabels(label_from_attrs(self.data[y_var]))
            else:
                self.set_ylabels(y_var)
        return self

    def set_xlabels(self, label=None, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        if label is None:
            label = label_from_attrs(self.data[self._x_var])
        for ax in self._bottom_axes:
            ax.set_xlabel(label, **kwargs)
        return self

    def set_ylabels(self, label=None, **kwargs):
        """Label the y axis on the left column of the grid."""
        if label is None:
            label = label_from_attrs(self.data[self._y_var])
        for ax in self._left_axes:
            ax.set_ylabel(label, **kwargs)
        return self

    def set_titles(self, template="{coord} = {value}", maxchar=30, size=None, **kwargs):
        """
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for plot titles containing {coord} and {value}
        maxchar : int
            Truncate titles at maxchar
        kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: FacetGrid object

        """
        import matplotlib as mpl

        if size is None:
            size = mpl.rcParams["axes.labelsize"]

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)

        if self._single_group:
            for d, ax in zip(self.name_dicts.flat, self.axes.flat):
                # Only label the ones with data
                if d is not None:
                    coord, value = list(d.items()).pop()
                    title = nicetitle(coord, value, maxchar=maxchar)
                    ax.set_title(title, size=size, **kwargs)
        else:
            # The row titles on the right edge of the grid
            for index, (ax, row_name, handle) in enumerate(
                zip(self.axes[:, -1], self.row_names, self.row_labels)
            ):
                title = nicetitle(coord=self._row_var, value=row_name, maxchar=maxchar)
                if not handle:
                    self.row_labels[index] = ax.annotate(
                        title,
                        xy=(1.02, 0.5),
                        xycoords="axes fraction",
                        rotation=270,
                        ha="left",
                        va="center",
                        **kwargs,
                    )
                else:
                    handle.set_text(title)

            # The column titles on the top row
            for index, (ax, col_name, handle) in enumerate(
                zip(self.axes[0, :], self.col_names, self.col_labels)
            ):
                title = nicetitle(coord=self._col_var, value=col_name, maxchar=maxchar)
                if not handle:
                    self.col_labels[index] = ax.set_title(title, size=size, **kwargs)
                else:
                    handle.set_text(title)

        return self

    def set_ticks(self, max_xticks=_NTICKS, max_yticks=_NTICKS, fontsize=_FONTSIZE):
        """
        Set and control tick behavior

        Parameters
        ----------
        max_xticks, max_yticks : int, optional
            Maximum number of labeled ticks to plot on x, y axes
        fontsize : string or int
            Font size as used by matplotlib text

        Returns
        -------
        self : FacetGrid object

        """
        from matplotlib.ticker import MaxNLocator

        # Both are necessary
        x_major_locator = MaxNLocator(nbins=max_xticks)
        y_major_locator = MaxNLocator(nbins=max_yticks)

        for ax in self.axes.flat:
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            for tick in itertools.chain(
                ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()
            ):
                tick.label1.set_fontsize(fontsize)

        return self

    def map(self, func, *args, **kwargs):
        """
        Apply a plotting function to each facet's subset of the data.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : FacetGrid object

        """
        plt = import_matplotlib_pyplot()

        for ax, namedict in zip(self.axes.flat, self.name_dicts.flat):
            if namedict is not None:
                data = self.data.loc[namedict]
                plt.sca(ax)
                innerargs = [data[a].values for a in args]
                maybe_mappable = func(*innerargs, **kwargs)
                # TODO: better way to verify that an artist is mappable?
                # https://stackoverflow.com/questions/33023036/is-it-possible-to-detect-if-a-matplotlib-artist-is-a-mappable-suitable-for-use-w#33023522
                if maybe_mappable and hasattr(maybe_mappable, "autoscale_None"):
                    self._mappables.append(maybe_mappable)

        self._finalize_grid(*args[:2])

        return self


def _easy_facetgrid(
    data,
    plotfunc,
    kind,
    x=None,
    y=None,
    row=None,
    col=None,
    col_wrap=None,
    sharex=True,
    sharey=True,
    aspect=None,
    size=None,
    subplot_kws=None,
    ax=None,
    figsize=None,
    **kwargs,
):
    """
    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods

    kwargs are the arguments to 2d plotting method
    """
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError("cannot provide both `figsize` and `size` arguments")

    g = FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        aspect=aspect,
        size=size,
        subplot_kws=subplot_kws,
    )

    if kind == "line":
        return g.map_dataarray_line(plotfunc, x, y, **kwargs)

    if kind == "dataarray":
        return g.map_dataarray(plotfunc, x, y, **kwargs)

    if kind == "dataset":
        return g.map_dataset(plotfunc, x, y, **kwargs)
