from __future__ import absolute_import, division, print_function

import functools
import itertools
import warnings

import numpy as np

from ..core.formatting import format_item
from ..core.pycompat import getargspec
from .utils import (
    _determine_cmap_params, _infer_xy_labels, import_matplotlib_pyplot)

# Overrides axes.labelsize, xtick.major.size, ytick.major.size
# from mpl.rcParams
_FONTSIZE = 'small'
# For major ticks on x, y axes
_NTICKS = 5


def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[:(maxchar - 3)] + '...'

    return title


class FacetGrid(object):
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
    fig : matplotlib.Figure
        The figure containing all the axes
    name_dicts : numpy object array
        Contains dictionaries mapping coordinate names to values. None is
        used as a sentinel value for axes which should remain empty, ie.
        sometimes the bottom right grid

    """

    def __init__(self, data, col=None, row=None, col_wrap=None,
                 sharex=True, sharey=True, figsize=None, aspect=1, size=3,
                 subplot_kws=None):
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
            raise ValueError('Coordinates used for faceting cannot '
                             'contain repeated (nonunique) values.')

        # single_group is the grouping variable, if there is exactly one
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn('Ignoring col_wrap since both col and row '
                              'were passed')
        elif row and not col:
            single_group = row
        elif not row and col:
            single_group = col
        else:
            raise ValueError(
                'Pass a coordinate name as an argument for row or col')

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

        fig, axes = plt.subplots(nrow, ncol,
                                 sharex=sharex, sharey=sharey, squeeze=False,
                                 figsize=figsize, subplot_kw=subplot_kws)

        # Set up the lists of names for the row and column facet variables
        col_names = list(data[col].values) if col else []
        row_names = list(data[row].values) if row else []

        if single_group:
            full = [{single_group: x} for x in
                    data[single_group].values]
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

        # Next the private variables
        self._single_group = single_group
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._col_wrap = col_wrap
        self._x_var = None
        self._y_var = None
        self._cmap_extend = None
        self._mappables = []

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
        kwargs :
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """

        cmapkw = kwargs.get('cmap')
        colorskw = kwargs.get('colors')

        # colors is mutually exclusive with cmap
        if cmapkw and colorskw:
            raise ValueError("Can't specify both cmap and colors.")

        # These should be consistent with xarray.plot._plot2d
        cmap_kwargs = {'plot_data': self.data.values,
                       # MPL default
                       'levels': 7 if 'contour' in func.__name__ else None,
                       'filled': func.__name__ != 'contour',
                       }

        cmap_args = getargspec(_determine_cmap_params).args
        cmap_kwargs.update((a, kwargs[a]) for a in cmap_args if a in kwargs)

        cmap_params = _determine_cmap_params(**cmap_kwargs)

        if colorskw is not None:
            cmap_params['cmap'] = None

        # Order is important
        func_kwargs = kwargs.copy()
        func_kwargs.update(cmap_params)
        func_kwargs.update({'add_colorbar': False, 'add_labels': False})

        # Get x, y labels for the first subplot
        x, y = _infer_xy_labels(
            darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y,
            imshow=func.__name__ == 'imshow', rgb=kwargs.get('rgb', None))

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # None is the sentinel value
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(subset, x, y, ax=ax, **func_kwargs)
                self._mappables.append(mappable)

        self._cmap_extend = cmap_params.get('extend')
        self._finalize_grid(x, y)

        if kwargs.get('add_colorbar', True):
            self.add_colorbar()

        return self

    def _finalize_grid(self, *axlabels):
        """Finalize the annotations and layout."""
        self.set_axis_labels(*axlabels)
        self.set_titles()
        self.fig.tight_layout()

        for ax, namedict in zip(self.axes.flat, self.name_dicts.flat):
            if namedict is None:
                ax.set_visible(False)

    def add_colorbar(self, **kwargs):
        """Draw a colorbar
        """
        kwargs = kwargs.copy()
        if self._cmap_extend is not None:
            kwargs.setdefault('extend', self._cmap_extend)
        if getattr(self.data, 'name', None) is not None:
            kwargs.setdefault('label', self.data.name)
        self.cbar = self.fig.colorbar(self._mappables[-1],
                                      ax=list(self.axes.flat),
                                      **kwargs)
        return self

    def set_axis_labels(self, x_var=None, y_var=None):
        """Set axis labels on the left column and bottom row of the grid."""
        if x_var is not None:
            self._x_var = x_var
            self.set_xlabels(x_var)
        if y_var is not None:
            self._y_var = y_var
            self.set_ylabels(y_var)
        return self

    def set_xlabels(self, label=None, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        if label is None:
            label = self._x_var
        for ax in self._bottom_axes:
            ax.set_xlabel(label, **kwargs)
        return self

    def set_ylabels(self, label=None, **kwargs):
        """Label the y axis on the left column of the grid."""
        if label is None:
            label = self._y_var
        for ax in self._left_axes:
            ax.set_ylabel(label, **kwargs)
        return self

    def set_titles(self, template="{coord} = {value}", maxchar=30,
                   **kwargs):
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

        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar,
                                      template=template)

        if self._single_group:
            for d, ax in zip(self.name_dicts.flat, self.axes.flat):
                # Only label the ones with data
                if d is not None:
                    coord, value = list(d.items()).pop()
                    title = nicetitle(coord, value, maxchar=maxchar)
                    ax.set_title(title, **kwargs)
        else:
            # The row titles on the right edge of the grid
            for ax, row_name in zip(self.axes[:, -1], self.row_names):
                title = nicetitle(coord=self._row_var, value=row_name,
                                  maxchar=maxchar)
                ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                            rotation=270, ha="left", va="center", **kwargs)

            # The column titles on the top row
            for ax, col_name in zip(self.axes[0, :], self.col_names):
                title = nicetitle(coord=self._col_var, value=col_name,
                                  maxchar=maxchar)
                ax.set_title(title, **kwargs)

        return self

    def set_ticks(self, max_xticks=_NTICKS, max_yticks=_NTICKS,
                  fontsize=_FONTSIZE):
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
            for tick in itertools.chain(ax.xaxis.get_major_ticks(),
                                        ax.yaxis.get_major_ticks()):
                tick.label.set_fontsize(fontsize)

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
                # TODO: is it possible to verify that an artist is mappable?
                mappable = func(*innerargs, **kwargs)
                self._mappables.append(mappable)

        self._finalize_grid(*args[:2])

        return self
