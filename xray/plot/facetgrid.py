from __future__ import division

import warnings
import itertools
import functools

import numpy as np
import pandas as pd

from ..core.formatting import format_item
from .plot import _determine_cmap_params


# Overrides axes.labelsize, xtick.major.size, ytick.major.size
# from mpl.rcParams
_FONTSIZE = 'small'
# For major ticks on x, y axes
_NTICKS = 5


def _nicetitle(coord, value, maxchar, template):
    '''
    Put coord, value in template and truncate at maxchar
    '''
    prettyvalue = format_item(value)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[:(maxchar - 3)] + '...'

    return title


class FacetGrid(object):
    '''
    Initialize the matplotlib figure and FacetGrid object.

    The :class:`FacetGrid` is an object that links a xray DataArray to
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
    the DataArray and the variable names that are used to structure the grid. Then
    one or more plotting functions can be applied to each subset by calling
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

    '''

    def __init__(self, darray, col=None, row=None, col_wrap=None,
            aspect=1, size=3, margin_titles=True):
        '''
        Parameters
        ----------
        darray : DataArray
            xray DataArray to be plotted
        row, col : strings
            Dimesion names that define subsets of the data, which will be drawn on
            separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the column variable at this width, so that the column facets
        aspect : scalar, optional
            Aspect ratio of each facet, so that ``aspect * size`` gives the width
            of each facet in inches
        size : scalar, optional
            Height (in inches) of each facet. See also: ``aspect``
        margin_titles : bool, optional
            If ``True``, the titles for the row variable are drawn to the right of
            the last column. This option is experimental and may not work in all
            cases.

        '''

        import matplotlib.pyplot as plt

        # Handle corner case of nonunique coordinates
        rep_col = col is not None and not darray[col].to_index().is_unique
        rep_row = row is not None and not darray[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError('Coordinates used for faceting cannot '
                             'contain repeated (nonunique) values.')

        # self._single_group is the grouping variable, if there is exactly one
        if col and row:
            self._single_group = False
            self._nrow = len(darray[row])
            self._ncol = len(darray[col])
            self.nfacet = self._nrow * self._ncol
            if col_wrap is not None:
                warnings.warn('Ignoring col_wrap since both col and row '
                              'were passed')
        elif row and not col:
            self._single_group = row
        elif not row and col:
            self._single_group = col
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')

        # Compute grid shape
        if self._single_group:
            self.nfacet = len(darray[self._single_group])
            if col:
                # idea - could add heuristic for nice shapes like 3x4
                self._ncol = self.nfacet
                margin_titles = False
            if row:
                self._ncol = 1
            if col_wrap is not None:
                # Overrides previous settings
                self._ncol = col_wrap
            self._nrow = int(np.ceil(self.nfacet / self._ncol))

        # Calculate the base figure size with extra horizontal space for a colorbar
        self._cbar_space = 1
        figsize = (self._ncol * size * aspect + self._cbar_space, self._nrow * size)

        self.fig, self.axes = plt.subplots(self._nrow, self._ncol,
                sharex=True, sharey=True, squeeze=False, figsize=figsize)

        # Set up the lists of names for the row and column facet variables
        col_names = list(darray[col].values) if col else []
        row_names = list(darray[row].values) if row else []

        if self._single_group:
            full = [{self._single_group: x} for x in
                      darray[self._single_group].values]
            empty = [None for x in range(self._nrow * self._ncol - len(full))]
            name_dicts = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dicts = [{row: r, col: c} for r, c in rowcols]

        self.name_dicts = np.array(name_dicts).reshape(self._nrow, self._ncol)

        self.row_names = row_names
        self.col_names = col_names
        self.darray = darray
        self.row = row
        self.col = col
        self.col_wrap = col_wrap

        # Next the private variables
        self._margin_titles = margin_titles
        self._row_var = row
        self._col_var = col
        self._col_wrap = col_wrap

        self.set_titles()

    def __iter__(self):
        return self.axes.flat

    def map_dataarray(self, plotfunc, x, y, max_xticks=4, max_yticks=4,
            fontsize=_FONTSIZE, **kwargs):
        '''
        Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        plotfunc : callable
            A plotting function with the same signature as a 2d xray
            plotting method such as `xray.plot.imshow`
        x, y : string
            Names of the coordinates to plot on x, y axes
        max_xticks, max_yticks : int, optional
            Maximum number of labeled ticks to plot on x, y axes
        max_yticks : int
            Maximum number of tick marks to place on y axis
        fontsize : string or int
            Font size as used by matplotlib text
        kwargs :
            additional keyword arguments to plotfunc

        Returns
        -------
        self : FacetGrid object

        '''
        import matplotlib.pyplot as plt

        # These should be consistent with xray.plot._plot2d
        cmap_kwargs = {'plot_data': self.darray.values,
                'vmin': None,
                'vmax': None,
                'cmap': None,
                'center': None,
                'robust': False,
                'extend': None,
                'levels': 7 if 'contour' in plotfunc.__name__ else None, # MPL default
                'filled': plotfunc.__name__ != 'contour',
                }

        # Allow kwargs to override these defaults
        for param in kwargs:
            if param in cmap_kwargs:
                cmap_kwargs[param] = kwargs[param]

        # colormap inference has to happen here since all the data in self.darray
        # is required to make the right choice
        cmap_params = _determine_cmap_params(**cmap_kwargs)

        if 'contour' in plotfunc.__name__:
            # extend is a keyword argument only for contour and contourf, but
            # passing it to the colorbar is sufficient for imshow and
            # pcolormesh
            kwargs['extend'] = cmap_params['extend']
            kwargs['levels'] = cmap_params['levels']

        defaults = {
                'add_colorbar': False,
                'add_labels': False,
                'norm': cmap_params.pop('cnorm'),
                }

        # Order is important
        defaults.update(cmap_params)
        defaults.update(kwargs)

        for d, ax in zip(self.name_dicts.flat, self.axes.flat):
            # None is the sentinel value
            if d is not None:
                subset = self.darray.loc[d]
                mappable = plotfunc(subset, x, y, ax=ax, **defaults)

        self.x = x
        self.y = y

        # Left side labels
        for ax in self.axes[:, 0]:
            ax.set_ylabel(self.y)

        # Bottom labels
        for ax in self.axes[-1, :]:
            ax.set_xlabel(self.x)

        self.fig.tight_layout()

        if self._single_group:
            for d, ax in zip(self.name_dicts.flat, self.axes.flat):
                if d is None:
                    ax.set_visible(False)

        # colorbar
        if kwargs.get('add_colorbar', True):
            cbar = self.fig.colorbar(mappable,
                                     ax=list(self.axes.flat),
                                     extend=cmap_params['extend'])

            if self.darray.name:
                cbar.set_label(self.darray.name, rotation=270,
                        verticalalignment='bottom')

        # This happens here rather than __init__ since FacetGrid.map should
        # use default ticks
        self.set_ticks(max_xticks, max_yticks, fontsize)

        return self

    def set_titles(self, template="{coord} = {value}", maxchar=30,
            fontsize=_FONTSIZE, **kwargs):
        '''
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for plot titles containing {coord} and {value}
        maxchar : int
            Truncate titles at maxchar
        fontsize : string or int
            Passed to matplotlib.text
        kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: FacetGrid object

        '''

        import matplotlib as mpl

        kwargs['fontsize'] = fontsize

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
            if self._margin_titles:
                for ax, row_name in zip(self.axes[:, -1], self.row_names):
                    title = nicetitle(coord=self.row, value=row_name, maxchar=maxchar)
                    ax.annotate(title, xy=(1.02, .5), xycoords="axes fraction",
                                rotation=270, ha="left", va="center", **kwargs)

            # The column titles on the top row
            for ax, col_name in zip(self.axes[0, :], self.col_names):
                title = nicetitle(coord=self.col, value=col_name, maxchar=maxchar)
                ax.set_title(title, **kwargs)

        return self

    def set_ticks(self, max_xticks=_NTICKS, max_yticks=_NTICKS, fontsize=_FONTSIZE):
        '''
        Sets tick behavior.

        Refer to documentation in :meth:`FacetGrid.map_dataarray`
        '''
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

    def map(self, func, *args, **kwargs):
        '''Apply a plotting function to each facet's subset of the data.

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

        '''
        import matplotlib.pyplot as plt

        for ax, namedict in zip(self, self.name_dicts.flat):
            if namedict is not None:
                data = self.darray[namedict]
                plt.sca(ax)
                innerargs = [data[a].values for a in args]
                func(*innerargs, **kwargs)

        return self
