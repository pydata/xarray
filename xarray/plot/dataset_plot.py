from __future__ import absolute_import, division, print_function

import functools
import warnings

import numpy as np
import pandas as pd

from ..core.common import ones_like
from .facetgrid import FacetGrid
from .utils import (
    ROBUST_PERCENTILE, _determine_cmap_params, _ensure_numeric,
    _interval_to_double_bound_points, _interval_to_mid_points,
    _valid_other_type, get_axis, import_matplotlib_pyplot, label_from_attrs)


def _infer_scatter_meta_data(ds, x, y, hue, add_legend, add_colorbar):
    dvars = set(ds.data_vars.keys())
    error_msg = (' must be either one of ({0:s})'
                 .format(', '.join(dvars)))

    if x not in dvars:
        raise ValueError(x + error_msg)

    if y not in dvars:
        raise ValueError(y + error_msg)

    if hue:
        if add_legend is None and add_colorbar is None:
            if not _ensure_numeric(ds[hue].values):
                add_legend = True
                add_colorbar = False
            else:
                add_legend = False
                add_colorbar = True

        if add_colorbar is None:
            if add_legend is True:
                add_colorbar = False
            else:
                if _ensure_numeric(ds[hue].values):
                    add_colorbar = True
                else:
                    add_colorbar = False

        elif add_legend is None:
            if add_colorbar is True:
                add_legend = False
            else:
                add_legend = True

        elif add_colorbar is True and not _ensure_numeric(ds[hue].values):
            raise ValueError('Cannot create a colorbar for a non numeric'
                             ' coordinate')

    elif add_legend or add_colorbar:
        raise ValueError('hue must be specified for generating a legend'
                         ' or colorbar')

    dims = ds[x].dims
    if ds[y].dims != dims:
        raise ValueError('{} and {} must have the same dimensions.'
                         ''.format(x, y))

    dims_coords = set(list(ds.coords) + list(ds.dims))
    if hue is not None and hue not in dims_coords:
        raise ValueError(hue + ' must be either one of ({0:s})'
                               ''.format(', '.join(dims_coords)))

    if hue:
        hue_label = label_from_attrs(ds.coords[hue])
    else:
        hue_label = None

    return {'add_legend': add_legend,
            'add_colorbar': add_colorbar,
            'hue_label': hue_label,
            'xlabel': label_from_attrs(ds[x]),
            'ylabel': label_from_attrs(ds[y]),
            'hue_values': ds[x].coords[hue] if add_legend else None}


def _infer_scatter_data(ds, x, y, hue, add_legend):
    dims = set(ds[x].dims)
    if add_legend:
        dims.remove(hue)
        xplt = ds[x].stack(stackdim=dims).transpose('stackdim', hue).values
        yplt = ds[y].stack(stackdim=dims).transpose('stackdim', hue).values
        return {'x': xplt, 'y': yplt}
    else:
        data = {'x': ds[x].values.flatten(),
                'y': ds[y].values.flatten(),
                'color': None}
        if hue:
            data['color'] = ((ones_like(ds[x]) * ds.coords[hue])
                             .values.flatten())
        return data


def scatter(ds, x, y, hue=None, col=None, row=None,
            col_wrap=None, sharex=True, sharey=True, aspect=None,
            size=None, subplot_kws=None, add_legend=None,
            add_colorbar=None, **kwargs):

    if kwargs.get('_meta_data', None):
        add_colorbar = kwargs['_meta_data']['add_colorbar']
    else:
        meta_data = _infer_scatter_meta_data(ds, x, y, hue,
                                             add_legend, add_colorbar)
        add_colorbar = meta_data['add_colorbar']
        add_legend = meta_data['add_legend']

    if col or row:
        ax = kwargs.pop('ax', None)
        figsize = kwargs.pop('figsize', None)
        if ax is not None:
            raise ValueError("Can't use axes when making faceted plots.")
        if aspect is None:
            aspect = 1
        if size is None:
            size = 3
        elif figsize is not None:
            raise ValueError('Cannot provide both `figsize` and '
                             '`size` arguments')

        g = FacetGrid(data=ds, col=col, row=row, col_wrap=col_wrap,
                      sharex=sharex, sharey=sharey, figsize=figsize,
                      aspect=aspect, size=size, subplot_kws=subplot_kws)
        return g.map_scatter(x=x, y=y, hue=hue, add_legend=add_legend,
                             add_colorbar=add_colorbar, **kwargs)

    data = _infer_scatter_data(ds, x, y, hue, add_legend)

    figsize = kwargs.pop('figsize', None)
    ax = kwargs.pop('ax', None)
    ax = get_axis(figsize, size, aspect, ax)
    if add_legend:
        primitive = ax.plot(data['x'], data['y'], '.')
    else:
        primitive = ax.scatter(data['x'], data['y'], c=data['color'])
    if '_meta_data' in kwargs:  # if this was called from map_scatter,
        return primitive        # finish here. Else, make labels

    if meta_data.get('xlabel', None):
        ax.set_xlabel(meta_data.get('xlabel'))

    if meta_data.get('ylabel', None):
        ax.set_ylabel(meta_data.get('ylabel'))
    if add_legend:
        ax.legend(handles=primitive,
                  labels=list(meta_data['hue_values'].values),
                  title=meta_data.get('hue_label', None))
    if add_colorbar:
        cbar = ax.figure.colorbar(primitive)
        if meta_data.get('hue_label', None):
            cbar.ax.set_ylabel(meta_data.get('hue_label'))

    return primitive


class _Dataset_PlotMethods(object):
    """
    Enables use of xarray.plot functions as attributes on a Dataset.
    For example, Dataset.plot.scatter
    """

    def __init__(self, dataset):
        self._ds = dataset

    def __call__(self, *args, **kwargs):
        raise ValueError('Dataset.plot cannot be called directly. Use'
                         'an explicit plot method, e.g. ds.plot.scatter(...)')

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        return scatter(self._ds, *args, **kwargs)
