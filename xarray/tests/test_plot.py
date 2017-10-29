from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

import inspect

import numpy as np
import pandas as pd
from datetime import datetime
import pytest

from xarray import DataArray

import xarray.plot as xplt
from xarray.plot.plot import _infer_interval_breaks
from xarray.plot.utils import (_determine_cmap_params,
                               _build_discrete_cmap,
                               _color_palette, import_seaborn)

from . import TestCase, requires_matplotlib, requires_seaborn, raises_regex


@pytest.mark.flaky
@pytest.mark.skip(reason='maybe flaky')
def text_in_fig():
    '''
    Return the set of all text in the figure
    '''
    return {t.get_text() for t in plt.gcf().findobj(mpl.text.Text)}


def find_possible_colorbars():
    # nb. this function also matches meshes from pcolormesh
    return plt.gcf().findobj(mpl.collections.QuadMesh)


def substring_in_axes(substring, ax):
    '''
    Return True if a substring is found anywhere in an axes
    '''
    alltxt = set([t.get_text() for t in ax.findobj(mpl.text.Text)])
    for txt in alltxt:
        if substring in txt:
            return True
    return False


def easy_array(shape, start=0, stop=1):
    '''
    Make an array with desired shape using np.linspace

    shape is a tuple like (2, 3)
    '''
    a = np.linspace(start, stop, num=np.prod(shape))
    return a.reshape(shape)


@requires_matplotlib
class PlotTestCase(TestCase):

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')

    def pass_in_axis(self, plotmethod):
        fig, axes = plt.subplots(ncols=2)
        plotmethod(ax=axes[0])
        self.assertTrue(axes[0].has_data())

    @pytest.mark.slow
    def imshow_called(self, plotmethod):
        plotmethod()
        images = plt.gca().findobj(mpl.image.AxesImage)
        return len(images) > 0

    def contourf_called(self, plotmethod):
        plotmethod()
        paths = plt.gca().findobj(mpl.collections.PathCollection)
        return len(paths) > 0


class TestPlot(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test1d(self):
        self.darray[:, 0, 0].plot()

    def test_2d_before_squeeze(self):
        a = DataArray(easy_array((1, 5)))
        a.plot()

    def test2d_uniform_calls_imshow(self):
        self.assertTrue(self.imshow_called(self.darray[:, :, 0].plot.imshow))

    @pytest.mark.slow
    def test2d_nonuniform_calls_contourf(self):
        a = self.darray[:, :, 0]
        a.coords['dim_1'] = [2, 1, 89]
        self.assertTrue(self.contourf_called(a.plot.contourf))

    def test3d(self):
        self.darray.plot()

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot)

    def test__infer_interval_breaks(self):
        self.assertArrayEqual([-0.5, 0.5, 1.5], _infer_interval_breaks([0, 1]))
        self.assertArrayEqual([-0.5, 0.5, 5.0, 9.5, 10.5],
                              _infer_interval_breaks([0, 1, 9, 10]))
        self.assertArrayEqual(
            pd.date_range('20000101', periods=4) - np.timedelta64(12, 'h'),
            _infer_interval_breaks(pd.date_range('20000101', periods=3)))

        # make a bounded 2D array that we will center and re-infer
        xref, yref = np.meshgrid(np.arange(6), np.arange(5))
        cx = (xref[1:, 1:] + xref[:-1, :-1]) / 2
        cy = (yref[1:, 1:] + yref[:-1, :-1]) / 2
        x = _infer_interval_breaks(cx, axis=1)
        x = _infer_interval_breaks(x, axis=0)
        y = _infer_interval_breaks(cy, axis=1)
        y = _infer_interval_breaks(y, axis=0)
        np.testing.assert_allclose(xref, x)
        np.testing.assert_allclose(yref, y)

    def test_datetime_dimension(self):
        nrow = 3
        ncol = 4
        time = pd.date_range('2000-01-01', periods=nrow)
        a = DataArray(easy_array((nrow, ncol)),
                      coords=[('time', time), ('y', range(ncol))])
        a.plot()
        ax = plt.gca()
        self.assertTrue(ax.has_data())

    @pytest.mark.slow
    def test_convenient_facetgrid(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        d.coords['z'] = list('abcd')
        g = d.plot(x='x', y='y', col='z', col_wrap=2, cmap='cool')

        self.assertArrayEqual(g.axes.shape, [2, 2])
        for ax in g.axes.flat:
            self.assertTrue(ax.has_data())

        with raises_regex(ValueError, '[Ff]acet'):
            d.plot(x='x', y='y', col='z', ax=plt.gca())

        with raises_regex(ValueError, '[Ff]acet'):
            d[0].plot(x='x', y='y', col='z', ax=plt.gca())

    @pytest.mark.slow
    def test_subplot_kws(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        d.coords['z'] = list('abcd')
        g = d.plot(x='x', y='y', col='z', col_wrap=2, cmap='cool',
                   subplot_kws=dict(facecolor='r'))
        for ax in g.axes.flat:
            try:
                # mpl V2
                self.assertEqual(ax.get_facecolor()[0:3],
                                 mpl.colors.to_rgb('r'))
            except AttributeError:
                self.assertEqual(ax.get_axis_bgcolor(), 'r')

    @pytest.mark.slow
    def test_plot_size(self):
        self.darray[:, 0, 0].plot(figsize=(13, 5))
        assert tuple(plt.gcf().get_size_inches()) == (13, 5)

        self.darray.plot(figsize=(13, 5))
        assert tuple(plt.gcf().get_size_inches()) == (13, 5)

        self.darray.plot(size=5)
        assert plt.gcf().get_size_inches()[1] == 5

        self.darray.plot(size=5, aspect=2)
        assert tuple(plt.gcf().get_size_inches()) == (10, 5)

        with raises_regex(ValueError, 'cannot provide both'):
            self.darray.plot(ax=plt.gca(), figsize=(3, 4))

        with raises_regex(ValueError, 'cannot provide both'):
            self.darray.plot(size=5, figsize=(3, 4))

        with raises_regex(ValueError, 'cannot provide both'):
            self.darray.plot(size=5, ax=plt.gca())

        with raises_regex(ValueError, 'cannot provide `aspect`'):
            self.darray.plot(aspect=1)

    @pytest.mark.slow
    def test_convenient_facetgrid_4d(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = d.plot(x='x', y='y', col='columns', row='rows')

        self.assertArrayEqual(g.axes.shape, [3, 2])
        for ax in g.axes.flat:
            self.assertTrue(ax.has_data())

        with raises_regex(ValueError, '[Ff]acet'):
            d.plot(x='x', y='y', col='columns', ax=plt.gca())


class TestPlot1D(PlotTestCase):

    def setUp(self):
        d = [0, 1.1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))},
                                dims='period')

    def test_xlabel_is_index_name(self):
        self.darray.plot()
        self.assertEqual('period', plt.gca().get_xlabel())

    def test_no_label_name_on_y_axis(self):
        self.darray.plot()
        self.assertEqual('', plt.gca().get_ylabel())

    def test_ylabel_is_data_name(self):
        self.darray.name = 'temperature'
        self.darray.plot()
        self.assertEqual(self.darray.name, plt.gca().get_ylabel())

    def test_wrong_dims_raises_valueerror(self):
        twodims = DataArray(easy_array((2, 5)))
        with pytest.raises(ValueError):
            twodims.plot.line()

    def test_format_string(self):
        self.darray.plot.line('ro')

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot.line)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray([1, 2, 3], {'letter': ['a', 'b', 'c']},
                      dims='letter')
        with raises_regex(TypeError, r'[Pp]lot'):
            a.plot.line()

    def test_primitive_returned(self):
        p = self.darray.plot.line()
        self.assertTrue(isinstance(p[0], mpl.lines.Line2D))

    @pytest.mark.slow
    def test_plot_nans(self):
        self.darray[1] = np.nan
        self.darray.plot.line()

    def test_x_ticks_are_rotated_for_time(self):
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.arange(len(time)), [('t', time)])
        a.plot.line()
        rotation = plt.gca().get_xticklabels()[0].get_rotation()
        self.assertFalse(rotation == 0)

    def test_slice_in_title(self):
        self.darray.coords['d'] = 10
        self.darray.plot.line()
        title = plt.gca().get_title()
        self.assertEqual('d = 10', title)


class TestPlotHistogram(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_3d_array(self):
        self.darray.plot.hist()

    def test_title_no_name(self):
        self.darray.plot.hist()
        self.assertEqual('', plt.gca().get_title())

    def test_title_uses_name(self):
        self.darray.name = 'testpoints'
        self.darray.plot.hist()
        self.assertIn(self.darray.name, plt.gca().get_title())

    def test_ylabel_is_count(self):
        self.darray.plot.hist()
        self.assertEqual('Count', plt.gca().get_ylabel())

    def test_can_pass_in_kwargs(self):
        nbins = 5
        self.darray.plot.hist(bins=nbins)
        self.assertEqual(nbins, len(plt.gca().patches))

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot.hist)

    def test_primitive_returned(self):
        h = self.darray.plot.hist()
        self.assertTrue(isinstance(h[-1][0], mpl.patches.Rectangle))

    @pytest.mark.slow
    def test_plot_nans(self):
        self.darray[0, 0, 0] = np.nan
        self.darray.plot.hist()


@requires_matplotlib
class TestDetermineCmapParams(TestCase):

    def setUp(self):
        self.data = np.linspace(0, 1, num=100)

    def test_robust(self):
        cmap_params = _determine_cmap_params(self.data, robust=True)
        self.assertEqual(cmap_params['vmin'], np.percentile(self.data, 2))
        self.assertEqual(cmap_params['vmax'], np.percentile(self.data, 98))
        self.assertEqual(cmap_params['cmap'].name, 'viridis')
        self.assertEqual(cmap_params['extend'], 'both')
        self.assertIsNone(cmap_params['levels'])
        self.assertIsNone(cmap_params['norm'])

    def test_center(self):
        cmap_params = _determine_cmap_params(self.data, center=0.5)
        self.assertEqual(cmap_params['vmax'] - 0.5, 0.5 - cmap_params['vmin'])
        self.assertEqual(cmap_params['cmap'], 'RdBu_r')
        self.assertEqual(cmap_params['extend'], 'neither')
        self.assertIsNone(cmap_params['levels'])
        self.assertIsNone(cmap_params['norm'])

    @pytest.mark.slow
    def test_integer_levels(self):
        data = self.data + 1

        # default is to cover full data range but with no guarantee on Nlevels
        for level in np.arange(2, 10, dtype=int):
            cmap_params = _determine_cmap_params(data, levels=level)
            self.assertEqual(cmap_params['vmin'], cmap_params['levels'][0])
            self.assertEqual(cmap_params['vmax'], cmap_params['levels'][-1])
            self.assertEqual(cmap_params['extend'], 'neither')

        # with min max we are more strict
        cmap_params = _determine_cmap_params(data, levels=5, vmin=0, vmax=5,
                                             cmap='Blues')
        self.assertEqual(cmap_params['vmin'], 0)
        self.assertEqual(cmap_params['vmax'], 5)
        self.assertEqual(cmap_params['vmin'], cmap_params['levels'][0])
        self.assertEqual(cmap_params['vmax'], cmap_params['levels'][-1])
        self.assertEqual(cmap_params['cmap'].name, 'Blues')
        self.assertEqual(cmap_params['extend'], 'neither')
        self.assertEqual(cmap_params['cmap'].N, 4)
        self.assertEqual(cmap_params['norm'].N, 5)

        cmap_params = _determine_cmap_params(data, levels=5,
                                             vmin=0.5, vmax=1.5)
        self.assertEqual(cmap_params['cmap'].name, 'viridis')
        self.assertEqual(cmap_params['extend'], 'max')

        cmap_params = _determine_cmap_params(data, levels=5,
                                             vmin=1.5)
        self.assertEqual(cmap_params['cmap'].name, 'viridis')
        self.assertEqual(cmap_params['extend'], 'min')

        cmap_params = _determine_cmap_params(data, levels=5,
                                             vmin=1.3, vmax=1.5)
        self.assertEqual(cmap_params['cmap'].name, 'viridis')
        self.assertEqual(cmap_params['extend'], 'both')

    def test_list_levels(self):
        data = self.data + 1

        orig_levels = [0, 1, 2, 3, 4, 5]
        # vmin and vmax should be ignored if levels are explicitly provided
        cmap_params = _determine_cmap_params(data, levels=orig_levels,
                                             vmin=0, vmax=3)
        self.assertEqual(cmap_params['vmin'], 0)
        self.assertEqual(cmap_params['vmax'], 5)
        self.assertEqual(cmap_params['cmap'].N, 5)
        self.assertEqual(cmap_params['norm'].N, 6)

        for wrap_levels in [list, np.array, pd.Index, DataArray]:
            cmap_params = _determine_cmap_params(
                data, levels=wrap_levels(orig_levels))
            self.assertArrayEqual(cmap_params['levels'], orig_levels)

    def test_divergentcontrol(self):
        neg = self.data - 0.1
        pos = self.data

        # Default with positive data will be a normal cmap
        cmap_params = _determine_cmap_params(pos)
        self.assertEqual(cmap_params['vmin'], 0)
        self.assertEqual(cmap_params['vmax'], 1)
        self.assertEqual(cmap_params['cmap'].name, "viridis")

        # Default with negative data will be a divergent cmap
        cmap_params = _determine_cmap_params(neg)
        self.assertEqual(cmap_params['vmin'], -0.9)
        self.assertEqual(cmap_params['vmax'], 0.9)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")

        # Setting vmin or vmax should prevent this only if center is false
        cmap_params = _determine_cmap_params(neg, vmin=-0.1, center=False)
        self.assertEqual(cmap_params['vmin'], -0.1)
        self.assertEqual(cmap_params['vmax'], 0.9)
        self.assertEqual(cmap_params['cmap'].name, "viridis")
        cmap_params = _determine_cmap_params(neg, vmax=0.5, center=False)
        self.assertEqual(cmap_params['vmin'], -0.1)
        self.assertEqual(cmap_params['vmax'], 0.5)
        self.assertEqual(cmap_params['cmap'].name, "viridis")

        # Setting center=False too
        cmap_params = _determine_cmap_params(neg, center=False)
        self.assertEqual(cmap_params['vmin'], -0.1)
        self.assertEqual(cmap_params['vmax'], 0.9)
        self.assertEqual(cmap_params['cmap'].name, "viridis")

        # However, I should still be able to set center and have a div cmap
        cmap_params = _determine_cmap_params(neg, center=0)
        self.assertEqual(cmap_params['vmin'], -0.9)
        self.assertEqual(cmap_params['vmax'], 0.9)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")

        # Setting vmin or vmax alone will force symmetric bounds around center
        cmap_params = _determine_cmap_params(neg, vmin=-0.1)
        self.assertEqual(cmap_params['vmin'], -0.1)
        self.assertEqual(cmap_params['vmax'], 0.1)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")
        cmap_params = _determine_cmap_params(neg, vmax=0.5)
        self.assertEqual(cmap_params['vmin'], -0.5)
        self.assertEqual(cmap_params['vmax'], 0.5)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")
        cmap_params = _determine_cmap_params(neg, vmax=0.6, center=0.1)
        self.assertEqual(cmap_params['vmin'], -0.4)
        self.assertEqual(cmap_params['vmax'], 0.6)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")

        # But this is only true if vmin or vmax are negative
        cmap_params = _determine_cmap_params(pos, vmin=-0.1)
        self.assertEqual(cmap_params['vmin'], -0.1)
        self.assertEqual(cmap_params['vmax'], 0.1)
        self.assertEqual(cmap_params['cmap'], "RdBu_r")
        cmap_params = _determine_cmap_params(pos, vmin=0.1)
        self.assertEqual(cmap_params['vmin'], 0.1)
        self.assertEqual(cmap_params['vmax'], 1)
        self.assertEqual(cmap_params['cmap'].name, "viridis")
        cmap_params = _determine_cmap_params(pos, vmax=0.5)
        self.assertEqual(cmap_params['vmin'], 0)
        self.assertEqual(cmap_params['vmax'], 0.5)
        self.assertEqual(cmap_params['cmap'].name, "viridis")

        # If both vmin and vmax are provided, output is non-divergent
        cmap_params = _determine_cmap_params(neg, vmin=-0.2, vmax=0.6)
        self.assertEqual(cmap_params['vmin'], -0.2)
        self.assertEqual(cmap_params['vmax'], 0.6)
        self.assertEqual(cmap_params['cmap'].name, "viridis")


@requires_matplotlib
class TestDiscreteColorMap(TestCase):

    def setUp(self):
        x = np.arange(start=0, stop=10, step=2)
        y = np.arange(start=9, stop=-7, step=-3)
        xy = np.dstack(np.meshgrid(x, y))
        distance = np.linalg.norm(xy, axis=2)
        self.darray = DataArray(distance, list(zip(('y', 'x'), (y, x))))
        self.data_min = distance.min()
        self.data_max = distance.max()

    @pytest.mark.slow
    def test_recover_from_seaborn_jet_exception(self):
        pal = _color_palette('jet', 4)
        self.assertTrue(type(pal) == np.ndarray)
        self.assertEqual(len(pal), 4)

    @pytest.mark.slow
    def test_build_discrete_cmap(self):
        for (cmap, levels, extend, filled) in [('jet', [0, 1], 'both', False),
                                               ('hot', [-4, 4], 'max', True)]:
            ncmap, cnorm = _build_discrete_cmap(cmap, levels, extend, filled)
            self.assertEqual(ncmap.N, len(levels) - 1)
            self.assertEqual(len(ncmap.colors), len(levels) - 1)
            self.assertEqual(cnorm.N, len(levels))
            self.assertArrayEqual(cnorm.boundaries, levels)
            self.assertEqual(max(levels), cnorm.vmax)
            self.assertEqual(min(levels), cnorm.vmin)
            if filled:
                self.assertEqual(ncmap.colorbar_extend, extend)
            else:
                self.assertEqual(ncmap.colorbar_extend, 'max')

    @pytest.mark.slow
    def test_discrete_colormap_list_of_levels(self):
        for extend, levels in [('max', [-1, 2, 4, 8, 10]),
                               ('both', [2, 5, 10, 11]),
                               ('neither', [0, 5, 10, 15]),
                               ('min', [2, 5, 10, 15])]:
            for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
                primitive = getattr(self.darray.plot, kind)(levels=levels)
                self.assertArrayEqual(levels, primitive.norm.boundaries)
                self.assertEqual(max(levels), primitive.norm.vmax)
                self.assertEqual(min(levels), primitive.norm.vmin)
                if kind != 'contour':
                    self.assertEqual(extend, primitive.cmap.colorbar_extend)
                else:
                    self.assertEqual('max', primitive.cmap.colorbar_extend)
                self.assertEqual(len(levels) - 1, len(primitive.cmap.colors))

    @pytest.mark.slow
    def test_discrete_colormap_int_levels(self):
        for extend, levels, vmin, vmax in [('neither', 7, None, None),
                                           ('neither', 7, None, 20),
                                           ('both', 7, 4, 8),
                                           ('min', 10, 4, 15)]:
            for kind in ['imshow', 'pcolormesh', 'contourf', 'contour']:
                primitive = getattr(self.darray.plot, kind)(levels=levels,
                                                            vmin=vmin,
                                                            vmax=vmax)
                self.assertGreaterEqual(levels,
                                        len(primitive.norm.boundaries) - 1)
                if vmax is None:
                    self.assertGreaterEqual(primitive.norm.vmax, self.data_max)
                else:
                    self.assertGreaterEqual(primitive.norm.vmax, vmax)
                if vmin is None:
                    self.assertLessEqual(primitive.norm.vmin, self.data_min)
                else:
                    self.assertLessEqual(primitive.norm.vmin, vmin)
                if kind != 'contour':
                    self.assertEqual(extend, primitive.cmap.colorbar_extend)
                else:
                    self.assertEqual('max', primitive.cmap.colorbar_extend)
                self.assertGreaterEqual(levels, len(primitive.cmap.colors))

    def test_discrete_colormap_list_levels_and_vmin_or_vmax(self):
        levels = [0, 5, 10, 15]
        primitive = self.darray.plot(levels=levels, vmin=-3, vmax=20)
        self.assertEqual(primitive.norm.vmax, max(levels))
        self.assertEqual(primitive.norm.vmin, min(levels))


class Common2dMixin:
    """
    Common tests for 2d plotting go here.

    These tests assume that a staticmethod for `self.plotfunc` exists.
    Should have the same name as the method.
    """

    def setUp(self):
        da = DataArray(easy_array(
            (10, 15), start=-1), dims=['y', 'x'])
        # add 2d coords
        ds = da.to_dataset(name='testvar')
        x, y = np.meshgrid(da.x.values, da.y.values)
        ds['x2d'] = DataArray(x, dims=['y', 'x'])
        ds['y2d'] = DataArray(y, dims=['y', 'x'])
        ds.set_coords(['x2d', 'y2d'], inplace=True)
        # set darray and plot method
        self.darray = ds.testvar
        self.plotmethod = getattr(self.darray.plot, self.plotfunc.__name__)

    def test_label_names(self):
        self.plotmethod()
        self.assertEqual('x', plt.gca().get_xlabel())
        self.assertEqual('y', plt.gca().get_ylabel())

    def test_1d_raises_valueerror(self):
        with raises_regex(ValueError, r'DataArray must be 2d'):
            self.plotfunc(self.darray[0, :])

    def test_3d_raises_valueerror(self):
        a = DataArray(easy_array((2, 3, 4)))
        with raises_regex(ValueError, r'DataArray must be 2d'):
            self.plotfunc(a)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray(easy_array((3, 2)),
                      coords=[['a', 'b', 'c'], ['d', 'e']])
        with raises_regex(TypeError, r'[Pp]lot'):
            self.plotfunc(a)

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.plotmethod)

    def test_xyincrease_false_changes_axes(self):
        self.plotmethod(xincrease=False, yincrease=False)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = xlim[0] - 14, xlim[1] - 0, ylim[0] - 9, ylim[1] - 0
        self.assertTrue(all(abs(x) < 1 for x in diffs))

    def test_xyincrease_true_changes_axes(self):
        self.plotmethod(xincrease=True, yincrease=True)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = xlim[0] - 0, xlim[1] - 14, ylim[0] - 0, ylim[1] - 9
        self.assertTrue(all(abs(x) < 1 for x in diffs))

    def test_plot_nans(self):
        x1 = self.darray[:5]
        x2 = self.darray.copy()
        x2[5:] = np.nan

        clim1 = self.plotfunc(x1).get_clim()
        clim2 = self.plotfunc(x2).get_clim()
        self.assertEqual(clim1, clim2)

    def test_viridis_cmap(self):
        cmap_name = self.plotmethod(cmap='viridis').get_cmap().name
        self.assertEqual('viridis', cmap_name)

    def test_default_cmap(self):
        cmap_name = self.plotmethod().get_cmap().name
        self.assertEqual('RdBu_r', cmap_name)

        cmap_name = self.plotfunc(abs(self.darray)).get_cmap().name
        self.assertEqual('viridis', cmap_name)

    @requires_seaborn
    def test_seaborn_palette_as_cmap(self):
        cmap_name = self.plotmethod(
            levels=2, cmap='husl').get_cmap().name
        self.assertEqual('husl', cmap_name)

    def test_can_change_default_cmap(self):
        cmap_name = self.plotmethod(cmap='Blues').get_cmap().name
        self.assertEqual('Blues', cmap_name)

    def test_diverging_color_limits(self):
        artist = self.plotmethod()
        vmin, vmax = artist.get_clim()
        self.assertAlmostEqual(-vmin, vmax)

    def test_xy_strings(self):
        self.plotmethod('y', 'x')
        ax = plt.gca()
        self.assertEqual('y', ax.get_xlabel())
        self.assertEqual('x', ax.get_ylabel())

    def test_positional_coord_string(self):
        self.plotmethod(y='x')
        ax = plt.gca()
        self.assertEqual('x', ax.get_ylabel())
        self.assertEqual('y', ax.get_xlabel())

        self.plotmethod(x='x')
        ax = plt.gca()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('y', ax.get_ylabel())

    def test_bad_x_string_exception(self):
        with raises_regex(
                ValueError, 'x and y must be coordinate variables'):
            self.plotmethod('not_a_real_dim', 'y')
        with raises_regex(
                ValueError, 'x must be a dimension name if y is not supplied'):
            self.plotmethod(x='not_a_real_dim')
        with raises_regex(
                ValueError, 'y must be a dimension name if x is not supplied'):
            self.plotmethod(y='not_a_real_dim')
        self.darray.coords['z'] = 100

    def test_coord_strings(self):
        # 1d coords (same as dims)
        self.assertEqual({'x', 'y'}, set(self.darray.dims))
        self.plotmethod(y='y', x='x')

    def test_non_linked_coords(self):
        # plot with coordinate names that are not dimensions
        self.darray.coords['newy'] = self.darray.y + 150
        # Normal case, without transpose
        self.plotfunc(self.darray, x='x', y='newy')
        ax = plt.gca()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('newy', ax.get_ylabel())
        # ax limits might change between plotfuncs
        # simply ensure that these high coords were passed over
        self.assertTrue(np.min(ax.get_ylim()) > 100.)

    def test_non_linked_coords_transpose(self):
        # plot with coordinate names that are not dimensions,
        # and with transposed y and x axes
        # This used to raise an error with pcolormesh and contour
        # https://github.com/pydata/xarray/issues/788
        self.darray.coords['newy'] = self.darray.y + 150
        self.plotfunc(self.darray, x='newy', y='x')
        ax = plt.gca()
        self.assertEqual('newy', ax.get_xlabel())
        self.assertEqual('x', ax.get_ylabel())
        # ax limits might change between plotfuncs
        # simply ensure that these high coords were passed over
        self.assertTrue(np.min(ax.get_xlim()) > 100.)

    def test_default_title(self):
        a = DataArray(easy_array((4, 3, 2)), dims=['a', 'b', 'c'])
        a.coords['c'] = [0, 1]
        a.coords['d'] = u'foo'
        self.plotfunc(a.isel(c=1))
        title = plt.gca().get_title()
        self.assertTrue('c = 1, d = foo' == title or 'd = foo, c = 1' == title)

    def test_colorbar_default_label(self):
        self.darray.name = 'testvar'
        self.plotmethod(add_colorbar=True)
        self.assertIn(self.darray.name, text_in_fig())

    def test_no_labels(self):
        self.darray.name = 'testvar'
        self.plotmethod(add_labels=False)
        alltxt = text_in_fig()
        for string in ['x', 'y', 'testvar']:
            self.assertNotIn(string, alltxt)

    def test_colorbar_kwargs(self):
        # replace label
        self.darray.name = 'testvar'
        self.plotmethod(add_colorbar=True, cbar_kwargs={'label':'MyLabel'})
        alltxt = text_in_fig()
        self.assertIn('MyLabel', alltxt)
        self.assertNotIn('testvar', alltxt)
        # you can use mapping types as well
        self.plotmethod(add_colorbar=True, cbar_kwargs=(('label', 'MyLabel'),))
        alltxt = text_in_fig()
        self.assertIn('MyLabel', alltxt)
        self.assertNotIn('testvar', alltxt)
        # change cbar ax
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(ax=ax, cbar_ax=cax, add_colorbar=True,
                        cbar_kwargs={'label':'MyBar'})
        self.assertTrue(ax.has_data())
        self.assertTrue(cax.has_data())
        alltxt = text_in_fig()
        self.assertIn('MyBar', alltxt)
        self.assertNotIn('testvar', alltxt)
        # note that there are two ways to achieve this
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(ax=ax, add_colorbar=True,
                        cbar_kwargs={'label':'MyBar', 'cax':cax})
        self.assertTrue(ax.has_data())
        self.assertTrue(cax.has_data())
        alltxt = text_in_fig()
        self.assertIn('MyBar', alltxt)
        self.assertNotIn('testvar', alltxt)
        # see that no colorbar is respected
        self.plotmethod(add_colorbar=False)
        self.assertNotIn('testvar', text_in_fig())
        # check that error is raised
        pytest.raises(ValueError, self.plotmethod,
                          add_colorbar=False, cbar_kwargs= {'label':'label'})

    def test_verbose_facetgrid(self):
        a = easy_array((10, 15, 3))
        d = DataArray(a, dims=['y', 'x', 'z'])
        g = xplt.FacetGrid(d, col='z')
        g.map_dataarray(self.plotfunc, 'x', 'y')
        for ax in g.axes.flat:
            self.assertTrue(ax.has_data())

    def test_2d_function_and_method_signature_same(self):
        func_sig = inspect.getcallargs(self.plotfunc, self.darray)
        method_sig = inspect.getcallargs(self.plotmethod)
        del method_sig['_PlotMethods_obj']
        del func_sig['darray']
        self.assertEqual(func_sig, method_sig)

    def test_convenient_facetgrid(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        g = self.plotfunc(d, x='x', y='y', col='z', col_wrap=2)

        self.assertArrayEqual(g.axes.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axes):
            self.assertTrue(ax.has_data())
            if x == 0:
                self.assertEqual('y', ax.get_ylabel())
            else:
                self.assertEqual('', ax.get_ylabel())
            if y == 1:
                self.assertEqual('x', ax.get_xlabel())
            else:
                self.assertEqual('', ax.get_xlabel())

        # Infering labels
        g = self.plotfunc(d, col='z', col_wrap=2)
        self.assertArrayEqual(g.axes.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axes):
            self.assertTrue(ax.has_data())
            if x == 0:
                self.assertEqual('y', ax.get_ylabel())
            else:
                self.assertEqual('', ax.get_ylabel())
            if y == 1:
                self.assertEqual('x', ax.get_xlabel())
            else:
                self.assertEqual('', ax.get_xlabel())

    def test_convenient_facetgrid_4d(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = self.plotfunc(d, x='x', y='y', col='columns', row='rows')

        self.assertArrayEqual(g.axes.shape, [3, 2])
        for ax in g.axes.flat:
            self.assertTrue(ax.has_data())

    def test_facetgrid_cmap(self):
        # Regression test for GH592
        data = (np.random.random(size=(20, 25, 12)) + np.linspace(-3, 3, 12))
        d = DataArray(data, dims=['x', 'y', 'time'])
        fg = d.plot.pcolormesh(col='time')
        # check that all color limits are the same
        self.assertEqual(len(set(m.get_clim() for m in fg._mappables)), 1)
        # check that all colormaps are the same
        self.assertEqual(len(set(m.get_cmap().name for m in fg._mappables)), 1)


@pytest.mark.slow
class TestContourf(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contourf)

    @pytest.mark.slow
    def test_contourf_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.contourf_called(self.darray.plot.imshow))
        self.assertTrue(self.contourf_called(self.darray.plot.contourf))

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.contour.QuadContourSet))

    @pytest.mark.slow
    def test_extend(self):
        artist = self.plotmethod()
        self.assertEqual(artist.extend, 'neither')

        self.darray[0, 0] = -100
        self.darray[-1, -1] = 100
        artist = self.plotmethod(robust=True)
        self.assertEqual(artist.extend, 'both')

        self.darray[0, 0] = 0
        self.darray[-1, -1] = 0
        artist = self.plotmethod(vmin=-0, vmax=10)
        self.assertEqual(artist.extend, 'min')

        artist = self.plotmethod(vmin=-10, vmax=0)
        self.assertEqual(artist.extend, 'max')

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x='x2d', y='y2d')
        # make sure labels came out ok
        ax = plt.gca()
        self.assertEqual('x2d', ax.get_xlabel())
        self.assertEqual('y2d', ax.get_ylabel())

    @pytest.mark.slow
    def test_levels(self):
        artist = self.plotmethod(levels=[-0.5, -0.4, 0.1])
        self.assertEqual(artist.extend, 'both')

        artist = self.plotmethod(levels=3)
        self.assertEqual(artist.extend, 'neither')


@pytest.mark.slow
class TestContour(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contour)

    def test_colors(self):
        # matplotlib cmap.colors gives an rgbA ndarray
        # when seaborn is used, instead we get an rgb tuple
        def _color_as_tuple(c):
            return tuple(c[:3])
        artist = self.plotmethod(colors='k')
        self.assertEqual(
            _color_as_tuple(artist.cmap.colors[0]),
            (0.0, 0.0, 0.0))

        artist = self.plotmethod(colors=['k', 'b'])
        self.assertEqual(
            _color_as_tuple(artist.cmap.colors[1]),
            (0.0, 0.0, 1.0))

        artist = self.darray.plot.contour(levels=[-0.5, 0., 0.5, 1.],
                                          colors=['k', 'r', 'w', 'b'])
        self.assertEqual(
            _color_as_tuple(artist.cmap.colors[1]),
            (1.0, 0.0, 0.0))
        self.assertEqual(
            _color_as_tuple(artist.cmap.colors[2]),
            (1.0, 1.0, 1.0))
        # the last color is now under "over"
        self.assertEqual(
            _color_as_tuple(artist.cmap._rgba_over),
            (0.0, 0.0, 1.0))

    def test_cmap_and_color_both(self):
        with pytest.raises(ValueError):
            self.plotmethod(colors='k', cmap='RdBu')

    def list_of_colors_in_cmap_deprecated(self):
        with pytest.raises(Exception):
            self.plotmethod(cmap=['k', 'b'])

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x='x2d', y='y2d')
        # make sure labels came out ok
        ax = plt.gca()
        self.assertEqual('x2d', ax.get_xlabel())
        self.assertEqual('y2d', ax.get_ylabel())

    def test_single_level(self):
        # this used to raise an error, but not anymore since
        # add_colorbar defaults to false
        self.plotmethod(levels=[0.1])
        self.plotmethod(levels=1)


class TestPcolormesh(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.pcolormesh)

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.collections.QuadMesh))

    def test_everything_plotted(self):
        artist = self.plotmethod()
        self.assertEqual(artist.get_array().size, self.darray.size)

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x='x2d', y='y2d')
        # make sure labels came out ok
        ax = plt.gca()
        self.assertEqual('x2d', ax.get_xlabel())
        self.assertEqual('y2d', ax.get_ylabel())

    def test_dont_infer_interval_breaks_for_cartopy(self):
        # Regression for GH 781
        ax = plt.gca()
        # Simulate a Cartopy Axis
        setattr(ax, 'projection', True)
        artist = self.plotmethod(x='x2d', y='y2d', ax=ax)
        self.assertTrue(isinstance(artist, mpl.collections.QuadMesh))
        # Let cartopy handle the axis limits and artist size
        self.assertTrue(artist.get_array().size <= self.darray.size)


@pytest.mark.slow
class TestImshow(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.imshow)

    @pytest.mark.slow
    def test_imshow_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.imshow_called(self.darray.plot.contourf))
        self.assertTrue(self.imshow_called(self.darray.plot.imshow))

    def test_xy_pixel_centered(self):
        self.darray.plot.imshow(yincrease=False)
        self.assertTrue(np.allclose([-0.5, 14.5], plt.gca().get_xlim()))
        self.assertTrue(np.allclose([9.5, -0.5], plt.gca().get_ylim()))

    def test_default_aspect_is_auto(self):
        self.darray.plot.imshow()
        self.assertEqual('auto', plt.gca().get_aspect())

    @pytest.mark.slow
    def test_cannot_change_mpl_aspect(self):

        with raises_regex(ValueError, 'not available in xarray'):
            self.darray.plot.imshow(aspect='equal')

        # with numbers we fall back to fig control
        self.darray.plot.imshow(size=5, aspect=2)
        self.assertEqual('auto', plt.gca().get_aspect())
        assert tuple(plt.gcf().get_size_inches()) == (10, 5)

    @pytest.mark.slow
    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.image.AxesImage))

    @pytest.mark.slow
    @requires_seaborn
    def test_seaborn_palette_needs_levels(self):
        with pytest.raises(ValueError):
            self.plotmethod(cmap='husl')

    def test_2d_coord_names(self):
        with raises_regex(ValueError, 'requires 1D coordinates'):
            self.plotmethod(x='x2d', y='y2d')


class TestFacetGrid(PlotTestCase):

    def setUp(self):
        d = easy_array((10, 15, 3))
        self.darray = DataArray(d, dims=['y', 'x', 'z'],
                                coords={'z': ['a', 'b', 'c']})
        self.g = xplt.FacetGrid(self.darray, col='z')

    @pytest.mark.slow
    def test_no_args(self):
        self.g.map_dataarray(xplt.contourf, 'x', 'y')

        # Don't want colorbar labeled with 'None'
        alltxt = text_in_fig()
        self.assertNotIn('None', alltxt)

        for ax in self.g.axes.flat:
            self.assertTrue(ax.has_data())

    @pytest.mark.slow
    def test_names_appear_somewhere(self):
        self.darray.name = 'testvar'
        self.g.map_dataarray(xplt.contourf, 'x', 'y')
        for k, ax in zip('abc', self.g.axes.flat):
            self.assertEqual('z = {0}'.format(k), ax.get_title())

        alltxt = text_in_fig()
        self.assertIn(self.darray.name, alltxt)
        for label in ['x', 'y']:
            self.assertIn(label, alltxt)

    @pytest.mark.slow
    def test_text_not_super_long(self):
        self.darray.coords['z'] = [100 * letter for letter in 'abc']
        g = xplt.FacetGrid(self.darray, col='z')
        g.map_dataarray(xplt.contour, 'x', 'y')
        alltxt = text_in_fig()
        maxlen = max(len(txt) for txt in alltxt)
        self.assertLess(maxlen, 50)

        t0 = g.axes[0, 0].get_title()
        self.assertTrue(t0.endswith('...'))

    @pytest.mark.slow
    def test_colorbar(self):
        vmin = self.darray.values.min()
        vmax = self.darray.values.max()
        expected = np.array((vmin, vmax))

        self.g.map_dataarray(xplt.imshow, 'x', 'y')

        for image in plt.gcf().findobj(mpl.image.AxesImage):
            clim = np.array(image.get_clim())
            self.assertTrue(np.allclose(expected, clim))

        self.assertEqual(1, len(find_possible_colorbars()))

    @pytest.mark.slow
    def test_empty_cell(self):
        g = xplt.FacetGrid(self.darray, col='z', col_wrap=2)
        g.map_dataarray(xplt.imshow, 'x', 'y')

        bottomright = g.axes[-1, -1]
        self.assertFalse(bottomright.has_data())
        self.assertFalse(bottomright.get_visible())

    @pytest.mark.slow
    def test_norow_nocol_error(self):
        with raises_regex(ValueError, r'[Rr]ow'):
            xplt.FacetGrid(self.darray)

    @pytest.mark.slow
    def test_groups(self):
        self.g.map_dataarray(xplt.imshow, 'x', 'y')
        upperleft_dict = self.g.name_dicts[0, 0]
        upperleft_array = self.darray.loc[upperleft_dict]
        z0 = self.darray.isel(z=0)

        self.assertDataArrayEqual(upperleft_array, z0)

    @pytest.mark.slow
    def test_float_index(self):
        self.darray.coords['z'] = [0.1, 0.2, 0.4]
        g = xplt.FacetGrid(self.darray, col='z')
        g.map_dataarray(xplt.imshow, 'x', 'y')

    @pytest.mark.slow
    def test_nonunique_index_error(self):
        self.darray.coords['z'] = [0.1, 0.2, 0.2]
        with raises_regex(ValueError, r'[Uu]nique'):
            xplt.FacetGrid(self.darray, col='z')

    @pytest.mark.slow
    def test_robust(self):
        z = np.zeros((20, 20, 2))
        darray = DataArray(z, dims=['y', 'x', 'z'])
        darray[:, :, 1] = 1
        darray[2, 0, 0] = -1000
        darray[3, 0, 0] = 1000
        g = xplt.FacetGrid(darray, col='z')
        g.map_dataarray(xplt.imshow, 'x', 'y', robust=True)

        # Color limits should be 0, 1
        # The largest number displayed in the figure should be less than 21
        numbers = set()
        alltxt = text_in_fig()
        for txt in alltxt:
            try:
                numbers.add(float(txt))
            except ValueError:
                pass
        largest = max(abs(x) for x in numbers)
        self.assertLess(largest, 21)

    @pytest.mark.slow
    def test_can_set_vmin_vmax(self):
        vmin, vmax = 50.0, 1000.0
        expected = np.array((vmin, vmax))
        self.g.map_dataarray(xplt.imshow, 'x', 'y', vmin=vmin, vmax=vmax)

        for image in plt.gcf().findobj(mpl.image.AxesImage):
            clim = np.array(image.get_clim())
            self.assertTrue(np.allclose(expected, clim))

    @pytest.mark.slow
    def test_can_set_norm(self):
        norm = mpl.colors.SymLogNorm(0.1)
        self.g.map_dataarray(xplt.imshow, 'x', 'y', norm=norm)
        for image in plt.gcf().findobj(mpl.image.AxesImage):
            self.assertIs(image.norm, norm)

    @pytest.mark.slow
    def test_figure_size(self):

        self.assertArrayEqual(self.g.fig.get_size_inches(), (10, 3))

        g = xplt.FacetGrid(self.darray, col='z', size=6)
        self.assertArrayEqual(g.fig.get_size_inches(), (19, 6))

        g = self.darray.plot.imshow(col='z', size=6)
        self.assertArrayEqual(g.fig.get_size_inches(), (19, 6))

        g = xplt.FacetGrid(self.darray, col='z', size=4, aspect=0.5)
        self.assertArrayEqual(g.fig.get_size_inches(), (7, 4))

        g = xplt.FacetGrid(self.darray, col='z', figsize=(9, 4))
        self.assertArrayEqual(g.fig.get_size_inches(), (9, 4))

        with raises_regex(ValueError, "cannot provide both"):
            g = xplt.plot(self.darray, row=2, col='z', figsize=(6, 4), size=6)

        with raises_regex(ValueError, "Can't use"):
            g = xplt.plot(self.darray, row=2, col='z', ax=plt.gca(), size=6)

    @pytest.mark.slow
    def test_num_ticks(self):
        nticks = 99
        maxticks = nticks + 1
        self.g.map_dataarray(xplt.imshow, 'x', 'y')
        self.g.set_ticks(max_xticks=nticks, max_yticks=nticks)

        for ax in self.g.axes.flat:
            xticks = len(ax.get_xticks())
            yticks = len(ax.get_yticks())
            self.assertLessEqual(xticks, maxticks)
            self.assertLessEqual(yticks, maxticks)
            self.assertGreaterEqual(xticks, nticks / 2.0)
            self.assertGreaterEqual(yticks, nticks / 2.0)

    @pytest.mark.slow
    def test_map(self):
        self.g.map(plt.contourf, 'x', 'y', Ellipsis)
        self.g.map(lambda: None)

    @pytest.mark.slow
    def test_map_dataset(self):
        g = xplt.FacetGrid(self.darray.to_dataset(name='foo'), col='z')
        g.map(plt.contourf, 'x', 'y', 'foo')

        alltxt = text_in_fig()
        for label in ['x', 'y']:
            self.assertIn(label, alltxt)
        # everything has a label
        self.assertNotIn('None', alltxt)

        # colorbar can't be inferred automatically
        self.assertNotIn('foo', alltxt)
        self.assertEqual(0, len(find_possible_colorbars()))

        g.add_colorbar(label='colors!')
        self.assertIn('colors!', text_in_fig())
        self.assertEqual(1, len(find_possible_colorbars()))

    @pytest.mark.slow
    def test_set_axis_labels(self):
        g = self.g.map_dataarray(xplt.contourf, 'x', 'y')
        g.set_axis_labels('longitude', 'latitude')
        alltxt = text_in_fig()
        for label in ['longitude', 'latitude']:
            self.assertIn(label, alltxt)

    @pytest.mark.slow
    def test_facetgrid_colorbar(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'], name='foo')

        d.plot.imshow(x='x', y='y', col='z')
        self.assertEqual(1, len(find_possible_colorbars()))

        d.plot.imshow(x='x', y='y', col='z', add_colorbar=True)
        self.assertEqual(1, len(find_possible_colorbars()))

        d.plot.imshow(x='x', y='y', col='z', add_colorbar=False)
        self.assertEqual(0, len(find_possible_colorbars()))

    @pytest.mark.slow
    def test_facetgrid_polar(self):
        # test if polar projection in FacetGrid does not raise an exception
        self.darray.plot.pcolormesh(col='z',
                                    subplot_kws=dict(projection='polar'),
                                    sharex=False, sharey=False)


class TestFacetGrid4d(PlotTestCase):

    def setUp(self):
        a = easy_array((10, 15, 3, 2))
        darray = DataArray(a, dims=['y', 'x', 'col', 'row'])
        darray.coords['col'] = np.array(['col' + str(x) for x in
                                         darray.coords['col'].values])
        darray.coords['row'] = np.array(['row' + str(x) for x in
                                         darray.coords['row'].values])

        self.darray = darray

    @pytest.mark.slow
    def test_default_labels(self):
        g = xplt.FacetGrid(self.darray, col='col', row='row')
        self.assertEqual((2, 3), g.axes.shape)

        g.map_dataarray(xplt.imshow, 'x', 'y')

        # Rightmost column should be labeled
        for label, ax in zip(self.darray.coords['row'].values, g.axes[:, -1]):
            self.assertTrue(substring_in_axes(label, ax))

        # Top row should be labeled
        for label, ax in zip(self.darray.coords['col'].values, g.axes[0, :]):
            self.assertTrue(substring_in_axes(label, ax))


class TestDatetimePlot(PlotTestCase):

    def setUp(self):
        '''
        Create a DataArray with a time-axis that contains datetime objects.
        '''
        month = np.arange(1, 13, 1)
        data = np.sin(2 * np.pi * month / 12.0)

        darray = DataArray(data, dims=['time'])
        darray.coords['time'] = np.array([datetime(2017, m, 1) for m in month])

        self.darray = darray

    def test_datetime_line_plot(self):
        # test if line plot raises no Exception
        self.darray.plot.line()


@requires_seaborn
def test_import_seaborn_no_warning():
    # GH1633
    with pytest.warns(None) as record:
        import_seaborn()
    assert len(record) == 0


@requires_matplotlib
def test_plot_seaborn_no_import_warning():
    # GH1633
    with pytest.warns(None) as record:
        _color_palette('Blues', 4)
    assert len(record) == 0
