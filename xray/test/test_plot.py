import numpy as np
import pandas as pd

from xray import DataArray

import xray.plot as xplt
from xray.plot.plot import (_infer_interval_breaks,
                            _determine_cmap_params,
                            _determine_discrete_cmap_params)

from . import TestCase, requires_matplotlib

try:
    import matplotlib as mpl
    # Using a different backend makes Travis CI work.
    mpl.use('Agg')
    # Order of imports is important here.
    import matplotlib.pyplot as plt
except ImportError:
    pass


@requires_matplotlib
class PlotTestCase(TestCase):

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')

    def pass_in_axis(self, plotmethod):
        fig, axes = plt.subplots(ncols=2)
        plotmethod(ax=axes[0])
        self.assertTrue(axes[0].has_data())

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
        self.darray = DataArray(np.random.randn(2, 3, 4))

    def test1d(self):
        self.darray[:, 0, 0].plot()

    def test2d_uniform_calls_imshow(self):
        self.assertTrue(self.imshow_called(self.darray[:, :, 0].plot))

    def test2d_nonuniform_calls_contourf(self):
        a = self.darray[:, :, 0]
        a.coords['dim_1'] = [2, 1, 89]
        self.assertTrue(self.contourf_called(a.plot))

    def test3d(self):
        self.darray.plot()

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot)

    def test__infer_interval_breaks(self):
        self.assertArrayEqual([-0.5, 0.5, 1.5], _infer_interval_breaks([0, 1]))
        self.assertArrayEqual([-0.5, 0.5, 5.0, 9.5, 10.5],
                              _infer_interval_breaks([0, 1, 9, 10]))
        self.assertArrayEqual(pd.date_range('20000101', periods=4) - np.timedelta64(12, 'h'),
                              _infer_interval_breaks(pd.date_range('20000101', periods=3)))


class TestPlot1D(PlotTestCase):

    def setUp(self):
        d = [0, 1.1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

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
        twodims = DataArray(np.arange(10).reshape(2, 5))
        with self.assertRaises(ValueError):
            twodims.plot.line()

    def test_format_string(self):
        self.darray.plot.line('ro')

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot.line)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray([1, 2, 3], {'letter': ['a', 'b', 'c']})
        with self.assertRaisesRegexp(TypeError, r'[Pp]lot'):
            a.plot.line()

    def test_primitive_returned(self):
        p = self.darray.plot.line()
        self.assertTrue(isinstance(p[0], mpl.lines.Line2D))

    def test_plot_nans(self):
        self.darray[1] = np.nan
        self.darray.plot.line()

    def test_x_ticks_are_rotated_for_time(self):
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.arange(len(time)), {'t': time})
        a.plot.line()
        rotation = plt.gca().get_xticklabels()[0].get_rotation()
        self.assertFalse(rotation == 0)


class TestPlotHistogram(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(2, 3, 4))

    def test_3d_array(self):
        self.darray.plot.hist()

    def test_title_no_name(self):
        self.darray.plot.hist()
        self.assertEqual('', plt.gca().get_title())

    def test_title_uses_name(self):
        self.darray.name = 'randompoints'
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

    def test_plot_nans(self):
        self.darray[0, 0, 0] = np.nan
        self.darray.plot.hist()


@requires_matplotlib
class TestDetermineCmapParams(TestCase):
    def test_robust(self):
        data = np.random.RandomState(1).rand(100)
        vmin, vmax, cmap, extend, cnorm = _determine_cmap_params(data,
                                                                 robust=True)
        self.assertEqual(vmin, np.percentile(data, 2))
        self.assertEqual(vmax, np.percentile(data, 98))
        self.assertEqual(cmap.name, 'viridis')
        self.assertEqual(extend, 'both')

    def test_center(self):
        data = np.random.RandomState(2).rand(100)
        vmin, vmax, cmap, extend, cnorm = _determine_cmap_params(data,
                                                                 center=0.5)
        self.assertEqual(vmax - 0.5, 0.5 - vmin)
        self.assertEqual(cmap, 'RdBu_r')
        self.assertEqual(extend, 'neither')


@requires_matplotlib
class TestDetermineDiscreteCmapParams(TestCase):
    def test_integer_levels(self):
        levels = 8
        vmin = -5
        vmax = 5
        cmap, cnorm = _determine_discrete_cmap_params('Spectral', levels,
                                                      vmin, vmax, 'neither')
        self.assertEqual(cmap.N, levels)
        self.assertEqual(cnorm.N, levels + 1)
        self.assertEqual(cnorm.vmin, vmin)
        self.assertEqual(cnorm.vmax, vmax + 10 * np.finfo(float).eps)

        cmap, cnorm = _determine_discrete_cmap_params('Blues', levels,
                                                      vmin, vmax, 'both')
        # extension colors are not included here
        self.assertEqual(cmap.N, levels)
        self.assertEqual(cnorm.N, levels + 1)
        self.assertEqual(cnorm.vmin, vmin)
        self.assertEqual(cnorm.vmax, vmax + 10 * np.finfo(float).eps)

    def test_list_levels(self):
        levels = [-4, -2, 0, 2, 4]
        vmin = -5
        vmax = 5

        cmap, cnorm = _determine_discrete_cmap_params('Spectral', levels,
                                                      vmin, vmax, 'neither')
        self.assertEqual(cmap.N, len(levels) - 1)
        self.assertEqual(cnorm.N, len(levels))
        self.assertEqual(cnorm.vmin, min(levels))
        self.assertEqual(cnorm.vmax, max(levels))

        cmap, cnorm = _determine_discrete_cmap_params('Greens_r', levels,
                                                      vmin, vmax, 'both')
        self.assertEqual(cmap.N, len(levels) - 1)
        self.assertEqual(cnorm.N, len(levels))
        self.assertEqual(cnorm.vmin, min(levels))
        self.assertEqual(cnorm.vmax, max(levels))

        # levels as an array
        cmap, cnorm = _determine_discrete_cmap_params('Greens_r',
                                                      np.array(levels),
                                                      vmin, vmax, 'both')
        # levels as a DataArray
        cmap, cnorm = _determine_discrete_cmap_params('Greens_r',
                                                      DataArray(levels),
                                                      vmin, vmax, 'both')


class Common2dMixin:
    """
    Common tests for 2d plotting go here.

    These tests assume that a staticmethod for `self.plotfunc` exists.
    Should have the same name as the method.
    """
    def setUp(self):
        rs = np.random.RandomState(123)
        self.darray = DataArray(rs.randn(10, 15), dims=['y', 'x'])
        self.plotmethod = getattr(self.darray.plot, self.plotfunc.__name__)

    def test_label_names(self):
        self.plotmethod()
        self.assertEqual('x', plt.gca().get_xlabel())
        self.assertEqual('y', plt.gca().get_ylabel())

    def test_1d_raises_valueerror(self):
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            self.plotfunc(self.darray[0, :])

    def test_3d_raises_valueerror(self):
        a = DataArray(np.random.randn(2, 3, 4))
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            self.plotfunc(a)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray(np.random.randn(3, 2),
                      coords=[['a', 'b', 'c'], ['d', 'e']])
        with self.assertRaisesRegexp(TypeError, r'[Pp]lot'):
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

    def test_can_change_default_cmap(self):
        cmap_name = self.plotmethod(cmap='jet').get_cmap().name
        self.assertEqual('jet', cmap_name)

    def test_diverging_color_limits(self):
        artist = self.plotmethod()
        vmin, vmax = artist.get_clim()
        self.assertAlmostEqual(-vmin, vmax)


class TestContourf(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contourf)

    def test_contourf_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.contourf_called(self.darray.plot.imshow))
        self.assertTrue(self.contourf_called(self.darray.plot.contourf))

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.contour.QuadContourSet))

    def test_extend(self):
        artist = self.plotmethod()
        self.assertEqual(artist.extend, 'neither')

        artist = self.plotmethod(robust=True)
        self.assertEqual(artist.extend, 'both')

        artist = self.plotmethod(vmin=-0, vmax=10)
        self.assertEqual(artist.extend, 'min')

        artist = self.plotmethod(vmin=-10, vmax=0)
        self.assertEqual(artist.extend, 'max')


class TestContour(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contour)


class TestPcolormesh(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.pcolormesh)

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.collections.QuadMesh))

    def test_everything_plotted(self):
        artist = self.plotmethod()
        self.assertEqual(artist.get_array().size, self.darray.size)


class TestImshow(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.imshow)

    def test_imshow_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.imshow_called(self.darray.plot.contourf))
        self.assertTrue(self.imshow_called(self.darray.plot.imshow))

    def test_xy_pixel_centered(self):
        self.darray.plot.imshow()
        self.assertTrue(np.allclose([-0.5, 14.5], plt.gca().get_xlim()))
        self.assertTrue(np.allclose([9.5, -0.5], plt.gca().get_ylim()))

    def test_default_aspect_is_auto(self):
        self.darray.plot.imshow()
        self.assertEqual('auto', plt.gca().get_aspect())

    def test_can_change_aspect(self):
        self.darray.plot.imshow(aspect='equal')
        self.assertEqual('equal', plt.gca().get_aspect())

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.image.AxesImage))
