import numpy as np

from xray import DataArray
# Shouldn't need the core here?
from xray.core.plotting import plot_imshow, plot_contourf

from . import TestCase, requires_matplotlib

try:
    import matplotlib as mpl
    # Using a different backend makes Travis CI work.
    mpl.use('Agg')
    # Order of imports is important here.
    import matplotlib.pyplot as plt
except ImportError:
    pass

# TODO - Add NaN handling and tests

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
            twodims.plot_line()

    def test_format_string(self):
        self.darray.plot_line('ro')

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot_line)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray([1, 2, 3], {'letter': ['a', 'b', 'c']})
        with self.assertRaisesRegexp(TypeError, r'[Ii]ndex'):
            a.plot_line()

    def test_primitive_returned(self):
        p = self.darray.plot_line()
        self.assertTrue(isinstance(p[0], mpl.lines.Line2D))

    def test_plot_nans(self):
        self.darray[1] = np.nan
        self.darray.plot_line()


class TestPlotHistogram(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(2, 3, 4))

    def test_3d_array(self):
        self.darray.plot_hist()

    def test_title_no_name(self):
        self.darray.plot_hist()
        self.assertEqual('', plt.gca().get_title())

    def test_title_uses_name(self):
        self.darray.name = 'randompoints'
        self.darray.plot_hist()
        self.assertIn(self.darray.name, plt.gca().get_title())

    def test_ylabel_is_count(self):
        self.darray.plot_hist()
        self.assertEqual('Count', plt.gca().get_ylabel())

    def test_can_pass_in_kwargs(self):
        nbins = 5
        self.darray.plot_hist(bins=nbins)
        self.assertEqual(nbins, len(plt.gca().patches))

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot_hist)

    def test_primitive_returned(self):
        h = self.darray.plot_hist()
        self.assertTrue(isinstance(h[-1][0], mpl.patches.Rectangle))

    def test_plot_nans(self):
        self.darray[0, 0, :] = np.nan
        self.darray.plot_hist()


class Common2dMixin:
    """
    Common tests for 2d plotting go here. 

    These tests assume that `self.plotfunc` exists and is defined in the
    setUp. Should have the same name as the method.
    """
    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), dims=['y', 'x'])
        self.plotmethod = getattr(self.darray, self.plotfunc.__name__)

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
        a = DataArray(np.random.randn(3, 2), coords=[['a', 'b', 'c'],
            ['d', 'e']])
        with self.assertRaisesRegexp(TypeError, r'[Ii]ndex'):
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
        self.darray[0, 0] = np.nan
        self.plotmethod()


class TestContourf(Common2dMixin, PlotTestCase):

    def setUp(self):
        self.plotfunc = plot_contourf
        super(TestContourf, self).setUp()

    def test_contourf_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.contourf_called(self.darray.plot_imshow))
        self.assertTrue(self.contourf_called(self.darray.plot_contourf))

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.contour.QuadContourSet))


class TestImshow(Common2dMixin, PlotTestCase):

    def setUp(self):
        self.plotfunc = plot_imshow
        super(TestImshow, self).setUp()

    def test_imshow_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.imshow_called(self.darray.plot_contourf))
        self.assertTrue(self.imshow_called(self.darray.plot_imshow))

    def test_xy_pixel_centered(self):
        self.darray.plot_imshow()
        self.assertTrue(np.allclose([-0.5, 14.5], plt.gca().get_xlim()))
        self.assertTrue(np.allclose([9.5, -0.5], plt.gca().get_ylim()))

    def test_default_aspect_is_auto(self):
        self.darray.plot_imshow()
        self.assertEqual('auto', plt.gca().get_aspect())

    def test_can_change_aspect(self):
        self.darray.plot_imshow(aspect='equal')
        self.assertEqual('equal', plt.gca().get_aspect())

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        self.assertTrue(isinstance(artist, mpl.image.AxesImage))
