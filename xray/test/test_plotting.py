import numpy as np
import pandas as pd

from xray import Dataset, DataArray

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

    def pass_in_axis(self, plotfunc):
        fig, axes = plt.subplots(ncols=2)
        plotfunc(ax=axes[0])
        self.assertTrue(axes[0].has_data())

    def imshow_called(self, plotfunc):
        ax = plotfunc()
        images = ax.findobj(mpl.image.AxesImage)
        return len(images) > 0

    def contourf_called(self, plotfunc):
        ax = plotfunc()
        paths = ax.findobj(mpl.collections.PathCollection)
        return len(paths) > 0


class TestPlot(PlotTestCase):

    def setUp(self):
        d = np.arange(24).reshape(2, 3, 4)
        self.darray = DataArray(d)

    def test1d(self):
        self.darray[0, 0, :].plot()

    def test2d_uniform_calls_imshow(self):
        a = self.darray[0, :, :]
        self.assertTrue(self.imshow_called(a.plot))

    def test2d_nonuniform_calls_contourf(self):
        a = self.darray[0, :, :]
        a.coords['dim_1'] = [0, 10, 2]
        self.assertTrue(self.contourf_called(a.plot))

    def test3d(self):
        self.darray.plot()

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot)


class TestPlot1D(PlotTestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    def test_xlabel_is_index_name(self):
        ax = self.darray.plot()
        self.assertEqual('period', ax.get_xlabel())

    def test_no_label_name_on_y_axis(self):
        ax = self.darray.plot()
        self.assertEqual('', ax.get_ylabel())

    def test_ylabel_is_data_name(self):
        self.darray.name = 'temperature'
        ax = self.darray.plot()
        self.assertEqual(self.darray.name, ax.get_ylabel())

    def test_wrong_dims_raises_valueerror(self):
        twodims = DataArray(np.arange(10).reshape(2, 5))
        with self.assertRaises(ValueError):
            twodims.plot_line()

    def test_format_string(self):
        self.darray.plot_line('ro')

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot_line)

# TODO - Add NaN handling and tests

class TestPlot2D(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), 
                dims=['y', 'x'])

    def test_contour_label_names(self):
        ax = self.darray.plot_contourf()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('y', ax.get_ylabel())

    def test_imshow_label_names(self):
        ax = self.darray.plot_imshow()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('y', ax.get_ylabel())

    def test_1d_raises_valueerror(self):
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            self.darray[0, :].plot_imshow()

    def test_3d_raises_valueerror(self):
        da = DataArray(np.random.randn(2, 3, 4))
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            da.plot_imshow()

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot_imshow)
        self.pass_in_axis(self.darray.plot_contourf)

    def test_imshow_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.imshow_called(self.darray.plot_contourf))
        self.assertTrue(self.imshow_called(self.darray.plot_imshow))

    def test_contourf_called(self):
        # Having both statements ensures the test works properly
        self.assertFalse(self.contourf_called(self.darray.plot_imshow))
        self.assertTrue(self.contourf_called(self.darray.plot_contourf))

    def test_imshow_xy_pixel_centered(self):
        ax = self.darray.plot_imshow()
        self.assertTrue(np.allclose([-0.5, 14.5], ax.get_xlim()))
        self.assertTrue(np.allclose([9.5, -0.5], ax.get_ylim()))

    def test_default_aspect_is_auto(self):
        ax = self.darray.plot_imshow()
        self.assertEqual('auto', ax.get_aspect())

    def test_can_change_aspect(self):
        ax = self.darray.plot_imshow(aspect='equal')
        self.assertEqual('equal', ax.get_aspect())


class TestPlotHist(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(2, 3, 4))

    def test_3d_array(self):
        self.darray.plot_hist()

    def test_title_no_name(self):
        ax = self.darray.plot_hist()
        self.assertEqual('', ax.get_title())

    def test_title_uses_name(self):
        self.darray.name = 'randompoints'
        ax = self.darray.plot_hist()
        self.assertIn(self.darray.name, ax.get_title())

    def test_ylabel_is_count(self):
        ax = self.darray.plot_hist()
        self.assertEqual('Count', ax.get_ylabel())

    def test_can_pass_in_kwargs(self):
        nbins = 5
        ax = self.darray.plot_hist(bins=nbins)
        self.assertEqual(nbins, len(ax.patches))

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot_hist)
