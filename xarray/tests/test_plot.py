import inspect
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.plot import _infer_interval_breaks
from xarray.plot.utils import (
    _build_discrete_cmap,
    _color_palette,
    _determine_cmap_params,
    import_seaborn,
    label_from_attrs,
)

from . import (
    assert_array_equal,
    assert_equal,
    has_nc_time_axis,
    raises_regex,
    requires_cftime,
    requires_matplotlib,
    requires_matplotlib2,
    requires_nc_time_axis,
    requires_seaborn,
)

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass


@pytest.mark.flaky
@pytest.mark.skip(reason="maybe flaky")
def text_in_fig():
    """
    Return the set of all text in the figure
    """
    return {t.get_text() for t in plt.gcf().findobj(mpl.text.Text)}


def find_possible_colorbars():
    # nb. this function also matches meshes from pcolormesh
    return plt.gcf().findobj(mpl.collections.QuadMesh)


def substring_in_axes(substring, ax):
    """
    Return True if a substring is found anywhere in an axes
    """
    alltxt = {t.get_text() for t in ax.findobj(mpl.text.Text)}
    for txt in alltxt:
        if substring in txt:
            return True
    return False


def easy_array(shape, start=0, stop=1):
    """
    Make an array with desired shape using np.linspace

    shape is a tuple like (2, 3)
    """
    a = np.linspace(start, stop, num=np.prod(shape))
    return a.reshape(shape)


@requires_matplotlib
class PlotTestCase:
    @pytest.fixture(autouse=True)
    def setup(self):
        yield
        # Remove all matplotlib figures
        plt.close("all")

    def pass_in_axis(self, plotmethod):
        fig, axes = plt.subplots(ncols=2)
        plotmethod(ax=axes[0])
        assert axes[0].has_data()

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
    @pytest.fixture(autouse=True)
    def setup_array(self):
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_label_from_attrs(self):
        da = self.darray.copy()
        assert "" == label_from_attrs(da)

        da.name = "a"
        da.attrs["units"] = "a_units"
        da.attrs["long_name"] = "a_long_name"
        da.attrs["standard_name"] = "a_standard_name"
        assert "a_long_name [a_units]" == label_from_attrs(da)

        da.attrs.pop("long_name")
        assert "a_standard_name [a_units]" == label_from_attrs(da)
        da.attrs.pop("units")
        assert "a_standard_name" == label_from_attrs(da)

        da.attrs["units"] = "a_units"
        da.attrs.pop("standard_name")
        assert "a [a_units]" == label_from_attrs(da)

        da.attrs.pop("units")
        assert "a" == label_from_attrs(da)

    def test1d(self):
        self.darray[:, 0, 0].plot()

        with raises_regex(ValueError, "None"):
            self.darray[:, 0, 0].plot(x="dim_1")

    def test_1d_x_y_kw(self):
        z = np.arange(10)
        da = DataArray(np.cos(z), dims=["z"], coords=[z], name="f")

        xy = [[None, None], [None, "z"], ["z", None]]

        f, ax = plt.subplots(3, 1)
        for aa, (x, y) in enumerate(xy):
            da.plot(x=x, y=y, ax=ax.flat[aa])

        with raises_regex(ValueError, "cannot"):
            da.plot(x="z", y="z")

        with raises_regex(ValueError, "None"):
            da.plot(x="f", y="z")

        with raises_regex(ValueError, "None"):
            da.plot(x="z", y="f")

    # Test for bug in GH issue #2725
    def test_infer_line_data(self):
        current = DataArray(
            name="I",
            data=np.array([5, 8]),
            dims=["t"],
            coords={
                "t": (["t"], np.array([0.1, 0.2])),
                "V": (["t"], np.array([100, 200])),
            },
        )

        # Plot current against voltage
        line = current.plot.line(x="V")[0]
        assert_array_equal(line.get_xdata(), current.coords["V"].values)

        # Plot current against time
        line = current.plot.line()[0]
        assert_array_equal(line.get_xdata(), current.coords["t"].values)

    def test_2d_line(self):
        with raises_regex(ValueError, "hue"):
            self.darray[:, :, 0].plot.line()

        self.darray[:, :, 0].plot.line(hue="dim_1")
        self.darray[:, :, 0].plot.line(x="dim_1")
        self.darray[:, :, 0].plot.line(y="dim_1")
        self.darray[:, :, 0].plot.line(x="dim_0", hue="dim_1")
        self.darray[:, :, 0].plot.line(y="dim_0", hue="dim_1")

        with raises_regex(ValueError, "cannot"):
            self.darray[:, :, 0].plot.line(x="dim_1", y="dim_0", hue="dim_1")

    def test_2d_line_accepts_legend_kw(self):
        self.darray[:, :, 0].plot.line(x="dim_0", add_legend=False)
        assert not plt.gca().get_legend()
        plt.cla()
        self.darray[:, :, 0].plot.line(x="dim_0", add_legend=True)
        assert plt.gca().get_legend()
        # check whether legend title is set
        assert plt.gca().get_legend().get_title().get_text() == "dim_1"

    def test_2d_line_accepts_x_kw(self):
        self.darray[:, :, 0].plot.line(x="dim_0")
        assert plt.gca().get_xlabel() == "dim_0"
        plt.cla()
        self.darray[:, :, 0].plot.line(x="dim_1")
        assert plt.gca().get_xlabel() == "dim_1"

    def test_2d_line_accepts_hue_kw(self):
        self.darray[:, :, 0].plot.line(hue="dim_0")
        assert plt.gca().get_legend().get_title().get_text() == "dim_0"
        plt.cla()
        self.darray[:, :, 0].plot.line(hue="dim_1")
        assert plt.gca().get_legend().get_title().get_text() == "dim_1"

    def test_2d_coords_line_plot(self):
        lon, lat = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(0, 30, 4))
        lon += lat / 10
        lat += lon / 10
        da = xr.DataArray(
            np.arange(20).reshape(4, 5),
            dims=["y", "x"],
            coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
        )

        hdl = da.plot.line(x="lon", hue="x")
        assert len(hdl) == 5

        plt.clf()
        hdl = da.plot.line(x="lon", hue="y")
        assert len(hdl) == 4

        with pytest.raises(ValueError, match="For 2D inputs, hue must be a dimension"):
            da.plot.line(x="lon", hue="lat")

    def test_2d_before_squeeze(self):
        a = DataArray(easy_array((1, 5)))
        a.plot()

    def test2d_uniform_calls_imshow(self):
        assert self.imshow_called(self.darray[:, :, 0].plot.imshow)

    @pytest.mark.slow
    def test2d_nonuniform_calls_contourf(self):
        a = self.darray[:, :, 0]
        a.coords["dim_1"] = [2, 1, 89]
        assert self.contourf_called(a.plot.contourf)

    def test2d_1d_2d_coordinates_contourf(self):
        sz = (20, 10)
        depth = easy_array(sz)
        a = DataArray(
            easy_array(sz),
            dims=["z", "time"],
            coords={"depth": (["z", "time"], depth), "time": np.linspace(0, 1, sz[1])},
        )

        a.plot.contourf(x="time", y="depth")

    def test3d(self):
        self.darray.plot()

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot)

    def test__infer_interval_breaks(self):
        assert_array_equal([-0.5, 0.5, 1.5], _infer_interval_breaks([0, 1]))
        assert_array_equal(
            [-0.5, 0.5, 5.0, 9.5, 10.5], _infer_interval_breaks([0, 1, 9, 10])
        )
        assert_array_equal(
            pd.date_range("20000101", periods=4) - np.timedelta64(12, "h"),
            _infer_interval_breaks(pd.date_range("20000101", periods=3)),
        )

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

        # test that ValueError is raised for non-monotonic 1D inputs
        with pytest.raises(ValueError):
            _infer_interval_breaks(np.array([0, 2, 1]), check_monotonic=True)

    def test_geo_data(self):
        # Regression test for gh2250
        # Realistic coordinates taken from the example dataset
        lat = np.array(
            [
                [16.28, 18.48, 19.58, 19.54, 18.35],
                [28.07, 30.52, 31.73, 31.68, 30.37],
                [39.65, 42.27, 43.56, 43.51, 42.11],
                [50.52, 53.22, 54.55, 54.50, 53.06],
            ]
        )
        lon = np.array(
            [
                [-126.13, -113.69, -100.92, -88.04, -75.29],
                [-129.27, -115.62, -101.54, -87.32, -73.26],
                [-133.10, -118.00, -102.31, -86.42, -70.76],
                [-137.85, -120.99, -103.28, -85.28, -67.62],
            ]
        )
        data = np.sqrt(lon ** 2 + lat ** 2)
        da = DataArray(
            data,
            dims=("y", "x"),
            coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
        )
        da.plot(x="lon", y="lat")
        ax = plt.gca()
        assert ax.has_data()
        da.plot(x="lat", y="lon")
        ax = plt.gca()
        assert ax.has_data()

    def test_datetime_dimension(self):
        nrow = 3
        ncol = 4
        time = pd.date_range("2000-01-01", periods=nrow)
        a = DataArray(
            easy_array((nrow, ncol)), coords=[("time", time), ("y", range(ncol))]
        )
        a.plot()
        ax = plt.gca()
        assert ax.has_data()

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:tight_layout cannot")
    def test_convenient_facetgrid(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=["y", "x", "z"])
        d.coords["z"] = list("abcd")
        g = d.plot(x="x", y="y", col="z", col_wrap=2, cmap="cool")

        assert_array_equal(g.axes.shape, [2, 2])
        for ax in g.axes.flat:
            assert ax.has_data()

        with raises_regex(ValueError, "[Ff]acet"):
            d.plot(x="x", y="y", col="z", ax=plt.gca())

        with raises_regex(ValueError, "[Ff]acet"):
            d[0].plot(x="x", y="y", col="z", ax=plt.gca())

    @pytest.mark.slow
    @requires_matplotlib2
    def test_subplot_kws(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=["y", "x", "z"])
        d.coords["z"] = list("abcd")
        g = d.plot(
            x="x",
            y="y",
            col="z",
            col_wrap=2,
            cmap="cool",
            subplot_kws=dict(facecolor="r"),
        )
        for ax in g.axes.flat:
            # mpl V2
            assert ax.get_facecolor()[0:3] == mpl.colors.to_rgb("r")

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

        with raises_regex(ValueError, "cannot provide both"):
            self.darray.plot(ax=plt.gca(), figsize=(3, 4))

        with raises_regex(ValueError, "cannot provide both"):
            self.darray.plot(size=5, figsize=(3, 4))

        with raises_regex(ValueError, "cannot provide both"):
            self.darray.plot(size=5, ax=plt.gca())

        with raises_regex(ValueError, "cannot provide `aspect`"):
            self.darray.plot(aspect=1)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:tight_layout cannot")
    def test_convenient_facetgrid_4d(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=["y", "x", "columns", "rows"])
        g = d.plot(x="x", y="y", col="columns", row="rows")

        assert_array_equal(g.axes.shape, [3, 2])
        for ax in g.axes.flat:
            assert ax.has_data()

        with raises_regex(ValueError, "[Ff]acet"):
            d.plot(x="x", y="y", col="columns", ax=plt.gca())

    def test_coord_with_interval(self):
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins("dim_0", bins).mean(xr.ALL_DIMS).plot()


class TestPlot1D(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        d = [0, 1.1, 0, 2]
        self.darray = DataArray(d, coords={"period": range(len(d))}, dims="period")
        self.darray.period.attrs["units"] = "s"

    def test_xlabel_is_index_name(self):
        self.darray.plot()
        assert "period [s]" == plt.gca().get_xlabel()

    def test_no_label_name_on_x_axis(self):
        self.darray.plot(y="period")
        assert "" == plt.gca().get_xlabel()

    def test_no_label_name_on_y_axis(self):
        self.darray.plot()
        assert "" == plt.gca().get_ylabel()

    def test_ylabel_is_data_name(self):
        self.darray.name = "temperature"
        self.darray.attrs["units"] = "degrees_Celsius"
        self.darray.plot()
        assert "temperature [degrees_Celsius]" == plt.gca().get_ylabel()

    def test_xlabel_is_data_name(self):
        self.darray.name = "temperature"
        self.darray.attrs["units"] = "degrees_Celsius"
        self.darray.plot(y="period")
        assert "temperature [degrees_Celsius]" == plt.gca().get_xlabel()

    def test_format_string(self):
        self.darray.plot.line("ro")

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot.line)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray([1, 2, 3], {"letter": ["a", "b", "c"]}, dims="letter")
        with raises_regex(TypeError, r"[Pp]lot"):
            a.plot.line()

    def test_primitive_returned(self):
        p = self.darray.plot.line()
        assert isinstance(p[0], mpl.lines.Line2D)

    @pytest.mark.slow
    def test_plot_nans(self):
        self.darray[1] = np.nan
        self.darray.plot.line()

    def test_x_ticks_are_rotated_for_time(self):
        time = pd.date_range("2000-01-01", "2000-01-10")
        a = DataArray(np.arange(len(time)), [("t", time)])
        a.plot.line()
        rotation = plt.gca().get_xticklabels()[0].get_rotation()
        assert rotation != 0

    def test_xyincrease_false_changes_axes(self):
        self.darray.plot.line(xincrease=False, yincrease=False)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = xlim[1] - xlim[0], ylim[1] - ylim[0]
        assert all(x < 0 for x in diffs)

    def test_slice_in_title(self):
        self.darray.coords["d"] = 10
        self.darray.plot.line()
        title = plt.gca().get_title()
        assert "d = 10" == title


class TestPlotStep(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_step(self):
        self.darray[0, 0].plot.step()

    def test_coord_with_interval_step(self):
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins("dim_0", bins).mean(xr.ALL_DIMS).plot.step()
        assert len(plt.gca().lines[0].get_xdata()) == ((len(bins) - 1) * 2)


class TestPlotHistogram(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_3d_array(self):
        self.darray.plot.hist()

    def test_xlabel_uses_name(self):
        self.darray.name = "testpoints"
        self.darray.attrs["units"] = "testunits"
        self.darray.plot.hist()
        assert "testpoints [testunits]" == plt.gca().get_xlabel()

    def test_title_is_histogram(self):
        self.darray.plot.hist()
        assert "Histogram" == plt.gca().get_title()

    def test_can_pass_in_kwargs(self):
        nbins = 5
        self.darray.plot.hist(bins=nbins)
        assert nbins == len(plt.gca().patches)

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.darray.plot.hist)

    def test_primitive_returned(self):
        h = self.darray.plot.hist()
        assert isinstance(h[-1][0], mpl.patches.Rectangle)

    @pytest.mark.slow
    def test_plot_nans(self):
        self.darray[0, 0, 0] = np.nan
        self.darray.plot.hist()

    def test_hist_coord_with_interval(self):
        (
            self.darray.groupby_bins("dim_0", [-1, 0, 1, 2])
            .mean(xr.ALL_DIMS)
            .plot.hist(range=(-1, 2))
        )


@requires_matplotlib
class TestDetermineCmapParams:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.data = np.linspace(0, 1, num=100)

    def test_robust(self):
        cmap_params = _determine_cmap_params(self.data, robust=True)
        assert cmap_params["vmin"] == np.percentile(self.data, 2)
        assert cmap_params["vmax"] == np.percentile(self.data, 98)
        assert cmap_params["cmap"] == "viridis"
        assert cmap_params["extend"] == "both"
        assert cmap_params["levels"] is None
        assert cmap_params["norm"] is None

    def test_center(self):
        cmap_params = _determine_cmap_params(self.data, center=0.5)
        assert cmap_params["vmax"] - 0.5 == 0.5 - cmap_params["vmin"]
        assert cmap_params["cmap"] == "RdBu_r"
        assert cmap_params["extend"] == "neither"
        assert cmap_params["levels"] is None
        assert cmap_params["norm"] is None

    def test_cmap_sequential_option(self):
        with xr.set_options(cmap_sequential="magma"):
            cmap_params = _determine_cmap_params(self.data)
            assert cmap_params["cmap"] == "magma"

    def test_cmap_sequential_explicit_option(self):
        with xr.set_options(cmap_sequential=mpl.cm.magma):
            cmap_params = _determine_cmap_params(self.data)
            assert cmap_params["cmap"] == mpl.cm.magma

    def test_cmap_divergent_option(self):
        with xr.set_options(cmap_divergent="magma"):
            cmap_params = _determine_cmap_params(self.data, center=0.5)
            assert cmap_params["cmap"] == "magma"

    def test_nan_inf_are_ignored(self):
        cmap_params1 = _determine_cmap_params(self.data)
        data = self.data
        data[50:55] = np.nan
        data[56:60] = np.inf
        cmap_params2 = _determine_cmap_params(data)
        assert cmap_params1["vmin"] == cmap_params2["vmin"]
        assert cmap_params1["vmax"] == cmap_params2["vmax"]

    @pytest.mark.slow
    def test_integer_levels(self):
        data = self.data + 1

        # default is to cover full data range but with no guarantee on Nlevels
        for level in np.arange(2, 10, dtype=int):
            cmap_params = _determine_cmap_params(data, levels=level)
            assert cmap_params["vmin"] == cmap_params["levels"][0]
            assert cmap_params["vmax"] == cmap_params["levels"][-1]
            assert cmap_params["extend"] == "neither"

        # with min max we are more strict
        cmap_params = _determine_cmap_params(
            data, levels=5, vmin=0, vmax=5, cmap="Blues"
        )
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 5
        assert cmap_params["vmin"] == cmap_params["levels"][0]
        assert cmap_params["vmax"] == cmap_params["levels"][-1]
        assert cmap_params["cmap"].name == "Blues"
        assert cmap_params["extend"] == "neither"
        assert cmap_params["cmap"].N == 4
        assert cmap_params["norm"].N == 5

        cmap_params = _determine_cmap_params(data, levels=5, vmin=0.5, vmax=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "max"

        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "min"

        cmap_params = _determine_cmap_params(data, levels=5, vmin=1.3, vmax=1.5)
        assert cmap_params["cmap"].name == "viridis"
        assert cmap_params["extend"] == "both"

    def test_list_levels(self):
        data = self.data + 1

        orig_levels = [0, 1, 2, 3, 4, 5]
        # vmin and vmax should be ignored if levels are explicitly provided
        cmap_params = _determine_cmap_params(data, levels=orig_levels, vmin=0, vmax=3)
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 5
        assert cmap_params["cmap"].N == 5
        assert cmap_params["norm"].N == 6

        for wrap_levels in [list, np.array, pd.Index, DataArray]:
            cmap_params = _determine_cmap_params(data, levels=wrap_levels(orig_levels))
            assert_array_equal(cmap_params["levels"], orig_levels)

    def test_divergentcontrol(self):
        neg = self.data - 0.1
        pos = self.data

        # Default with positive data will be a normal cmap
        cmap_params = _determine_cmap_params(pos)
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 1
        assert cmap_params["cmap"] == "viridis"

        # Default with negative data will be a divergent cmap
        cmap_params = _determine_cmap_params(neg)
        assert cmap_params["vmin"] == -0.9
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "RdBu_r"

        # Setting vmin or vmax should prevent this only if center is false
        cmap_params = _determine_cmap_params(neg, vmin=-0.1, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "viridis"
        cmap_params = _determine_cmap_params(neg, vmax=0.5, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "viridis"

        # Setting center=False too
        cmap_params = _determine_cmap_params(neg, center=False)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "viridis"

        # However, I should still be able to set center and have a div cmap
        cmap_params = _determine_cmap_params(neg, center=0)
        assert cmap_params["vmin"] == -0.9
        assert cmap_params["vmax"] == 0.9
        assert cmap_params["cmap"] == "RdBu_r"

        # Setting vmin or vmax alone will force symmetric bounds around center
        cmap_params = _determine_cmap_params(neg, vmin=-0.1)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.1
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(neg, vmax=0.5)
        assert cmap_params["vmin"] == -0.5
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(neg, vmax=0.6, center=0.1)
        assert cmap_params["vmin"] == -0.4
        assert cmap_params["vmax"] == 0.6
        assert cmap_params["cmap"] == "RdBu_r"

        # But this is only true if vmin or vmax are negative
        cmap_params = _determine_cmap_params(pos, vmin=-0.1)
        assert cmap_params["vmin"] == -0.1
        assert cmap_params["vmax"] == 0.1
        assert cmap_params["cmap"] == "RdBu_r"
        cmap_params = _determine_cmap_params(pos, vmin=0.1)
        assert cmap_params["vmin"] == 0.1
        assert cmap_params["vmax"] == 1
        assert cmap_params["cmap"] == "viridis"
        cmap_params = _determine_cmap_params(pos, vmax=0.5)
        assert cmap_params["vmin"] == 0
        assert cmap_params["vmax"] == 0.5
        assert cmap_params["cmap"] == "viridis"

        # If both vmin and vmax are provided, output is non-divergent
        cmap_params = _determine_cmap_params(neg, vmin=-0.2, vmax=0.6)
        assert cmap_params["vmin"] == -0.2
        assert cmap_params["vmax"] == 0.6
        assert cmap_params["cmap"] == "viridis"

    def test_norm_sets_vmin_vmax(self):
        vmin = self.data.min()
        vmax = self.data.max()

        for norm, extend in zip(
            [
                mpl.colors.LogNorm(),
                mpl.colors.LogNorm(vmin + 1, vmax - 1),
                mpl.colors.LogNorm(None, vmax - 1),
                mpl.colors.LogNorm(vmin + 1, None),
            ],
            ["neither", "both", "max", "min"],
        ):

            test_min = vmin if norm.vmin is None else norm.vmin
            test_max = vmax if norm.vmax is None else norm.vmax

            cmap_params = _determine_cmap_params(self.data, norm=norm)

            assert cmap_params["vmin"] == test_min
            assert cmap_params["vmax"] == test_max
            assert cmap_params["extend"] == extend
            assert cmap_params["norm"] == norm


@requires_matplotlib
class TestDiscreteColorMap:
    @pytest.fixture(autouse=True)
    def setUp(self):
        x = np.arange(start=0, stop=10, step=2)
        y = np.arange(start=9, stop=-7, step=-3)
        xy = np.dstack(np.meshgrid(x, y))
        distance = np.linalg.norm(xy, axis=2)
        self.darray = DataArray(distance, list(zip(("y", "x"), (y, x))))
        self.data_min = distance.min()
        self.data_max = distance.max()

    @pytest.mark.slow
    def test_recover_from_seaborn_jet_exception(self):
        pal = _color_palette("jet", 4)
        assert type(pal) == np.ndarray
        assert len(pal) == 4

    @pytest.mark.slow
    def test_build_discrete_cmap(self):
        for (cmap, levels, extend, filled) in [
            ("jet", [0, 1], "both", False),
            ("hot", [-4, 4], "max", True),
        ]:
            ncmap, cnorm = _build_discrete_cmap(cmap, levels, extend, filled)
            assert ncmap.N == len(levels) - 1
            assert len(ncmap.colors) == len(levels) - 1
            assert cnorm.N == len(levels)
            assert_array_equal(cnorm.boundaries, levels)
            assert max(levels) == cnorm.vmax
            assert min(levels) == cnorm.vmin
            if filled:
                assert ncmap.colorbar_extend == extend
            else:
                assert ncmap.colorbar_extend == "max"

    @pytest.mark.slow
    def test_discrete_colormap_list_of_levels(self):
        for extend, levels in [
            ("max", [-1, 2, 4, 8, 10]),
            ("both", [2, 5, 10, 11]),
            ("neither", [0, 5, 10, 15]),
            ("min", [2, 5, 10, 15]),
        ]:
            for kind in ["imshow", "pcolormesh", "contourf", "contour"]:
                primitive = getattr(self.darray.plot, kind)(levels=levels)
                assert_array_equal(levels, primitive.norm.boundaries)
                assert max(levels) == primitive.norm.vmax
                assert min(levels) == primitive.norm.vmin
                if kind != "contour":
                    assert extend == primitive.cmap.colorbar_extend
                else:
                    assert "max" == primitive.cmap.colorbar_extend
                assert len(levels) - 1 == len(primitive.cmap.colors)

    @pytest.mark.slow
    def test_discrete_colormap_int_levels(self):
        for extend, levels, vmin, vmax, cmap in [
            ("neither", 7, None, None, None),
            ("neither", 7, None, 20, mpl.cm.RdBu),
            ("both", 7, 4, 8, None),
            ("min", 10, 4, 15, None),
        ]:
            for kind in ["imshow", "pcolormesh", "contourf", "contour"]:
                primitive = getattr(self.darray.plot, kind)(
                    levels=levels, vmin=vmin, vmax=vmax, cmap=cmap
                )
                assert levels >= len(primitive.norm.boundaries) - 1
                if vmax is None:
                    assert primitive.norm.vmax >= self.data_max
                else:
                    assert primitive.norm.vmax >= vmax
                if vmin is None:
                    assert primitive.norm.vmin <= self.data_min
                else:
                    assert primitive.norm.vmin <= vmin
                if kind != "contour":
                    assert extend == primitive.cmap.colorbar_extend
                else:
                    assert "max" == primitive.cmap.colorbar_extend
                assert levels >= len(primitive.cmap.colors)

    def test_discrete_colormap_list_levels_and_vmin_or_vmax(self):
        levels = [0, 5, 10, 15]
        primitive = self.darray.plot(levels=levels, vmin=-3, vmax=20)
        assert primitive.norm.vmax == max(levels)
        assert primitive.norm.vmin == min(levels)

    def test_discrete_colormap_provided_boundary_norm(self):
        norm = mpl.colors.BoundaryNorm([0, 5, 10, 15], 4)
        primitive = self.darray.plot.contourf(norm=norm)
        np.testing.assert_allclose(primitive.levels, norm.boundaries)


class Common2dMixin:
    """
    Common tests for 2d plotting go here.

    These tests assume that a staticmethod for `self.plotfunc` exists.
    Should have the same name as the method.
    """

    @pytest.fixture(autouse=True)
    def setUp(self):
        da = DataArray(
            easy_array((10, 15), start=-1),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(15)},
        )
        # add 2d coords
        ds = da.to_dataset(name="testvar")
        x, y = np.meshgrid(da.x.values, da.y.values)
        ds["x2d"] = DataArray(x, dims=["y", "x"])
        ds["y2d"] = DataArray(y, dims=["y", "x"])
        ds = ds.set_coords(["x2d", "y2d"])
        # set darray and plot method
        self.darray = ds.testvar

        # Add CF-compliant metadata
        self.darray.attrs["long_name"] = "a_long_name"
        self.darray.attrs["units"] = "a_units"
        self.darray.x.attrs["long_name"] = "x_long_name"
        self.darray.x.attrs["units"] = "x_units"
        self.darray.y.attrs["long_name"] = "y_long_name"
        self.darray.y.attrs["units"] = "y_units"

        self.plotmethod = getattr(self.darray.plot, self.plotfunc.__name__)

    def test_label_names(self):
        self.plotmethod()
        assert "x_long_name [x_units]" == plt.gca().get_xlabel()
        assert "y_long_name [y_units]" == plt.gca().get_ylabel()

    def test_1d_raises_valueerror(self):
        with raises_regex(ValueError, r"DataArray must be 2d"):
            self.plotfunc(self.darray[0, :])

    def test_3d_raises_valueerror(self):
        a = DataArray(easy_array((2, 3, 4)))
        if self.plotfunc.__name__ == "imshow":
            pytest.skip()
        with raises_regex(ValueError, r"DataArray must be 2d"):
            self.plotfunc(a)

    def test_nonnumeric_index_raises_typeerror(self):
        a = DataArray(easy_array((3, 2)), coords=[["a", "b", "c"], ["d", "e"]])
        with raises_regex(TypeError, r"[Pp]lot"):
            self.plotfunc(a)

    def test_can_pass_in_axis(self):
        self.pass_in_axis(self.plotmethod)

    def test_xyincrease_defaults(self):

        # With default settings the axis must be ordered regardless
        # of the coords order.
        self.plotfunc(DataArray(easy_array((3, 2)), coords=[[1, 2, 3], [1, 2]]))
        bounds = plt.gca().get_ylim()
        assert bounds[0] < bounds[1]
        bounds = plt.gca().get_xlim()
        assert bounds[0] < bounds[1]
        # Inverted coords
        self.plotfunc(DataArray(easy_array((3, 2)), coords=[[3, 2, 1], [2, 1]]))
        bounds = plt.gca().get_ylim()
        assert bounds[0] < bounds[1]
        bounds = plt.gca().get_xlim()
        assert bounds[0] < bounds[1]

    def test_xyincrease_false_changes_axes(self):
        self.plotmethod(xincrease=False, yincrease=False)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = xlim[0] - 14, xlim[1] - 0, ylim[0] - 9, ylim[1] - 0
        assert all(abs(x) < 1 for x in diffs)

    def test_xyincrease_true_changes_axes(self):
        self.plotmethod(xincrease=True, yincrease=True)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = xlim[0] - 0, xlim[1] - 14, ylim[0] - 0, ylim[1] - 9
        assert all(abs(x) < 1 for x in diffs)

    def test_x_ticks_are_rotated_for_time(self):
        time = pd.date_range("2000-01-01", "2000-01-10")
        a = DataArray(np.random.randn(2, len(time)), [("xx", [1, 2]), ("t", time)])
        a.plot(x="t")
        rotation = plt.gca().get_xticklabels()[0].get_rotation()
        assert rotation != 0

    def test_plot_nans(self):
        x1 = self.darray[:5]
        x2 = self.darray.copy()
        x2[5:] = np.nan

        clim1 = self.plotfunc(x1).get_clim()
        clim2 = self.plotfunc(x2).get_clim()
        assert clim1 == clim2

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_can_plot_all_nans(self):
        # regression test for issue #1780
        self.plotfunc(DataArray(np.full((2, 2), np.nan)))

    @pytest.mark.filterwarnings("ignore: Attempting to set")
    def test_can_plot_axis_size_one(self):
        if self.plotfunc.__name__ not in ("contour", "contourf"):
            self.plotfunc(DataArray(np.ones((1, 1))))

    def test_disallows_rgb_arg(self):
        with pytest.raises(ValueError):
            # Always invalid for most plots.  Invalid for imshow with 2D data.
            self.plotfunc(DataArray(np.ones((2, 2))), rgb="not None")

    def test_viridis_cmap(self):
        cmap_name = self.plotmethod(cmap="viridis").get_cmap().name
        assert "viridis" == cmap_name

    def test_default_cmap(self):
        cmap_name = self.plotmethod().get_cmap().name
        assert "RdBu_r" == cmap_name

        cmap_name = self.plotfunc(abs(self.darray)).get_cmap().name
        assert "viridis" == cmap_name

    @requires_seaborn
    def test_seaborn_palette_as_cmap(self):
        cmap_name = self.plotmethod(levels=2, cmap="husl").get_cmap().name
        assert "husl" == cmap_name

    def test_can_change_default_cmap(self):
        cmap_name = self.plotmethod(cmap="Blues").get_cmap().name
        assert "Blues" == cmap_name

    def test_diverging_color_limits(self):
        artist = self.plotmethod()
        vmin, vmax = artist.get_clim()
        assert round(abs(-vmin - vmax), 7) == 0

    def test_xy_strings(self):
        self.plotmethod("y", "x")
        ax = plt.gca()
        assert "y_long_name [y_units]" == ax.get_xlabel()
        assert "x_long_name [x_units]" == ax.get_ylabel()

    def test_positional_coord_string(self):
        self.plotmethod(y="x")
        ax = plt.gca()
        assert "x_long_name [x_units]" == ax.get_ylabel()
        assert "y_long_name [y_units]" == ax.get_xlabel()

        self.plotmethod(x="x")
        ax = plt.gca()
        assert "x_long_name [x_units]" == ax.get_xlabel()
        assert "y_long_name [y_units]" == ax.get_ylabel()

    def test_bad_x_string_exception(self):
        with raises_regex(ValueError, "x and y must be coordinate variables"):
            self.plotmethod("not_a_real_dim", "y")
        with raises_regex(
            ValueError, "x must be a dimension name if y is not supplied"
        ):
            self.plotmethod(x="not_a_real_dim")
        with raises_regex(
            ValueError, "y must be a dimension name if x is not supplied"
        ):
            self.plotmethod(y="not_a_real_dim")
        self.darray.coords["z"] = 100

    def test_coord_strings(self):
        # 1d coords (same as dims)
        assert {"x", "y"} == set(self.darray.dims)
        self.plotmethod(y="y", x="x")

    def test_non_linked_coords(self):
        # plot with coordinate names that are not dimensions
        self.darray.coords["newy"] = self.darray.y + 150
        # Normal case, without transpose
        self.plotfunc(self.darray, x="x", y="newy")
        ax = plt.gca()
        assert "x_long_name [x_units]" == ax.get_xlabel()
        assert "newy" == ax.get_ylabel()
        # ax limits might change between plotfuncs
        # simply ensure that these high coords were passed over
        assert np.min(ax.get_ylim()) > 100.0

    def test_non_linked_coords_transpose(self):
        # plot with coordinate names that are not dimensions,
        # and with transposed y and x axes
        # This used to raise an error with pcolormesh and contour
        # https://github.com/pydata/xarray/issues/788
        self.darray.coords["newy"] = self.darray.y + 150
        self.plotfunc(self.darray, x="newy", y="x")
        ax = plt.gca()
        assert "newy" == ax.get_xlabel()
        assert "x_long_name [x_units]" == ax.get_ylabel()
        # ax limits might change between plotfuncs
        # simply ensure that these high coords were passed over
        assert np.min(ax.get_xlim()) > 100.0

    def test_default_title(self):
        a = DataArray(easy_array((4, 3, 2)), dims=["a", "b", "c"])
        a.coords["c"] = [0, 1]
        a.coords["d"] = "foo"
        self.plotfunc(a.isel(c=1))
        title = plt.gca().get_title()
        assert "c = 1, d = foo" == title or "d = foo, c = 1" == title

    def test_colorbar_default_label(self):
        self.plotmethod(add_colorbar=True)
        assert "a_long_name [a_units]" in text_in_fig()

    def test_no_labels(self):
        self.darray.name = "testvar"
        self.darray.attrs["units"] = "test_units"
        self.plotmethod(add_labels=False)
        alltxt = text_in_fig()
        for string in [
            "x_long_name [x_units]",
            "y_long_name [y_units]",
            "testvar [test_units]",
        ]:
            assert string not in alltxt

    def test_colorbar_kwargs(self):
        # replace label
        self.darray.attrs.pop("long_name")
        self.darray.attrs["units"] = "test_units"
        # check default colorbar label
        self.plotmethod(add_colorbar=True)
        alltxt = text_in_fig()
        assert "testvar [test_units]" in alltxt
        self.darray.attrs.pop("units")

        self.darray.name = "testvar"
        self.plotmethod(add_colorbar=True, cbar_kwargs={"label": "MyLabel"})
        alltxt = text_in_fig()
        assert "MyLabel" in alltxt
        assert "testvar" not in alltxt
        # you can use anything accepted by the dict constructor as well
        self.plotmethod(add_colorbar=True, cbar_kwargs=(("label", "MyLabel"),))
        alltxt = text_in_fig()
        assert "MyLabel" in alltxt
        assert "testvar" not in alltxt
        # change cbar ax
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(
            ax=ax, cbar_ax=cax, add_colorbar=True, cbar_kwargs={"label": "MyBar"}
        )
        assert ax.has_data()
        assert cax.has_data()
        alltxt = text_in_fig()
        assert "MyBar" in alltxt
        assert "testvar" not in alltxt
        # note that there are two ways to achieve this
        fig, (ax, cax) = plt.subplots(1, 2)
        self.plotmethod(
            ax=ax, add_colorbar=True, cbar_kwargs={"label": "MyBar", "cax": cax}
        )
        assert ax.has_data()
        assert cax.has_data()
        alltxt = text_in_fig()
        assert "MyBar" in alltxt
        assert "testvar" not in alltxt
        # see that no colorbar is respected
        self.plotmethod(add_colorbar=False)
        assert "testvar" not in text_in_fig()
        # check that error is raised
        pytest.raises(
            ValueError,
            self.plotmethod,
            add_colorbar=False,
            cbar_kwargs={"label": "label"},
        )

    def test_verbose_facetgrid(self):
        a = easy_array((10, 15, 3))
        d = DataArray(a, dims=["y", "x", "z"])
        g = xplt.FacetGrid(d, col="z")
        g.map_dataarray(self.plotfunc, "x", "y")
        for ax in g.axes.flat:
            assert ax.has_data()

    def test_2d_function_and_method_signature_same(self):
        func_sig = inspect.getcallargs(self.plotfunc, self.darray)
        method_sig = inspect.getcallargs(self.plotmethod)
        del method_sig["_PlotMethods_obj"]
        del func_sig["darray"]
        assert func_sig == method_sig

    @pytest.mark.filterwarnings("ignore:tight_layout cannot")
    def test_convenient_facetgrid(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=["y", "x", "z"])
        g = self.plotfunc(d, x="x", y="y", col="z", col_wrap=2)

        assert_array_equal(g.axes.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axes):
            assert ax.has_data()
            if x == 0:
                assert "y" == ax.get_ylabel()
            else:
                assert "" == ax.get_ylabel()
            if y == 1:
                assert "x" == ax.get_xlabel()
            else:
                assert "" == ax.get_xlabel()

        # Infering labels
        g = self.plotfunc(d, col="z", col_wrap=2)
        assert_array_equal(g.axes.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axes):
            assert ax.has_data()
            if x == 0:
                assert "y" == ax.get_ylabel()
            else:
                assert "" == ax.get_ylabel()
            if y == 1:
                assert "x" == ax.get_xlabel()
            else:
                assert "" == ax.get_xlabel()

    @pytest.mark.filterwarnings("ignore:tight_layout cannot")
    def test_convenient_facetgrid_4d(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=["y", "x", "columns", "rows"])
        g = self.plotfunc(d, x="x", y="y", col="columns", row="rows")

        assert_array_equal(g.axes.shape, [3, 2])
        for ax in g.axes.flat:
            assert ax.has_data()

    @pytest.mark.filterwarnings("ignore:This figure includes")
    def test_facetgrid_map_only_appends_mappables(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=["y", "x", "columns", "rows"])
        g = self.plotfunc(d, x="x", y="y", col="columns", row="rows")

        expected = g._mappables

        g.map(lambda: plt.plot(1, 1))
        actual = g._mappables

        assert expected == actual

    def test_facetgrid_cmap(self):
        # Regression test for GH592
        data = np.random.random(size=(20, 25, 12)) + np.linspace(-3, 3, 12)
        d = DataArray(data, dims=["x", "y", "time"])
        fg = d.plot.pcolormesh(col="time")
        # check that all color limits are the same
        assert len({m.get_clim() for m in fg._mappables}) == 1
        # check that all colormaps are the same
        assert len({m.get_cmap().name for m in fg._mappables}) == 1

    def test_facetgrid_cbar_kwargs(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=["y", "x", "columns", "rows"])
        g = self.plotfunc(
            d,
            x="x",
            y="y",
            col="columns",
            row="rows",
            cbar_kwargs={"label": "test_label"},
        )

        # catch contour case
        if hasattr(g, "cbar"):
            assert g.cbar._label == "test_label"

    def test_facetgrid_no_cbar_ax(self):
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=["y", "x", "columns", "rows"])
        with pytest.raises(ValueError):
            self.plotfunc(d, x="x", y="y", col="columns", row="rows", cbar_ax=1)

    def test_cmap_and_color_both(self):
        with pytest.raises(ValueError):
            self.plotmethod(colors="k", cmap="RdBu")

    def test_2d_coord_with_interval(self):
        for dim in self.darray.dims:
            gp = self.darray.groupby_bins(dim, range(15), restore_coord_dims=True).mean(
                dim
            )
            for kind in ["imshow", "pcolormesh", "contourf", "contour"]:
                getattr(gp.plot, kind)()

    def test_colormap_error_norm_and_vmin_vmax(self):
        norm = mpl.colors.LogNorm(0.1, 1e1)

        with pytest.raises(ValueError):
            self.darray.plot(norm=norm, vmin=2)

        with pytest.raises(ValueError):
            self.darray.plot(norm=norm, vmax=2)


@pytest.mark.slow
class TestContourf(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contourf)

    @pytest.mark.slow
    def test_contourf_called(self):
        # Having both statements ensures the test works properly
        assert not self.contourf_called(self.darray.plot.imshow)
        assert self.contourf_called(self.darray.plot.contourf)

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        assert isinstance(artist, mpl.contour.QuadContourSet)

    @pytest.mark.slow
    def test_extend(self):
        artist = self.plotmethod()
        assert artist.extend == "neither"

        self.darray[0, 0] = -100
        self.darray[-1, -1] = 100
        artist = self.plotmethod(robust=True)
        assert artist.extend == "both"

        self.darray[0, 0] = 0
        self.darray[-1, -1] = 0
        artist = self.plotmethod(vmin=-0, vmax=10)
        assert artist.extend == "min"

        artist = self.plotmethod(vmin=-10, vmax=0)
        assert artist.extend == "max"

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x="x2d", y="y2d")
        # make sure labels came out ok
        ax = plt.gca()
        assert "x2d" == ax.get_xlabel()
        assert "y2d" == ax.get_ylabel()

    @pytest.mark.slow
    def test_levels(self):
        artist = self.plotmethod(levels=[-0.5, -0.4, 0.1])
        assert artist.extend == "both"

        artist = self.plotmethod(levels=3)
        assert artist.extend == "neither"


@pytest.mark.slow
class TestContour(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.contour)

    # matplotlib cmap.colors gives an rgbA ndarray
    # when seaborn is used, instead we get an rgb tuple
    @staticmethod
    def _color_as_tuple(c):
        return tuple(c[:3])

    def test_colors(self):

        # with single color, we don't want rgb array
        artist = self.plotmethod(colors="k")
        assert artist.cmap.colors[0] == "k"

        artist = self.plotmethod(colors=["k", "b"])
        assert self._color_as_tuple(artist.cmap.colors[1]) == (0.0, 0.0, 1.0)

        artist = self.darray.plot.contour(
            levels=[-0.5, 0.0, 0.5, 1.0], colors=["k", "r", "w", "b"]
        )
        assert self._color_as_tuple(artist.cmap.colors[1]) == (1.0, 0.0, 0.0)
        assert self._color_as_tuple(artist.cmap.colors[2]) == (1.0, 1.0, 1.0)
        # the last color is now under "over"
        assert self._color_as_tuple(artist.cmap._rgba_over) == (0.0, 0.0, 1.0)

    def test_colors_np_levels(self):

        # https://github.com/pydata/xarray/issues/3284
        levels = np.array([-0.5, 0.0, 0.5, 1.0])
        artist = self.darray.plot.contour(levels=levels, colors=["k", "r", "w", "b"])
        assert self._color_as_tuple(artist.cmap.colors[1]) == (1.0, 0.0, 0.0)
        assert self._color_as_tuple(artist.cmap.colors[2]) == (1.0, 1.0, 1.0)
        # the last color is now under "over"
        assert self._color_as_tuple(artist.cmap._rgba_over) == (0.0, 0.0, 1.0)

    def test_cmap_and_color_both(self):
        with pytest.raises(ValueError):
            self.plotmethod(colors="k", cmap="RdBu")

    def list_of_colors_in_cmap_raises_error(self):
        with raises_regex(ValueError, "list of colors"):
            self.plotmethod(cmap=["k", "b"])

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x="x2d", y="y2d")
        # make sure labels came out ok
        ax = plt.gca()
        assert "x2d" == ax.get_xlabel()
        assert "y2d" == ax.get_ylabel()

    def test_single_level(self):
        # this used to raise an error, but not anymore since
        # add_colorbar defaults to false
        self.plotmethod(levels=[0.1])
        self.plotmethod(levels=1)


class TestPcolormesh(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.pcolormesh)

    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        assert isinstance(artist, mpl.collections.QuadMesh)

    def test_everything_plotted(self):
        artist = self.plotmethod()
        assert artist.get_array().size == self.darray.size

    @pytest.mark.slow
    def test_2d_coord_names(self):
        self.plotmethod(x="x2d", y="y2d")
        # make sure labels came out ok
        ax = plt.gca()
        assert "x2d" == ax.get_xlabel()
        assert "y2d" == ax.get_ylabel()

    def test_dont_infer_interval_breaks_for_cartopy(self):
        # Regression for GH 781
        ax = plt.gca()
        # Simulate a Cartopy Axis
        setattr(ax, "projection", True)
        artist = self.plotmethod(x="x2d", y="y2d", ax=ax)
        assert isinstance(artist, mpl.collections.QuadMesh)
        # Let cartopy handle the axis limits and artist size
        assert artist.get_array().size <= self.darray.size


@pytest.mark.slow
class TestImshow(Common2dMixin, PlotTestCase):

    plotfunc = staticmethod(xplt.imshow)

    @pytest.mark.slow
    def test_imshow_called(self):
        # Having both statements ensures the test works properly
        assert not self.imshow_called(self.darray.plot.contourf)
        assert self.imshow_called(self.darray.plot.imshow)

    def test_xy_pixel_centered(self):
        self.darray.plot.imshow(yincrease=False)
        assert np.allclose([-0.5, 14.5], plt.gca().get_xlim())
        assert np.allclose([9.5, -0.5], plt.gca().get_ylim())

    def test_default_aspect_is_auto(self):
        self.darray.plot.imshow()
        assert "auto" == plt.gca().get_aspect()

    @pytest.mark.slow
    def test_cannot_change_mpl_aspect(self):

        with raises_regex(ValueError, "not available in xarray"):
            self.darray.plot.imshow(aspect="equal")

        # with numbers we fall back to fig control
        self.darray.plot.imshow(size=5, aspect=2)
        assert "auto" == plt.gca().get_aspect()
        assert tuple(plt.gcf().get_size_inches()) == (10, 5)

    @pytest.mark.slow
    def test_primitive_artist_returned(self):
        artist = self.plotmethod()
        assert isinstance(artist, mpl.image.AxesImage)

    @pytest.mark.slow
    @requires_seaborn
    def test_seaborn_palette_needs_levels(self):
        with pytest.raises(ValueError):
            self.plotmethod(cmap="husl")

    def test_2d_coord_names(self):
        with raises_regex(ValueError, "requires 1D coordinates"):
            self.plotmethod(x="x2d", y="y2d")

    def test_plot_rgb_image(self):
        DataArray(
            easy_array((10, 15, 3), start=0), dims=["y", "x", "band"]
        ).plot.imshow()
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgb_image_explicit(self):
        DataArray(
            easy_array((10, 15, 3), start=0), dims=["y", "x", "band"]
        ).plot.imshow(y="y", x="x", rgb="band")
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgb_faceted(self):
        DataArray(
            easy_array((2, 2, 10, 15, 3), start=0), dims=["a", "b", "y", "x", "band"]
        ).plot.imshow(row="a", col="b")
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgba_image_transposed(self):
        # We can handle the color axis being in any position
        DataArray(
            easy_array((4, 10, 15), start=0), dims=["band", "y", "x"]
        ).plot.imshow()

    def test_warns_ambigious_dim(self):
        arr = DataArray(easy_array((3, 3, 3)), dims=["y", "x", "band"])
        with pytest.warns(UserWarning):
            arr.plot.imshow()
        # but doesn't warn if dimensions specified
        arr.plot.imshow(rgb="band")
        arr.plot.imshow(x="x", y="y")

    def test_rgb_errors_too_many_dims(self):
        arr = DataArray(easy_array((3, 3, 3, 3)), dims=["y", "x", "z", "band"])
        with pytest.raises(ValueError):
            arr.plot.imshow(rgb="band")

    def test_rgb_errors_bad_dim_sizes(self):
        arr = DataArray(easy_array((5, 5, 5)), dims=["y", "x", "band"])
        with pytest.raises(ValueError):
            arr.plot.imshow(rgb="band")

    def test_normalize_rgb_imshow(self):
        for kwargs in (
            dict(vmin=-1),
            dict(vmax=2),
            dict(vmin=-1, vmax=1),
            dict(vmin=0, vmax=0),
            dict(vmin=0, robust=True),
            dict(vmax=-1, robust=True),
        ):
            da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
            arr = da.plot.imshow(**kwargs).get_array()
            assert 0 <= arr.min() <= arr.max() <= 1, kwargs

    def test_normalize_rgb_one_arg_error(self):
        da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
        # If passed one bound that implies all out of range, error:
        for kwargs in [dict(vmax=-1), dict(vmin=2)]:
            with pytest.raises(ValueError):
                da.plot.imshow(**kwargs)
        # If passed two that's just moving the range, *not* an error:
        for kwargs in [dict(vmax=-1, vmin=-1.2), dict(vmin=2, vmax=2.1)]:
            da.plot.imshow(**kwargs)

    def test_imshow_rgb_values_in_valid_range(self):
        da = DataArray(np.arange(75, dtype="uint8").reshape((5, 5, 3)))
        _, ax = plt.subplots()
        out = da.plot.imshow(ax=ax).get_array()
        assert out.dtype == np.uint8
        assert (out[..., :3] == da.values).all()  # Compare without added alpha

    @pytest.mark.filterwarnings("ignore:Several dimensions of this array")
    def test_regression_rgb_imshow_dim_size_one(self):
        # Regression: https://github.com/pydata/xarray/issues/1966
        da = DataArray(easy_array((1, 3, 3), start=0.0, stop=1.0))
        da.plot.imshow()

    def test_origin_overrides_xyincrease(self):
        da = DataArray(easy_array((3, 2)), coords=[[-2, 0, 2], [-1, 1]])
        da.plot.imshow(origin="upper")
        assert plt.xlim()[0] < 0
        assert plt.ylim()[1] < 0

        plt.clf()
        da.plot.imshow(origin="lower")
        assert plt.xlim()[0] < 0
        assert plt.ylim()[0] < 0


class TestFacetGrid(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        d = easy_array((10, 15, 3))
        self.darray = DataArray(d, dims=["y", "x", "z"], coords={"z": ["a", "b", "c"]})
        self.g = xplt.FacetGrid(self.darray, col="z")

    @pytest.mark.slow
    def test_no_args(self):
        self.g.map_dataarray(xplt.contourf, "x", "y")

        # Don't want colorbar labeled with 'None'
        alltxt = text_in_fig()
        assert "None" not in alltxt

        for ax in self.g.axes.flat:
            assert ax.has_data()

    @pytest.mark.slow
    def test_names_appear_somewhere(self):
        self.darray.name = "testvar"
        self.g.map_dataarray(xplt.contourf, "x", "y")
        for k, ax in zip("abc", self.g.axes.flat):
            assert "z = {}".format(k) == ax.get_title()

        alltxt = text_in_fig()
        assert self.darray.name in alltxt
        for label in ["x", "y"]:
            assert label in alltxt

    @pytest.mark.slow
    def test_text_not_super_long(self):
        self.darray.coords["z"] = [100 * letter for letter in "abc"]
        g = xplt.FacetGrid(self.darray, col="z")
        g.map_dataarray(xplt.contour, "x", "y")
        alltxt = text_in_fig()
        maxlen = max(len(txt) for txt in alltxt)
        assert maxlen < 50

        t0 = g.axes[0, 0].get_title()
        assert t0.endswith("...")

    @pytest.mark.slow
    def test_colorbar(self):
        vmin = self.darray.values.min()
        vmax = self.darray.values.max()
        expected = np.array((vmin, vmax))

        self.g.map_dataarray(xplt.imshow, "x", "y")

        for image in plt.gcf().findobj(mpl.image.AxesImage):
            clim = np.array(image.get_clim())
            assert np.allclose(expected, clim)

        assert 1 == len(find_possible_colorbars())

    @pytest.mark.slow
    def test_empty_cell(self):
        g = xplt.FacetGrid(self.darray, col="z", col_wrap=2)
        g.map_dataarray(xplt.imshow, "x", "y")

        bottomright = g.axes[-1, -1]
        assert not bottomright.has_data()
        assert not bottomright.get_visible()

    @pytest.mark.slow
    def test_norow_nocol_error(self):
        with raises_regex(ValueError, r"[Rr]ow"):
            xplt.FacetGrid(self.darray)

    @pytest.mark.slow
    def test_groups(self):
        self.g.map_dataarray(xplt.imshow, "x", "y")
        upperleft_dict = self.g.name_dicts[0, 0]
        upperleft_array = self.darray.loc[upperleft_dict]
        z0 = self.darray.isel(z=0)

        assert_equal(upperleft_array, z0)

    @pytest.mark.slow
    def test_float_index(self):
        self.darray.coords["z"] = [0.1, 0.2, 0.4]
        g = xplt.FacetGrid(self.darray, col="z")
        g.map_dataarray(xplt.imshow, "x", "y")

    @pytest.mark.slow
    def test_nonunique_index_error(self):
        self.darray.coords["z"] = [0.1, 0.2, 0.2]
        with raises_regex(ValueError, r"[Uu]nique"):
            xplt.FacetGrid(self.darray, col="z")

    @pytest.mark.slow
    def test_robust(self):
        z = np.zeros((20, 20, 2))
        darray = DataArray(z, dims=["y", "x", "z"])
        darray[:, :, 1] = 1
        darray[2, 0, 0] = -1000
        darray[3, 0, 0] = 1000
        g = xplt.FacetGrid(darray, col="z")
        g.map_dataarray(xplt.imshow, "x", "y", robust=True)

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
        assert largest < 21

    @pytest.mark.slow
    def test_can_set_vmin_vmax(self):
        vmin, vmax = 50.0, 1000.0
        expected = np.array((vmin, vmax))
        self.g.map_dataarray(xplt.imshow, "x", "y", vmin=vmin, vmax=vmax)

        for image in plt.gcf().findobj(mpl.image.AxesImage):
            clim = np.array(image.get_clim())
            assert np.allclose(expected, clim)

    @pytest.mark.slow
    def test_can_set_norm(self):
        norm = mpl.colors.SymLogNorm(0.1)
        self.g.map_dataarray(xplt.imshow, "x", "y", norm=norm)
        for image in plt.gcf().findobj(mpl.image.AxesImage):
            assert image.norm is norm

    @pytest.mark.slow
    def test_figure_size(self):

        assert_array_equal(self.g.fig.get_size_inches(), (10, 3))

        g = xplt.FacetGrid(self.darray, col="z", size=6)
        assert_array_equal(g.fig.get_size_inches(), (19, 6))

        g = self.darray.plot.imshow(col="z", size=6)
        assert_array_equal(g.fig.get_size_inches(), (19, 6))

        g = xplt.FacetGrid(self.darray, col="z", size=4, aspect=0.5)
        assert_array_equal(g.fig.get_size_inches(), (7, 4))

        g = xplt.FacetGrid(self.darray, col="z", figsize=(9, 4))
        assert_array_equal(g.fig.get_size_inches(), (9, 4))

        with raises_regex(ValueError, "cannot provide both"):
            g = xplt.plot(self.darray, row=2, col="z", figsize=(6, 4), size=6)

        with raises_regex(ValueError, "Can't use"):
            g = xplt.plot(self.darray, row=2, col="z", ax=plt.gca(), size=6)

    @pytest.mark.slow
    def test_num_ticks(self):
        nticks = 99
        maxticks = nticks + 1
        self.g.map_dataarray(xplt.imshow, "x", "y")
        self.g.set_ticks(max_xticks=nticks, max_yticks=nticks)

        for ax in self.g.axes.flat:
            xticks = len(ax.get_xticks())
            yticks = len(ax.get_yticks())
            assert xticks <= maxticks
            assert yticks <= maxticks
            assert xticks >= nticks / 2.0
            assert yticks >= nticks / 2.0

    @pytest.mark.slow
    def test_map(self):
        assert self.g._finalized is False
        self.g.map(plt.contourf, "x", "y", Ellipsis)
        assert self.g._finalized is True
        self.g.map(lambda: None)

    @pytest.mark.slow
    def test_map_dataset(self):
        g = xplt.FacetGrid(self.darray.to_dataset(name="foo"), col="z")
        g.map(plt.contourf, "x", "y", "foo")

        alltxt = text_in_fig()
        for label in ["x", "y"]:
            assert label in alltxt
        # everything has a label
        assert "None" not in alltxt

        # colorbar can't be inferred automatically
        assert "foo" not in alltxt
        assert 0 == len(find_possible_colorbars())

        g.add_colorbar(label="colors!")
        assert "colors!" in text_in_fig()
        assert 1 == len(find_possible_colorbars())

    @pytest.mark.slow
    def test_set_axis_labels(self):
        g = self.g.map_dataarray(xplt.contourf, "x", "y")
        g.set_axis_labels("longitude", "latitude")
        alltxt = text_in_fig()
        for label in ["longitude", "latitude"]:
            assert label in alltxt

    @pytest.mark.slow
    def test_facetgrid_colorbar(self):
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=["y", "x", "z"], name="foo")

        d.plot.imshow(x="x", y="y", col="z")
        assert 1 == len(find_possible_colorbars())

        d.plot.imshow(x="x", y="y", col="z", add_colorbar=True)
        assert 1 == len(find_possible_colorbars())

        d.plot.imshow(x="x", y="y", col="z", add_colorbar=False)
        assert 0 == len(find_possible_colorbars())

    @pytest.mark.slow
    def test_facetgrid_polar(self):
        # test if polar projection in FacetGrid does not raise an exception
        self.darray.plot.pcolormesh(
            col="z", subplot_kws=dict(projection="polar"), sharex=False, sharey=False
        )


@pytest.mark.filterwarnings("ignore:tight_layout cannot")
class TestFacetGrid4d(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        a = easy_array((10, 15, 3, 2))
        darray = DataArray(a, dims=["y", "x", "col", "row"])
        darray.coords["col"] = np.array(
            ["col" + str(x) for x in darray.coords["col"].values]
        )
        darray.coords["row"] = np.array(
            ["row" + str(x) for x in darray.coords["row"].values]
        )

        self.darray = darray

    @pytest.mark.slow
    def test_default_labels(self):
        g = xplt.FacetGrid(self.darray, col="col", row="row")
        assert (2, 3) == g.axes.shape

        g.map_dataarray(xplt.imshow, "x", "y")

        # Rightmost column should be labeled
        for label, ax in zip(self.darray.coords["row"].values, g.axes[:, -1]):
            assert substring_in_axes(label, ax)

        # Top row should be labeled
        for label, ax in zip(self.darray.coords["col"].values, g.axes[0, :]):
            assert substring_in_axes(label, ax)


@pytest.mark.filterwarnings("ignore:tight_layout cannot")
class TestFacetedLinePlotsLegend(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.darray = xr.tutorial.scatter_example_dataset()

    def test_legend_labels(self):
        fg = self.darray.A.plot.line(col="x", row="w", hue="z")
        all_legend_labels = [t.get_text() for t in fg.figlegend.texts]
        # labels in legend should be ['0', '1', '2', '3']
        assert sorted(all_legend_labels) == ["0", "1", "2", "3"]


@pytest.mark.filterwarnings("ignore:tight_layout cannot")
class TestFacetedLinePlots(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.darray = DataArray(
            np.random.randn(10, 6, 3, 4),
            dims=["hue", "x", "col", "row"],
            coords=[range(10), range(6), range(3), ["A", "B", "C", "C++"]],
            name="Cornelius Ortega the 1st",
        )

        self.darray.hue.name = "huename"
        self.darray.hue.attrs["units"] = "hunits"
        self.darray.x.attrs["units"] = "xunits"
        self.darray.col.attrs["units"] = "colunits"
        self.darray.row.attrs["units"] = "rowunits"

    def test_facetgrid_shape(self):
        g = self.darray.plot(row="row", col="col", hue="hue")
        assert g.axes.shape == (len(self.darray.row), len(self.darray.col))

        g = self.darray.plot(row="col", col="row", hue="hue")
        assert g.axes.shape == (len(self.darray.col), len(self.darray.row))

    def test_unnamed_args(self):
        g = self.darray.plot.line("o--", row="row", col="col", hue="hue")
        lines = [
            q for q in g.axes.flat[0].get_children() if isinstance(q, mpl.lines.Line2D)
        ]
        # passing 'o--' as argument should set marker and linestyle
        assert lines[0].get_marker() == "o"
        assert lines[0].get_linestyle() == "--"

    def test_default_labels(self):
        g = self.darray.plot(row="row", col="col", hue="hue")
        # Rightmost column should be labeled
        for label, ax in zip(self.darray.coords["row"].values, g.axes[:, -1]):
            assert substring_in_axes(label, ax)

        # Top row should be labeled
        for label, ax in zip(self.darray.coords["col"].values, g.axes[0, :]):
            assert substring_in_axes(str(label), ax)

        # Leftmost column should have array name
        for ax in g.axes[:, 0]:
            assert substring_in_axes(self.darray.name, ax)

    def test_test_empty_cell(self):
        g = self.darray.isel(row=1).drop("row").plot(col="col", hue="hue", col_wrap=2)
        bottomright = g.axes[-1, -1]
        assert not bottomright.has_data()
        assert not bottomright.get_visible()

    def test_set_axis_labels(self):
        g = self.darray.plot(row="row", col="col", hue="hue")
        g.set_axis_labels("longitude", "latitude")
        alltxt = text_in_fig()

        assert "longitude" in alltxt
        assert "latitude" in alltxt

    def test_axes_in_faceted_plot(self):
        with pytest.raises(ValueError):
            self.darray.plot.line(row="row", col="col", x="x", ax=plt.axes())

    def test_figsize_and_size(self):
        with pytest.raises(ValueError):
            self.darray.plot.line(row="row", col="col", x="x", size=3, figsize=4)

    def test_wrong_num_of_dimensions(self):
        with pytest.raises(ValueError):
            self.darray.plot(row="row", hue="hue")
            self.darray.plot.line(row="row", hue="hue")


@requires_matplotlib
class TestDatasetScatterPlots(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        das = [
            DataArray(
                np.random.randn(3, 3, 4, 4),
                dims=["x", "row", "col", "hue"],
                coords=[range(k) for k in [3, 3, 4, 4]],
            )
            for _ in [1, 2]
        ]
        ds = Dataset({"A": das[0], "B": das[1]})
        ds.hue.name = "huename"
        ds.hue.attrs["units"] = "hunits"
        ds.x.attrs["units"] = "xunits"
        ds.col.attrs["units"] = "colunits"
        ds.row.attrs["units"] = "rowunits"
        ds.A.attrs["units"] = "Aunits"
        ds.B.attrs["units"] = "Bunits"
        self.ds = ds

    @pytest.mark.parametrize(
        "add_guide, hue_style, legend, colorbar",
        [
            (None, None, False, True),
            (False, None, False, False),
            (True, None, False, True),
            (True, "continuous", False, True),
            (False, "discrete", False, False),
            (True, "discrete", True, False),
        ],
    )
    def test_add_guide(self, add_guide, hue_style, legend, colorbar):

        meta_data = _infer_meta_data(
            self.ds, x="A", y="B", hue="hue", hue_style=hue_style, add_guide=add_guide
        )
        assert meta_data["add_legend"] is legend
        assert meta_data["add_colorbar"] is colorbar

    def test_facetgrid_shape(self):
        g = self.ds.plot.scatter(x="A", y="B", row="row", col="col")
        assert g.axes.shape == (len(self.ds.row), len(self.ds.col))

        g = self.ds.plot.scatter(x="A", y="B", row="col", col="row")
        assert g.axes.shape == (len(self.ds.col), len(self.ds.row))

    def test_default_labels(self):
        g = self.ds.plot.scatter("A", "B", row="row", col="col", hue="hue")

        # Top row should be labeled
        for label, ax in zip(self.ds.coords["col"].values, g.axes[0, :]):
            assert substring_in_axes(str(label), ax)

        # Bottom row should have name of x array name and units
        for ax in g.axes[-1, :]:
            assert ax.get_xlabel() == "A [Aunits]"

        # Leftmost column should have name of y array name and units
        for ax in g.axes[:, 0]:
            assert ax.get_ylabel() == "B [Bunits]"

    def test_axes_in_faceted_plot(self):
        with pytest.raises(ValueError):
            self.ds.plot.scatter(x="A", y="B", row="row", ax=plt.axes())

    def test_figsize_and_size(self):
        with pytest.raises(ValueError):
            self.ds.plot.scatter(x="A", y="B", row="row", size=3, figsize=4)

    @pytest.mark.parametrize(
        "x, y, hue_style, add_guide",
        [
            ("A", "B", "something", True),
            ("A", "B", "discrete", True),
            ("A", "B", None, True),
            ("A", "The Spanish Inquisition", None, None),
            ("The Spanish Inquisition", "B", None, True),
        ],
    )
    def test_bad_args(self, x, y, hue_style, add_guide):
        with pytest.raises(ValueError):
            self.ds.plot.scatter(x, y, hue_style=hue_style, add_guide=add_guide)

    @pytest.mark.xfail(reason="datetime,timedelta hue variable not supported.")
    @pytest.mark.parametrize("hue_style", ["discrete", "continuous"])
    def test_datetime_hue(self, hue_style):
        ds2 = self.ds.copy()
        ds2["hue"] = pd.date_range("2000-1-1", periods=4)
        ds2.plot.scatter(x="A", y="B", hue="hue", hue_style=hue_style)

        ds2["hue"] = pd.timedelta_range("-1D", periods=4, freq="D")
        ds2.plot.scatter(x="A", y="B", hue="hue", hue_style=hue_style)

    def test_facetgrid_hue_style(self):
        # Can't move this to pytest.mark.parametrize because py35-min
        # doesn't have mpl.
        for hue_style, map_type in zip(
            ["discrete", "continuous"], [list, mpl.collections.PathCollection]
        ):
            g = self.ds.plot.scatter(
                x="A", y="B", row="row", col="col", hue="hue", hue_style=hue_style
            )
            # for 'discrete' a list is appended to _mappables
            # for 'continuous', should be single PathCollection
            assert isinstance(g._mappables[-1], map_type)

    @pytest.mark.parametrize(
        "x, y, hue, markersize", [("A", "B", "x", "col"), ("x", "row", "A", "B")]
    )
    def test_scatter(self, x, y, hue, markersize):
        self.ds.plot.scatter(x, y, hue=hue, markersize=markersize)

    def test_non_numeric_legend(self):
        ds2 = self.ds.copy()
        ds2["hue"] = ["a", "b", "c", "d"]
        lines = ds2.plot.scatter(x="A", y="B", hue="hue")
        # should make a discrete legend
        assert lines[0].axes.legend_ is not None
        # and raise an error if explicitly not allowed to do so
        with pytest.raises(ValueError):
            ds2.plot.scatter(x="A", y="B", hue="hue", hue_style="continuous")

    def test_add_legend_by_default(self):
        sc = self.ds.plot.scatter(x="A", y="B", hue="hue")
        assert len(sc.figure.axes) == 2


class TestDatetimePlot(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Create a DataArray with a time-axis that contains datetime objects.
        """
        month = np.arange(1, 13, 1)
        data = np.sin(2 * np.pi * month / 12.0)

        darray = DataArray(data, dims=["time"])
        darray.coords["time"] = np.array([datetime(2017, m, 1) for m in month])

        self.darray = darray

    def test_datetime_line_plot(self):
        # test if line plot raises no Exception
        self.darray.plot.line()


@requires_nc_time_axis
@requires_cftime
class TestCFDatetimePlot(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Create a DataArray with a time-axis that contains cftime.datetime
        objects.
        """
        # case for 1d array
        data = np.random.rand(4, 12)
        time = xr.cftime_range(start="2017", periods=12, freq="1M", calendar="noleap")
        darray = DataArray(data, dims=["x", "time"])
        darray.coords["time"] = time

        self.darray = darray

    def test_cfdatetime_line_plot(self):
        self.darray.isel(x=0).plot.line()

    def test_cfdatetime_pcolormesh_plot(self):
        self.darray.plot.pcolormesh()

    def test_cfdatetime_contour_plot(self):
        self.darray.plot.contour()


@requires_cftime
@pytest.mark.skipif(has_nc_time_axis, reason="nc_time_axis is installed")
class TestNcAxisNotInstalled(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Create a DataArray with a time-axis that contains cftime.datetime
        objects.
        """
        month = np.arange(1, 13, 1)
        data = np.sin(2 * np.pi * month / 12.0)
        darray = DataArray(data, dims=["time"])
        darray.coords["time"] = xr.cftime_range(
            start="2017", periods=12, freq="1M", calendar="noleap"
        )

        self.darray = darray

    def test_ncaxis_notinstalled_line_plot(self):
        with raises_regex(ImportError, "optional `nc-time-axis`"):
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
        _color_palette("Blues", 4)
    assert len(record) == 0


test_da_list = [
    DataArray(easy_array((10,))),
    DataArray(easy_array((10, 3))),
    DataArray(easy_array((10, 3, 2))),
]


@requires_matplotlib
class TestAxesKwargs:
    @pytest.mark.parametrize("da", test_da_list)
    @pytest.mark.parametrize("xincrease", [True, False])
    def test_xincrease_kwarg(self, da, xincrease):
        plt.clf()
        da.plot(xincrease=xincrease)
        assert plt.gca().xaxis_inverted() == (not xincrease)

    @pytest.mark.parametrize("da", test_da_list)
    @pytest.mark.parametrize("yincrease", [True, False])
    def test_yincrease_kwarg(self, da, yincrease):
        plt.clf()
        da.plot(yincrease=yincrease)
        assert plt.gca().yaxis_inverted() == (not yincrease)

    @pytest.mark.parametrize("da", test_da_list)
    @pytest.mark.parametrize("xscale", ["linear", "log", "logit", "symlog"])
    def test_xscale_kwarg(self, da, xscale):
        plt.clf()
        da.plot(xscale=xscale)
        assert plt.gca().get_xscale() == xscale

    @pytest.mark.parametrize(
        "da", [DataArray(easy_array((10,))), DataArray(easy_array((10, 3)))]
    )
    @pytest.mark.parametrize("yscale", ["linear", "log", "logit", "symlog"])
    def test_yscale_kwarg(self, da, yscale):
        plt.clf()
        da.plot(yscale=yscale)
        assert plt.gca().get_yscale() == yscale

    @pytest.mark.parametrize("da", test_da_list)
    def test_xlim_kwarg(self, da):
        plt.clf()
        expected = (0.0, 1000.0)
        da.plot(xlim=[0, 1000])
        assert plt.gca().get_xlim() == expected

    @pytest.mark.parametrize("da", test_da_list)
    def test_ylim_kwarg(self, da):
        plt.clf()
        da.plot(ylim=[0, 1000])
        expected = (0.0, 1000.0)
        assert plt.gca().get_ylim() == expected

    @pytest.mark.parametrize("da", test_da_list)
    def test_xticks_kwarg(self, da):
        plt.clf()
        da.plot(xticks=np.arange(5))
        expected = np.arange(5).tolist()
        assert np.all(plt.gca().get_xticks() == expected)

    @pytest.mark.parametrize("da", test_da_list)
    def test_yticks_kwarg(self, da):
        plt.clf()
        da.plot(yticks=np.arange(5))
        expected = np.arange(5)
        assert np.all(plt.gca().get_yticks() == expected)
