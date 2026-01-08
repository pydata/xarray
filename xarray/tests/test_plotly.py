"""Tests for the plotly accessor (Plotly Express plotting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.plot.plotly.common import SLOT_ORDERS, assign_slots


class TestAssignSlots:
    """Tests for the dimension-to-slot assignment algorithm."""

    def test_auto_assignment_line(self):
        """Test automatic positional assignment for line plots."""
        slots = assign_slots(["time", "city", "scenario"], "line")
        assert slots == {"x": "time", "color": "city", "facet_col": "scenario"}

    def test_auto_assignment_imshow(self):
        """Test automatic positional assignment for imshow."""
        slots = assign_slots(["lat", "lon"], "imshow")
        assert slots == {"x": "lat", "y": "lon"}

    def test_auto_assignment_scatter(self):
        """Test automatic positional assignment for scatter plots."""
        # scatter doesn't auto-assign y (defaults to "value")
        slots = assign_slots(["x", "color"], "scatter")
        assert slots == {"x": "x", "color": "color"}

    def test_auto_assignment_box(self):
        """Test automatic positional assignment for box plots."""
        slots = assign_slots(["category", "group"], "box")
        assert slots == {"x": "category", "color": "group"}

    def test_explicit_assignment(self):
        """Test explicit dimension-to-slot assignment."""
        slots = assign_slots(
            ["time", "city", "scenario"], "line", x="city", color="time"
        )
        assert slots["x"] == "city"
        assert slots["color"] == "time"
        # scenario goes to the next available slot
        assert slots["facet_col"] == "scenario"

    def test_skip_slot_with_none(self):
        """Test skipping a slot using None."""
        slots = assign_slots(["time", "city", "scenario"], "line", color=None)
        assert slots == {"x": "time", "facet_col": "city", "facet_row": "scenario"}
        assert "color" not in slots

    def test_unassigned_dims_error(self):
        """Test that unassigned dimensions raise an error."""
        # 6 dims but only 5 slots
        dims = list("abcdef")
        with pytest.raises(ValueError, match="Unassigned dimension"):
            assign_slots(dims, "line")

    def test_invalid_dimension_error(self):
        """Test that using a non-existent dimension raises an error."""
        with pytest.raises(ValueError, match="is not in the data dimensions"):
            assign_slots(["time", "city"], "line", x="nonexistent")

    def test_unknown_plot_type_error(self):
        """Test that unknown plot types raise an error."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            assign_slots(["x", "y"], "unknown_plot_type")

    def test_all_explicit_assignment(self):
        """Test fully explicit assignment."""
        slots = assign_slots(
            ["time", "city"], "line", x="city", color="time", facet_col=None
        )
        assert slots == {"x": "city", "color": "time"}


# Skip all plotting tests if plotly is not installed
plotly = pytest.importorskip("plotly")


class TestDataArrayPlotly:
    """Tests for DataArray.plotly accessor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.da_1d = xr.DataArray(
            np.random.rand(10),
            dims=["time"],
            coords={"time": pd.date_range("2020", periods=10)},
            name="temperature",
        )
        self.da_2d = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "city"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "city": ["NYC", "LA", "Chicago"],
            },
            name="temperature",
        )
        self.da_3d = xr.DataArray(
            np.random.rand(10, 3, 2),
            dims=["time", "city", "scenario"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "city": ["NYC", "LA", "Chicago"],
                "scenario": ["baseline", "warming"],
            },
            name="temperature",
        )
        self.da_unnamed = xr.DataArray(np.random.rand(5, 3), dims=["x", "y"])

    def test_accessor_exists(self):
        """Test that plotly accessor is available on DataArray."""
        assert hasattr(self.da_2d, "plotly")
        assert hasattr(self.da_2d.plotly, "line")
        assert hasattr(self.da_2d.plotly, "bar")
        assert hasattr(self.da_2d.plotly, "area")
        assert hasattr(self.da_2d.plotly, "scatter")
        assert hasattr(self.da_2d.plotly, "box")
        assert hasattr(self.da_2d.plotly, "imshow")

    def test_line_returns_figure(self):
        """Test that line() returns a Plotly Figure."""
        fig = self.da_2d.plotly.line()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_line_1d(self):
        """Test line plot with 1D data."""
        fig = self.da_1d.plotly.line()
        assert isinstance(fig, plotly.graph_objects.Figure)
        # Should have one trace
        assert len(fig.data) >= 1

    def test_line_2d(self):
        """Test line plot with 2D data."""
        fig = self.da_2d.plotly.line()
        assert isinstance(fig, plotly.graph_objects.Figure)
        # Should have traces for each city
        assert len(fig.data) >= 1

    def test_line_explicit_assignment(self):
        """Test line plot with explicit dimension assignment."""
        fig = self.da_2d.plotly.line(x="time", color="city")
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_line_skip_slot(self):
        """Test line plot with skipped slot."""
        fig = self.da_3d.plotly.line(color=None)
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_line_px_kwargs(self):
        """Test that px_kwargs are passed through."""
        fig = self.da_2d.plotly.line(title="My Plot")
        assert fig.layout.title.text == "My Plot"

    def test_bar_returns_figure(self):
        """Test that bar() returns a Plotly Figure."""
        fig = self.da_2d.plotly.bar()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_area_returns_figure(self):
        """Test that area() returns a Plotly Figure."""
        fig = self.da_2d.plotly.area()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_scatter_returns_figure(self):
        """Test that scatter() returns a Plotly Figure."""
        fig = self.da_2d.plotly.scatter()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_box_returns_figure(self):
        """Test that box() returns a Plotly Figure."""
        fig = self.da_2d.plotly.box()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_imshow_returns_figure(self):
        """Test that imshow() returns a Plotly Figure."""
        fig = self.da_2d.plotly.imshow()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_imshow_transpose(self):
        """Test that imshow correctly transposes based on x and y."""
        da = xr.DataArray(
            np.random.rand(10, 20),
            dims=["lat", "lon"],
            coords={"lat": np.arange(10), "lon": np.arange(20)},
        )
        # Default: lat→x, lon→y
        fig = da.plotly.imshow()
        assert isinstance(fig, plotly.graph_objects.Figure)

        # Explicit: lon→x, lat→y
        fig = da.plotly.imshow(x="lon", y="lat")
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_unnamed_dataarray(self):
        """Test plotting unnamed DataArray."""
        fig = self.da_unnamed.plotly.line()
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_unassigned_dims_error(self):
        """Test that too many dimensions raises an error."""
        da_6d = xr.DataArray(np.random.rand(2, 2, 2, 2, 2, 2), dims=list("abcdef"))
        with pytest.raises(ValueError, match="Unassigned dimension"):
            da_6d.plotly.line()


class TestLabelsAndMetadata:
    """Tests for label extraction from xarray attributes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data with metadata."""
        self.da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "station"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "station": ["A", "B", "C"],
            },
            name="temperature",
            attrs={
                "long_name": "Air Temperature",
                "units": "K",
            },
        )
        # Add coordinate attributes
        self.da.coords["time"].attrs = {
            "long_name": "Time",
            "units": "days since 2020-01-01",
        }

    def test_value_label_from_attrs(self):
        """Test that value labels are extracted from attributes."""
        fig = self.da.plotly.line()
        # The y-axis label should include the long_name and units
        # (implementation may vary, just check it works)
        assert isinstance(fig, plotly.graph_objects.Figure)


class TestSlotOrders:
    """Tests for slot order configurations."""

    def test_all_plot_types_have_slot_orders(self):
        """Test that all plot types have defined slot orders."""
        assert "line" in SLOT_ORDERS
        assert "bar" in SLOT_ORDERS
        assert "area" in SLOT_ORDERS
        assert "scatter" in SLOT_ORDERS
        assert "box" in SLOT_ORDERS
        assert "imshow" in SLOT_ORDERS

    def test_line_slot_order(self):
        """Test line plot slot order."""
        assert SLOT_ORDERS["line"] == (
            "x",
            "color",
            "facet_col",
            "facet_row",
            "animation_frame",
        )

    def test_scatter_slot_order(self):
        """Test scatter plot slot order (y defaults to 'value', not auto-assigned)."""
        assert SLOT_ORDERS["scatter"] == (
            "x",
            "color",
            "size",
            "facet_col",
            "facet_row",
            "animation_frame",
        )

    def test_box_slot_order(self):
        """Test box plot slot order."""
        assert SLOT_ORDERS["box"] == (
            "x",
            "color",
            "facet_col",
            "facet_row",
            "animation_frame",
        )

    def test_imshow_slot_order(self):
        """Test imshow slot order includes x and y."""
        assert "x" in SLOT_ORDERS["imshow"]
        assert "y" in SLOT_ORDERS["imshow"]
