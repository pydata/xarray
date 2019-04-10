import numpy as np
import numpy.testing as npt
import pytest

from xarray import DataArray
from . import requires_animatplot

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

# TODO should check that matplotlib >= 2.2 is present first?
try:
    import animatplot as amp
except ImportError:
    pass

from xarray.plot.animate import _create_timeline
import xarray.plot.animate


@requires_animatplot
class TestTimeline:
    def test_coord_timeline(self):
        da = DataArray([1, 2, 3],
                       coords={'duration': ('time', [0.1, 0.2, 0.3])},
                       dims='time')
        da.coords['duration'].attrs['units'] = 's'
        timeline = _create_timeline(da, animate='duration', fps=5)

        assert isinstance(timeline, amp.animation.Timeline)
        assert len(timeline) == len(da.coords['duration'])
        assert timeline.units == ' [s]'
        npt.assert_equal(timeline.t, da.coords['duration'].values)
        assert timeline.fps == 5

    def test_dim_timeline(self):
        da = DataArray([10, 20], dims='Time')
        timeline = _create_timeline(da, animate='Time', fps=5)

        assert isinstance(timeline, amp.animation.Timeline)
        assert len(timeline) == da.sizes['Time']
        assert timeline.units == ''
        npt.assert_equal(timeline.t, np.array([0, 1]))
        assert timeline.fps == 5

    def test_datetimeline(self):
        dates = np.array(['2000-01-01', '2000-01-02', '2000-01-03'],
                         dtype=np.datetime64)
        da = DataArray([1, 2, 3],
                       coords={'date': ('time', dates)}, dims='time')
        timeline = _create_timeline(da, animate='date', fps=5)

        assert str(timeline.t[0]) == '2000-01-01 00:00:00'


@requires_animatplot
class TestAnimateLine:
    @pytest.fixture(autouse=True)
    def setUp(self):
        d = np.array([[0.0, 1.1, 0.0, 2],
                      [0.1, 1.3, 0.2, 2.1],
                      [0.1, 1.4, 0.3, 2.2],
                      [0.2, 1.3, 0.2, 2.3],
                      [0.1, 1.2, 0.2, 2.2]])
        coords = {'time': 10 * np.arange(d.shape[0]),
                  'position': 0.1 * np.arange(d.shape[1])}
        self.darray = DataArray(d, name='height',
                                coords=coords,
                                dims=('time', 'position'),
                                attrs={'units': 'm'})
        self.darray.time.attrs['units'] = 's'
        self.darray.position.attrs['units'] = 'cm'

    def test_animate_single_line_classes(self):
        anim = self.darray.plot(animate='time')
        assert isinstance(anim, amp.animation.Animation)

        line_block, title_block = anim.blocks

        assert isinstance(line_block, amp.blocks.Line)
        assert isinstance(title_block, amp.blocks.Title)

    def test_animate_single_line_data(self):
        line_block, title_block = self.darray.plot(animate='time').blocks

        assert len(line_block) == 5
        assert len(line_block) == len(title_block)

        npt.assert_equal(line_block.y, self.darray.transpose().values)
        npt.assert_equal(line_block.x[:, 0],
                         self.darray.coords['position'].values)

    def test_animate_single_line_text(self):
        anim = self.darray.plot(animate='time')
        line_block, title_block = anim.blocks

        assert title_block.titles[0] == 'time = 0'
        assert line_block.ax.get_xlabel() == 'position [cm]'
        assert anim.timeline.units == ' [s]'

    def test_animate_single_line_axes(self):
        line_block, title_block = self.darray.plot(animate='time').blocks

        # Check current axes is the plot (not the timeline etc.)
        assert plt.gca() is line_block.ax

    def test_animate_as_function(self):
        anim = xarray.plot.animate.line(self.darray, animate='time')
        assert isinstance(anim, amp.animation.Animation)

    def test_animate_as_argument(self):
        anim = self.darray.plot(animate='time')
        assert isinstance(anim, amp.animation.Animation)

        anim = self.darray.plot.line(animate='time')
        assert isinstance(anim, amp.animation.Animation)
