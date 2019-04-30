from functools import partial

import numpy as np
import numpy.testing as npt
import pytest

import xarray as xr
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

from .test_plot import PlotTestCase, easy_array

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


@pytest.fixture
def linedata():
    dat1 = np.array([[0.0, 1.1, 0.0, 2],
                     [0.1, 1.3, 0.2, 2.1],
                     [0.1, 1.4, 0.3, 2.2],
                     [0.2, 1.3, 0.2, 2.3],
                     [0.1, 1.2, 0.2, 2.2]])
    dat2 = np.array([[0.0, 1.1, 0.0, 2],
                     [0.1, 1.3, 0.2, 2.1],
                     [0.1, 1.4, 0.3, 2.2],
                     [0.2, 1.3, 0.2, 2.3],
                     [0.1, 1.2, 0.2, 2.2]])
    das = []
    for data in [dat1, dat2]:
        coords = {'time': 10 * np.arange(data.shape[0]),
                  'position': 0.1 * np.arange(data.shape[1])}
        da = DataArray(data, name='height', coords=coords,
                       dims=('time', 'position'), attrs={'units': 'm'})
        da.time.attrs['units'] = 's'
        da.position.attrs['units'] = 'cm'

        das.append(da)

    player = DataArray(name='player', data=['Tom', 'Bhavin'], dims='player')
    return xr.concat(das, dim=player)


@requires_animatplot
class TestAnimateLine(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self, linedata):
        self.darray = linedata.sel(player='Tom')

    def test_2d_animated_line_accepts_x_kw(self):
        self.darray.plot.line(x='position', animate='time')
        assert plt.gca().get_xlabel() == 'position [cm]'
        plt.cla()
        self.darray.plot.line(x='time', animate='position')
        assert plt.gca().get_xlabel() == 'time [s]'

    @pytest.mark.skip
    def test_2d_animated_line_accepts_y_kw(self):
        self.darray.plot.line(y='position', animate='time')
        assert plt.gca().get_ylabel() == 'position [cm]'
        plt.cla()
        self.darray.plot.line(y='time', animate='position')
        assert plt.gca().get_ylabel() == 'time [s]'

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

        assert title_block.titles[0] == 'time = 0, player = Tom'
        assert line_block.ax.get_xlabel() == 'position [cm]'
        assert anim.timeline.units == ' [s]'

    # TODO test that omitting title block is handled gracefully
    @pytest.mark.skip
    def test_no_labels(self):
        ...

    def test_can_pass_in_axis(self):
        self.pass_in_axis(partial(self.darray.plot, animate='time'))

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


@pytest.mark.xfail(reason="np.splitting the y data doesn't work for step plots"
                          "because they have lists of arrays for some reason")
@requires_animatplot
class TestAnimateStep(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.darray = DataArray(easy_array((4, 5, 6)))

    def test_coord_with_interval_step(self):
        bins = [-1, 0, 1, 2]
        da = self.darray.groupby_bins('dim_0', bins).mean(xr.ALL_DIMS)
        da = xr.concat([da, da * 2, da * 1.7], dim='new_dim')

        anim = da.plot.step(animate='new_dim')
        assert len(plt.gca().lines[0].get_xdata()) == ((len(bins) - 1) * 2)
        npt.assert_equal(anim.timeline.t, np.array([0, 1, 2]))


@requires_animatplot
class TestAnimateMultipleLines(PlotTestCase):
    @pytest.fixture(autouse=True)
    def setUp(self, linedata):
        self.darray = linedata

    def test_2d_animated_line_accepts_hue_kw(self):
        da = self.darray
        print(da)
        da.plot.line(hue='player', animate='time')
        assert (plt.gca().get_legend().get_title().get_text()
                == 'player')
        plt.cla()
        self.darray.plot.line(hue='time', animate='player')
        assert (plt.gca().get_legend().get_title().get_text()
                == 'time [s]')

    def test_animate_multiple_lines_classes(self):
        anim = self.darray.plot(animate='time', hue='player')
        assert isinstance(anim, amp.animation.Animation)

        line_block1, line_block2, title_block = anim.blocks

        assert isinstance(line_block1, amp.blocks.Line)
        assert isinstance(line_block2, amp.blocks.Line)
        assert isinstance(title_block, amp.blocks.Title)

    def test_animate_multiple_lines_data(self):
        anim = self.darray.plot(animate='time', hue='player')
        line_block1, _, title_block = anim.blocks

        assert len(line_block1) == 5
        assert len(line_block1) == len(title_block)

        expected = self.darray.isel(player=0).transpose('position', 'time')
        npt.assert_equal(line_block1.y, expected.values)
        npt.assert_equal(line_block1.x[:, 0],
                         self.darray.coords['position'].values)

    def test_animate_multiple_lines_text(self):
        anim = self.darray.plot(animate='time', hue='player')
        line_block1, _, title_block = anim.blocks

        assert title_block.titles[0] == 'time = 0'
        assert line_block1.ax.get_xlabel() == 'position [cm]'
        assert anim.timeline.units == ' [s]'

        # TODO check legend is correct

    def test_can_pass_in_axis(self):
        self.pass_in_axis(partial(self.darray.plot,
                                  animate='time', hue='player'))

    def test_animate_multiple_line_axes(self):
        line_block1, line_block2, _ = self.darray.plot(animate='time',
                                                       hue='player').blocks
        assert line_block1.ax is line_block2.ax

        # Check current axes is the plot (not the timeline etc.)
        assert plt.gca() is line_block1.ax


class TestAnimatedFacetGrid:
    def test_faceting_not_implemented(self):
        da = DataArray(easy_array(2, 3, 4))

        with pytest.raises(NotImplementedError):
            da.plot(animate='dim_0', col='dim_1')
