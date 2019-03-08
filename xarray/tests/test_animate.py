import numpy as np
import numpy.testing as npt
import pytest

from xarray import DataArray
from . import requires_animatplot

# TODO should check that matplotlib >= 2.2 is present first?
try:
    import animatplot as amp
except ImportError:
    pass

from xarray.plot.animate import animate_line, _create_timeline


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

    @pytest.mark.slow
    def test_animate_single_line(self):
        anim = self.darray.plot(animate='time')
        assert isinstance(anim, amp.animation.Animation)

        line_block, title_block = anim.blocks
        assert isinstance(line_block, amp.blocks.Line)
        assert isinstance(title_block, amp.blocks.Title)

        assert len(line_block) == 5
        assert len(line_block) == len(title_block)

        # TODO check many more things here
        # (also better testing in animatplot needed)
