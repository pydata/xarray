import numpy as np
import pytest

from xarray import DataArray
from . import requires_animatplot

# TODO should check that matplotlib >= 2.2 is present first?
try:
    import animatplot as amp
except ImportError:
    pass


@requires_animatplot
class TestAnimateLine:
    @pytest.fixture(autouse=True)
    def setUp(self):
        d = np.array([[0.0, 1.1, 0.0, 2],
                      [0.1, 1.3, 0.2, 2.1],
                      [0.1, 1.4, 0.3, 2.2],
                      [0.2, 1.3, 0.2, 2.3],
                      [0.1, 1.2, 0.2, 2.2]])
        self.darray = DataArray(d, name='height',
                                coords={'time': 10*np.arange(d.shape[0]),
                                        'position': 0.1*np.arange(d.shape[1])},
                                dims=('time', 'position'),
                                attrs={'units': 'm'})
        self.darray.time.attrs['units'] = 's'
        self.darray.position.attrs['units'] = 'cm'

    @pytest.mark.slow
    def test_animate_single_line(self):

        print(self.darray)
        a = self.darray.plot.line(animate_over='time')
        assert isinstance(a, amp.animation.Animation)

        line_block, title_block = a.blocks
        assert isinstance(line_block, amp.blocks.Line)
        assert isinstance(title_block, amp.blocks.Title)

        assert len(line_block) == 5
        assert len(line_block) == len(title_block)

        # TODO check many more things here
        # (also better testing in animatplot needed)
