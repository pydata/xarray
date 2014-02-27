import numpy as np

from xray.conventions import MaskedAndScaledArray
from . import TestCase


class TestMaskedAndScaledArray(TestCase):
    def test(self):
        x = MaskedAndScaledArray(np.arange(3), fill_value=0)
        self.assertEqual(x.dtype, np.dtype('float'))
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.size, 3)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 3)
        self.assertArrayEqual([np.nan, 1, 2], x)

        x = MaskedAndScaledArray(np.arange(3), add_offset=1)
        self.assertArrayEqual(np.arange(3) + 1, x)

        x = MaskedAndScaledArray(np.arange(3), scale_factor=2)
        self.assertArrayEqual(2 * np.arange(3), x)

        x = MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]), -99, 0.01, 1)
        expected = np.array([np.nan, 0.99, 1, 1.01, 1.02])
        self.assertArrayEqual(expected, x)
