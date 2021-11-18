from collections import OrderedDict
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose


def test_issue_6002():
    """Ref: https://github.com/pydata/xarray/issues/6002"""

    n_time = 1  # 1 : Fails, 2 : everything is fine

    from xarray.core.options import OPTIONS

    OPTIONS["use_bottleneck"] = True  # Set to False for work-around

    # Build some dataset
    dirs = np.linspace(0, 360, num=121)
    freqs = np.linspace(0, 4, num=192)
    spec_data = np.random.random(size=(n_time, 192, 121))

    dims = ("time", "freq", "dir")
    coords = OrderedDict()
    coords["time"] = range(n_time)
    coords["freq"] = freqs
    coords["dir"] = dirs

    xdata = xr.DataArray(
        data=spec_data,
        coords=coords,
        dims=dims,
        name="Spec name",
    ).to_dataset()

    expected = np.max(spec_data)

    xdata = xdata.transpose(..., "freq")  # remove this line and the script will run

    tm = xdata.max()

    assert_allclose(tm, expected)
