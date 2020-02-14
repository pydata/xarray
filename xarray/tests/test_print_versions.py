import io

import xarray


def test_show_versions():
    f = io.StringIO()
    xarray.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
