---
file_format: mystnb
kernelspec:
  name: python3
---

(internals-duckarrays)=

# Integrating with duck arrays

:::{warning}
This is an experimental feature. Please report any bugs or other difficulties on [xarray's issue tracker](https://github.com/pydata/xarray/issues).
:::

Xarray can wrap custom numpy-like arrays ("{term}`duck array`s") - see the {ref}`user guide documentation <userguide.duckarrays>`.
This page is intended for developers who are interested in wrapping a new custom array type with xarray.

(internals-duckarrays-requirements)=

## Duck array requirements

Xarray does not explicitly check that required methods are defined by the underlying duck array object before
attempting to wrap the given array. However, a wrapped array type should at a minimum define these attributes:

- `shape` property,
- `dtype` property,
- `ndim` property,
- `__array__` method,
- `__array_ufunc__` method,
- `__array_function__` method.

These need to be defined consistently with {py:class}`numpy.ndarray`, for example the array `shape`
property needs to obey [numpy's broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
(see also the [Python Array API standard's explanation](https://data-apis.org/array-api/latest/API_specification/broadcasting.html)
of these same rules).

(internals-duckarrays-array-api-standard)=

## Python Array API standard support

As an integration library xarray benefits greatly from the standardization of duck-array libraries' APIs, and so is a
big supporter of the [Python Array API Standard](https://data-apis.org/array-api/latest/).

We aim to support any array libraries that follow the Array API standard out-of-the-box. However, xarray does occasionally
call some numpy functions which are not (yet) part of the standard (e.g. {py:meth}`xarray.DataArray.pad` calls {py:func}`numpy.pad`).
See [xarray issue #7848](https://github.com/pydata/xarray/issues/7848) for a list of such functions. We can still support dispatching on these functions through
the array protocols above, it just means that if you exclusively implement the methods in the Python Array API standard
then some features in xarray will not work.

## Custom inline reprs

In certain situations (e.g. when printing the collapsed preview of
variables of a `Dataset`), xarray will display the repr of a {term}`duck array`
in a single line, truncating it to a certain number of characters. If that
would drop too much information, the {term}`duck array` may define a
`_repr_inline_` method that takes `max_width` (number of characters) as an
argument

```python
class MyDuckArray:
    ...

    def _repr_inline_(self, max_width):
        """format to a single line with at most max_width characters"""
        ...

    ...
```

To avoid duplicated information, this method must omit information about the shape and
{term}`dtype`. For example, the string representation of a `dask` array or a
`sparse` matrix would be:

```{code-cell}
import dask.array as da
import xarray as xr
import numpy as np
import sparse
```

```{code-cell}
a = da.linspace(0, 1, 20, chunks=2)
a
```

```{code-cell}
b = np.eye(10)
b[[5, 7, 3, 0], [6, 8, 2, 9]] = 2
b = sparse.COO.from_numpy(b)
b
```

```{code-cell}
xr.Dataset(dict(a=("x", a), b=(("y", "z"), b)))
```
