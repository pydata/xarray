---
file_format: mystnb
kernelspec:
  name: python3
---

```{eval-rst}
.. currentmodule:: xarray
```

(arrow)=

# Working with Arrow

[Apache Arrow] is a columnar memory format for tabular data. Many libraries
use Arrow to exchange data, for example [Polars] and [DuckDB].

{py:class}`Dataset` and {py:class}`DataArray` implement the
[Arrow PyCapsule Interface]. Libraries that support this interface can read
xarray objects directly.

Xarray flattens each object to a table. The table has one column for each
coordinate and one column for each data variable. Xarray does not copy the
data when it can. See {ref}`io.arrow` for the conditions.

```{warning}
This functionality is **experimental**. It can change in a future release.
```

```{code-cell}
:tags: [remove-input]

import numpy as np
import xarray as xr

np.random.seed(123456)
```

In this page, we use this dataset:

```{code-cell}
ds = xr.Dataset(
    {
        "temperature": (("time", "station"), 15 + 10 * np.random.rand(4, 3)),
        "pressure": (("time", "station"), 1000 + 20 * np.random.rand(4, 3)),
    },
    coords={
        "time": np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
        "station": ["A", "B", "C"],
    },
)
ds
```

## Polars

[Polars] is a DataFrame library. To make a Polars DataFrame from a
{py:class}`Dataset`, convert the dataset with {py:meth}`Dataset.to_arrow` first:

```{code-cell}
import polars as pl

df = pl.DataFrame(ds.to_arrow())
df
```

You can give a {py:class}`DataArray` directly to {py:class}`polars.DataFrame`.
The data column has the name of the data array:

```{code-cell}
pl.DataFrame(ds["temperature"])
```

Then use the Polars API on the DataFrame. For example, calculate the mean
temperature for each station:

```{code-cell}
df.group_by("station").agg(pl.col("temperature").mean()).sort("station")
```

## DuckDB

[DuckDB] is an in-process SQL database. DuckDB can query a Python variable by
its name. Use the name of an xarray object as a table name:

```{code-cell}
import duckdb

duckdb.sql(
    """
    SELECT station, avg(temperature) AS mean_temperature
    FROM ds
    GROUP BY station
    ORDER BY station
    """
)
```

You can also give the object to [`duckdb.from_arrow`][duckdb.from_arrow]:

```{code-cell}
rel = duckdb.from_arrow(ds)
rel.filter("pressure > 1010").order("time, station")
```

[apache arrow]: https://arrow.apache.org/
[arrow pycapsule interface]: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
[duckdb]: https://duckdb.org/
[duckdb.from_arrow]: https://duckdb.org/docs/stable/clients/python/reference/index.html#duckdb.from_arrow
[polars]: https://pola.rs/
