Getting Started
===============

.. ipython:: python
   :suppress:

   import numpy as np
   np.random.seed(123456)

Creating a ``Dataset``
----------------------

Let's create some ``XArray`` objects and put them in a ``Dataset``:

.. ipython:: python

    import xray
    import numpy as np
    import pandas as pd
    time = xray.XArray('time', pd.date_range('2010-01-01', periods=365))
    us_state = xray.XArray('us_state', ['WA', 'OR', 'CA', 'NV'])
    temp_data = (30 * np.cos(np.pi * np.linspace(-1, 1, 365).reshape(-1, 1))
                + 5 * np.arange(5, 9).reshape(1, -1)
                + 3 * np.random.randn(365, 4))
    temperature = xray.XArray(
       ['time', 'us_state'], temp_data, attributes={'units': 'degrees_F'})
    avg_rain = xray.XArray(
       'us_state', [27.66, 37.39, 17.28, 7.87], {'units': 'inches/year'})
    ds = xray.Dataset({'time': time, 'temperature': temperature,
                      'us_state': us_state, 'avg_rain': avg_rain},
                     attributes={'title': 'example dataset'})
    ds

This dataset contains two non-coordinate variables, ``temperature`` and
``avg_rain``, as well as the coordinates ``time`` and ``us_state``.

We can now access the contents of ``ds`` as self-described ``DatasetArray``
objects:

.. ipython:: python

    ds['temperature']

As you might guess, ``Dataset`` acts like a dictionary of variables. We
dictionary syntax to modify dataset variables in-place:

.. ipython:: python

    ds['foo'] = ('us_state', 0.1 * np.random.rand(4))
    ds
    del ds['foo']
    ds

On the first line, we used a shortcut: we specified the 'foo' variable by
a tuple of the arguments for ``XArray`` instead of an ``XArray`` object.
This works, because a dataset can contain only ``XArray`` objects.

We can also access some derived variables from time dimensions without
actually needing to put them in our dataset:

.. ipython:: python

    ds['time.dayofyear']

Dataset math
------------

We can manipulate variables in a dataset like numpy arrays, while still
keeping track of their metadata:

.. ipython:: python

    np.tan((ds['temperature'] + 10) ** 2)

Sometimes, we really want just the plain numpy array. That's easy, too:

.. ipython:: python

    ds['temperature'].data

An advantage of sticking with dataset arrays is that we can use dimension
based broadcasting instead of numpy's shape based broadcasting:

.. ipython:: python

    # this wouldn't work in numpy, because both these variables are 1d:
    ds['time.month'] * ds['avg_rain']

We can also apply operations across dimesions by name instead of using
axis numbers:

.. ipython:: python

    ds['temperature'].mean('time')

Integration with ``pandas``
---------------------------

Turning a dataset into a ``pandas.DataFrame`` broadcasts all the variables
over all dimensions:

.. ipython:: python

    df = ds.to_dataframe()
    df.head()

Using the ``plot`` method on pandas objects is almost certainly the easiest way
to plot xray objects:

.. ipython:: python

    # ds['temperature'].to_series() would work in place of df['temperature'] here
    @savefig series_plot_example.png width=6in
    df['temperature'].unstack('us_state').plot()
