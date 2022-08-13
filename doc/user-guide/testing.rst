.. _testing:

Testing your code
=================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

.. _asserts:

Asserts
-------

TODO

.. _hypothesis:

Hypothesis testing
------------------

.. note::

  Testing with hypothesis is a fairly advanced topic. Before reading this section it is recommended that you take a look
  at our guide to xarray's data structures, are familiar with conventional unit testing in pytest, and have seen the
  hypothesis library documentation.

``Hypothesis`` is a powerful library for property-based testing.
Instead of writing tests for one example at a time, it allows you to write tests parameterized by a source of many
dynamically generated examples. For example you might have written a test which you wish to be parameterized by the set
of all possible ``integers()``.

Property-based testing is extremely powerful, because (unlike more conventional example-based testing) it can find bugs
that you did not even think to look for!

Strategies
~~~~~~~~~~

Each source of examples is called a "strategy", and xarray provides a range of custom strategies which produce xarray
data structures containing arbitrary data. You can use these to efficiently test downstream code,
quickly ensuring that your code can handle xarray objects of all possible structures and contents.

These strategies are accessible in the :py:module::`xarray.testing.strategies` module, which provides

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/

   testing.strategies.valid_dtypes
   testing.strategies.np_arrays
   testing.strategies.dimension_names
   testing.strategies.variables
   testing.strategies.dataarrays
   testing.strategies.datasets
   testing.strategies.chunks
   testing.strategies.chunksizes

Generating Examples
~~~~~~~~~~~~~~~~~~~

To see an example of what each of these strategies might produce, you can call one followed by the `.example()` method,
which is a general hypothesis method valid for all strategies

.. ipython:: python

    import xarray.testing.strategies as xrst

    # TODO change this to dataarray once written
    xrst.variables().example()
    xrst.variables().example()
    xrst.variables().example()

You can see that calling `.example()` multiple times will generate different examples, giving you an idea of the wide
range of data that the xarray strategies can generate.

# TODO simple test example

.. ipython:: python

    import hypothesis.strategies as st

Chaining Strategies
~~~~~~~~~~~~~~~~~~~

Xarray's strategies can accept other strategies as arguments, allowing you to customise the contents of the generated
examples.

.. ipython:: python

    xrst.variables(data=xrst.np_arrays(shape=(3, 4)))

This also works with strategies defined in other packages, for example the ``chunks`` strategy defined in
``dask.array.strategies``.


Fixing Arguments
~~~~~~~~~~~~~~~~

If you want to fix one aspect of the data structure, whilst allowing variation in the generated examples
over all other aspects, then use ``st.just()``.

.. ipython:: python
    :okexcept:

    # Generates only dataarrays with dimensions ["x", "y"]
    xrst.dataarrays(dims=st.just(["x", "y"]))).example()

(This is technically another example of chaining strategies - ``hypothesis.strategies.just`` is simply a special
strategy that just contains a single example.)


Duck-type Conversion
~~~~~~~~~~~~~~~~~~~~

# TODO converting to duckarrays