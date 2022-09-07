.. _testing:

Testing your code
=================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

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

   testing.strategies.numeric_dtypes
   testing.strategies.np_arrays
   testing.strategies.names
   testing.strategies.dimension_names
   testing.strategies.dimension_sizes
   testing.strategies.attrs
   testing.strategies.variables
   testing.strategies.coordinate_variables
   testing.strategies.dataarrays
   testing.strategies.data_variables
   testing.strategies.datasets

Generating Examples
~~~~~~~~~~~~~~~~~~~

To see an example of what each of these strategies might produce, you can call one followed by the ``.example()`` method,
which is a general hypothesis method valid for all strategies.

.. ipython:: python

    import xarray.testing.strategies as xrst

    xrst.dataarrays().example()
    xrst.dataarrays().example()
    xrst.dataarrays().example()

You can see that calling ``.example()`` multiple times will generate different examples, giving you an idea of the wide
range of data that the xarray strategies can generate.

In your tests however you should not use ``.example()`` - instead you should parameterize your tests with the
``hypothesis.given`` decorator:

.. ipython:: python

    from hypothesis import given

.. ipython:: python

    @given(xrst.dataarrays())
    def test_function_that_acts_on_dataarrays(da):
        assert func(da) == ...


Chaining Strategies
~~~~~~~~~~~~~~~~~~~

Xarray's strategies can accept other strategies as arguments, allowing you to customise the contents of the generated
examples.

.. ipython:: python

    # generate a DataArray with shape (3, 4), but all other details still arbitrary
    xrst.dataarrays(
        data=xrst.np_arrays(shape=(3, 4), dtype=np.dtype("int32"))
    ).example()

This also works with custom strategies, or strategies defined in other packages.
For example you could create a ``chunks`` strategy to specify particular chunking patterns for a dask-backed array.

.. warning::
    When passing multiple different strategies to the same constructor the drawn examples must be mutually compatible.

    In order to construct a valid xarray object to return, our strategies must check that the
    variables / dimensions / coordinates are mutually compatible. If you pass multiple custom strategies to a strategy
    constructor which are not compatible in all cases, an error will be raised, *even if they are still compatible in
    other cases*. For example

    .. code-block::

        @given(st.data())
        def test_something_else_inefficiently(data):
            arrs = npst.arrays(dtype=numeric_dtypes)  # generates arrays of any shape
            dims = xrst.dimension_names()  # generates lists of any number of dimensions

            # Drawing examples from this strategy will raise a hypothesis.errors.InvalidArgument error.
            var = data.draw(xrst.variables(data=arrs, dims=dims))

            assert ...

    Here we have passed custom strategies which won't often be compatible: only rarely will the array's ``ndims``
    correspond to the number of dimensions drawn. We forbid arguments that are only *sometimes* compatible in order to
    avoid extremely poor example generation performance (as generating invalid examples and rejecting them is
    potentially unboundedly inefficient).


Fixing Arguments
~~~~~~~~~~~~~~~~

If you want to fix one aspect of the data structure, whilst allowing variation in the generated examples
over all other aspects, then use ``hypothesis.strategies.just()``.

.. ipython:: python

    import hypothesis.strategies as st

    # Generates only dataarrays with dimensions ["x", "y"]
    xrst.dataarrays(dims=st.just(["x", "y"])).example()

(This is technically another example of chaining strategies - ``hypothesis.strategies.just`` is simply a special
strategy that just contains a single example.)

To fix the length of dimensions you can instead pass `dims` as a mapping of dimension names to lengths
(i.e. following xarray objects' ``.sizes()`` property), e.g.

.. ipython:: python

    # Generates only dataarrays with dimensions ["x", "y"], of lengths 2 & 3 respectively
    xrst.dataarrays(dims=st.just({"x": 2, "y": 3})).example()

You can also use this to specify that you want examples which are missing some part of the data structure, for instance

.. ipython:: python

    # Generates only dataarrays with no coordinates
    xrst.datasets(data_vars=st.just({})).example()

Through a combination of chaining strategies and fixing arguments, you can specify quite complicated requirements on the
objects your chained strategy will generate.

.. ipython:: python

    fixed_x_variable_y_maybe_z = st.fixed_dictionaries(
        {"x": st.just(2), "y": st.integers(3, 4)}, optional={"z": st.just(2)}
    )

    fixed_x_variable_y_maybe_z.example()

    special_dataarrays = xrst.dataarrays(dims=fixed_x_variable_y_maybe_z)

    special_dataarrays.example()
    special_dataarrays.example()

Here we have used one of hypothesis' built-in strategies ``fixed_dictionaries`` to create a strategy which generates
mappings of dimension names to lengths (i.e. the ``size`` of the xarray object we want).
This particular strategy will always generate an ``x`` dimension of length 2, and a ``y`` dimension of
length either 3 or 4, and will sometimes also generate a ``z`` dimension of length 2.
By feeding this strategy for dictionaries into the `dims` argument of xarray's `dataarrays` strategy, we can generate
arbitrary ``DataArray`` objects whose dimensions will always match these specifications.


Creating Duck-type Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray objects don't have to wrap numpy arrays, in fact they can wrap any array type which presents the same API as a
numpy array (so-called "duck array wrapping", see :ref:`_internals.duck_arrays`).

Imagine we want to write a strategy which generates arbitrary `DataArray` objects, each of which wraps a
``sparse.COO`` array instead of a ``numpy.ndarray``. How could we do that? There are two ways:

1. Create a xarray object with numpy data and use ``.map()`` to convert the underlying array to a
different type:

.. ipython:: python
    :okexcept:

    import sparse

.. ipython:: python
    :okexcept:

    def convert_to_sparse(arr):
        if arr.ndim == 0:
            return arr
        else:
            return sparse.COO.from_numpy(arr)

.. ipython:: python
    :okexcept:

    sparse_dataarrays = xrst.dataarrays().map(convert_to_sparse)

    sparse_dataarrays.example()
    sparse_dataarrays.example()

2. Pass a strategy which generates the duck-typed arrays directly to the ``data`` argument of the xarray
strategies:

.. ipython:: python
    :okexcept:

    @st.composite
    def sparse_arrays(draw) -> st.SearchStrategy[sparse._coo.core.COO]:
        """Strategy which generates random sparse.COO arrays"""
        shape = draw(npst.array_shapes())
        density = draw(st.integers(min_value=0, max_value=1))
        return sparse.random(shape, density=density)

.. ipython:: python
    :okexcept:

    sparse_dataarrays = xrst.dataarrays(data=sparse_arrays())

    sparse_dataarrays.example()
    sparse_dataarrays.example()
