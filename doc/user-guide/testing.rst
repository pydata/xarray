.. _testing:

Testing your code
=================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

.. _testing.hypothesis:

Hypothesis testing
------------------

.. note::

  Testing with hypothesis is a fairly advanced topic. Before reading this section it is recommended that you take a look
  at our guide to xarray's :ref:`data structures`, are familiar with conventional unit testing in
  `pytest <https://docs.pytest.org/>`_, and have seen the
  `hypothesis library documentation <https://hypothesis.readthedocs.io/>`_.

`The hypothesis library <https://hypothesis.readthedocs.io/>`_ is a powerful tool for property-based testing.
Instead of writing tests for one example at a time, it allows you to write tests parameterized by a source of many
dynamically generated examples. For example you might have written a test which you wish to be parameterized by the set
of all possible integers via :py:func:`hypothesis.strategies.integers()`.

Property-based testing is extremely powerful, because (unlike more conventional example-based testing) it can find bugs
that you did not even think to look for!

Strategies
~~~~~~~~~~

Each source of examples is called a "strategy", and xarray provides a range of custom strategies which produce xarray
data structures containing arbitrary data. You can use these to efficiently test downstream code,
quickly ensuring that your code can handle xarray objects of all possible structures and contents.

These strategies are accessible in the :py:mod:`xarray.testing.strategies` module, which provides

.. currentmodule:: xarray

.. autosummary::

   testing.strategies.supported_dtypes
   testing.strategies.names
   testing.strategies.dimension_names
   testing.strategies.dimension_sizes
   testing.strategies.attrs
   testing.strategies.variables
   testing.strategies.unique_subset_of

These build upon the numpy and array API strategies offered in :py:mod:`hypothesis.extra.numpy` and :py:mod:`hypothesis.extra.array_api`:

.. ipython:: python

    import hypothesis.extra.numpy as npst

Generating Examples
~~~~~~~~~~~~~~~~~~~

To see an example of what each of these strategies might produce, you can call one followed by the ``.example()`` method,
which is a general hypothesis method valid for all strategies.

.. ipython:: python

    import xarray.testing.strategies as xrst

    xrst.variables().example()
    xrst.variables().example()
    xrst.variables().example()

You can see that calling ``.example()`` multiple times will generate different examples, giving you an idea of the wide
range of data that the xarray strategies can generate.

In your tests however you should not use ``.example()`` - instead you should parameterize your tests with the
:py:func:`hypothesis.given` decorator:

.. ipython:: python

    from hypothesis import given

.. ipython:: python

    @given(xrst.variables())
    def test_function_that_acts_on_variables(var):
        assert func(var) == ...


Chaining Strategies
~~~~~~~~~~~~~~~~~~~

Xarray's strategies can accept other strategies as arguments, allowing you to customise the contents of the generated
examples.

.. ipython:: python

    # generate a Variable containing an array with a complex number dtype, but all other details still arbitrary
    from hypothesis.extra.numpy import complex_number_dtypes

    xrst.variables(dtype=complex_number_dtypes()).example()

This also works with custom strategies, or strategies defined in other packages.
For example you could imagine creating a ``chunks`` strategy to specify particular chunking patterns for a dask-backed array.

Fixing Arguments
~~~~~~~~~~~~~~~~

If you want to fix one aspect of the data structure, whilst allowing variation in the generated examples
over all other aspects, then use :py:func:`hypothesis.strategies.just()`.

.. ipython:: python

    import hypothesis.strategies as st

    # Generates only variable objects with dimensions ["x", "y"]
    xrst.variables(dims=st.just(["x", "y"])).example()

(This is technically another example of chaining strategies - :py:func:`hypothesis.strategies.just()` is simply a
special strategy that just contains a single example.)

To fix the length of dimensions you can instead pass ``dims`` as a mapping of dimension names to lengths
(i.e. following xarray objects' ``.sizes()`` property), e.g.

.. ipython:: python

    # Generates only variables with dimensions ["x", "y"], of lengths 2 & 3 respectively
    xrst.variables(dims=st.just({"x": 2, "y": 3})).example()

You can also use this to specify that you want examples which are missing some part of the data structure, for instance

.. ipython:: python

    # Generates a Variable with no attributes
    xrst.variables(attrs=st.just({})).example()

Through a combination of chaining strategies and fixing arguments, you can specify quite complicated requirements on the
objects your chained strategy will generate.

.. ipython:: python

    fixed_x_variable_y_maybe_z = st.fixed_dictionaries(
        {"x": st.just(2), "y": st.integers(3, 4)}, optional={"z": st.just(2)}
    )
    fixed_x_variable_y_maybe_z.example()

    special_variables = xrst.variables(dims=fixed_x_variable_y_maybe_z)

    special_variables.example()
    special_variables.example()

Here we have used one of hypothesis' built-in strategies :py:func:`hypothesis.strategies.fixed_dictionaries` to create a
strategy which generates mappings of dimension names to lengths (i.e. the ``size`` of the xarray object we want).
This particular strategy will always generate an ``x`` dimension of length 2, and a ``y`` dimension of
length either 3 or 4, and will sometimes also generate a ``z`` dimension of length 2.
By feeding this strategy for dictionaries into the ``dims`` argument of xarray's :py:func:`~st.variables` strategy,
we can generate arbitrary :py:class:`~xarray.Variable` objects whose dimensions will always match these specifications.

Generating Duck-type Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray objects don't have to wrap numpy arrays, in fact they can wrap any array type which presents the same API as a
numpy array (so-called "duck array wrapping", see :ref:`wrapping numpy-like arrays <internals.duckarrays>`).

Imagine we want to write a strategy which generates arbitrary ``Variable`` objects, each of which wraps a
:py:class:`sparse.COO` array instead of a ``numpy.ndarray``. How could we do that? There are two ways:

1. Create a xarray object with numpy data and use the hypothesis' ``.map()`` method to convert the underlying array to a
different type:

.. ipython:: python

    import sparse

.. ipython:: python

    def convert_to_sparse(var):
        return var.copy(data=sparse.COO.from_numpy(var.to_numpy()))

.. ipython:: python

    sparse_variables = xrst.variables(dims=xrst.dimension_names(min_dims=1)).map(
        convert_to_sparse
    )

    sparse_variables.example()
    sparse_variables.example()

2. Pass a function which returns a strategy which generates the duck-typed arrays directly to the ``array_strategy_fn`` argument of the xarray strategies:

.. ipython:: python

    def sparse_random_arrays(shape: tuple[int, ...]) -> sparse._coo.core.COO:
        """Strategy which generates random sparse.COO arrays"""
        if shape is None:
            shape = npst.array_shapes()
        else:
            shape = st.just(shape)
        density = st.integers(min_value=0, max_value=1)
        # note sparse.random does not accept a dtype kwarg
        return st.builds(sparse.random, shape=shape, density=density)


    def sparse_random_arrays_fn(
        *, shape: tuple[int, ...], dtype: np.dtype
    ) -> st.SearchStrategy[sparse._coo.core.COO]:
        return sparse_random_arrays(shape=shape)


.. ipython:: python

    sparse_random_variables = xrst.variables(
        array_strategy_fn=sparse_random_arrays_fn, dtype=st.just(np.dtype("float64"))
    )
    sparse_random_variables.example()

Either approach is fine, but one may be more convenient than the other depending on the type of the duck array which you
want to wrap.

Compatibility with the Python Array API Standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray aims to be compatible with any duck-array type that conforms to the `Python Array API Standard <https://data-apis.org/array-api/latest/>`_
(see our :ref:`docs on Array API Standard support <internals.duckarrays.array_api_standard>`).

.. warning::

    The strategies defined in :py:mod:`testing.strategies` are **not** guaranteed to use array API standard-compliant
    dtypes by default.
    For example arrays with the dtype ``np.dtype('float16')`` may be generated by :py:func:`testing.strategies.variables`
    (assuming the ``dtype`` kwarg was not explicitly passed), despite ``np.dtype('float16')`` not being in the
    array API standard.

If the array type you want to generate has an array API-compliant top-level namespace
(e.g. that which is conventionally imported as ``xp`` or similar),
you can use this neat trick:

.. ipython:: python

    import numpy as xp  # compatible in numpy 2.0

    # use `import numpy.array_api as xp` in numpy>=1.23,<2.0

    from hypothesis.extra.array_api import make_strategies_namespace

    xps = make_strategies_namespace(xp)

    xp_variables = xrst.variables(
        array_strategy_fn=xps.arrays,
        dtype=xps.scalar_dtypes(),
    )
    xp_variables.example()

Another array API-compliant duck array library would replace the import, e.g. ``import cupy as cp`` instead.

Testing over Subsets of Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common task when testing xarray user code is checking that your function works for all valid input dimensions.
We can chain strategies to achieve this, for which the helper strategy :py:func:`~testing.strategies.unique_subset_of`
is useful.

It works for lists of dimension names

.. ipython:: python

    dims = ["x", "y", "z"]
    xrst.unique_subset_of(dims).example()
    xrst.unique_subset_of(dims).example()

as well as for mappings of dimension names to sizes

.. ipython:: python

    dim_sizes = {"x": 2, "y": 3, "z": 4}
    xrst.unique_subset_of(dim_sizes).example()
    xrst.unique_subset_of(dim_sizes).example()

This is useful because operations like reductions can be performed over any subset of the xarray object's dimensions.
For example we can write a pytest test that tests that a reduction gives the expected result when applying that reduction
along any possible valid subset of the Variable's dimensions.

.. code-block:: python

    import numpy.testing as npt


    @given(st.data(), xrst.variables(dims=xrst.dimension_names(min_dims=1)))
    def test_mean(data, var):
        """Test that the mean of an xarray Variable is always equal to the mean of the underlying array."""

        # specify arbitrary reduction along at least one dimension
        reduction_dims = data.draw(xrst.unique_subset_of(var.dims, min_size=1))

        # create expected result (using nanmean because arrays with Nans will be generated)
        reduction_axes = tuple(var.get_axis_num(dim) for dim in reduction_dims)
        expected = np.nanmean(var.data, axis=reduction_axes)

        # assert property is always satisfied
        result = var.mean(dim=reduction_dims).data
        npt.assert_equal(expected, result)
