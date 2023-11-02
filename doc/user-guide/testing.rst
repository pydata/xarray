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

   testing.strategies.numeric_dtypes
   testing.strategies.np_arrays
   testing.strategies.names
   testing.strategies.dimension_names
   testing.strategies.dimension_sizes
   testing.strategies.attrs
   testing.strategies.variables

These build upon the numpy strategies offered in :py:mod:`hypothesis.extra.numpy`:

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


Creating Duck-type Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray objects don't have to wrap numpy arrays, in fact they can wrap any array type which presents the same API as a
numpy array (so-called "duck array wrapping", see :ref:`wrapping numpy-like arrays <internals.duck_arrays>`).

Imagine we want to write a strategy which generates arbitrary ``Variable`` objects, each of which wraps a
:py:class:`sparse.COO` array instead of a ``numpy.ndarray``. How could we do that? There are two ways:

1. Create a xarray object with numpy data and use the hypothesis' ``.map()`` method to convert the underlying array to a
different type:

.. ipython:: python
    :okexcept:

    import sparse

.. ipython:: python
    :okexcept:

    def convert_to_sparse(var):
        if var.ndim == 0:
            return var
        else:
            var.data = sparse.COO.from_numpy(da.values)
            return var

.. ipython:: python
    :okexcept:

    sparse_variables = xrst.variables().map(convert_to_sparse)

    sparse_variables.example()
    sparse_variables.example()

2. Pass a function which returns a strategy which generates the duck-typed arrays directly to the ``array_strategy_fn`` argument of the xarray strategies:

.. ipython:: python
    :okexcept:

    @st.composite
    def sparse_random_arrays(
        draw, shape: tuple[int] = None
    ) -> st.SearchStrategy[sparse._coo.core.COO]:
        """Strategy which generates random sparse.COO arrays"""
        if shape is None:
            shape = draw(npst.array_shapes())
        density = draw(st.integers(min_value=0, max_value=1))
        return sparse.random(
            shape=shape, density=density
        )  # note sparse.random does not accept a dtype kwarg


    def sparse_random_arrays_fn(
        *, shape: tuple[int] = None, dtype: np.dtype = None
    ) -> st.SearchStrategy[sparse._coo.core.COO]:
        return sparse_arrays(shape=shape)


.. ipython:: python
    :okexcept:

    sparse_random_variables = xrst.variables(
        array_strategy_fn=sparse_random_arrays_fn, dtype=st.just(np.dtype("float64"))
    )
    sparse_random_variables.example()

Either approach is fine, but one may be more convenient than the other depending on the type of the duck array which you
want to wrap.

If the array type you want to generate has a top-level namespace (e.g. that which is conventionally imported as ``xp`` or similar),
you can use this neat trick:

.. ipython:: python
    :okexcept:

    import numpy.array_api as xp  # available in numpy 1.26.0

    from hypothesis.extra.array_api import make_strategies_namespace

    numpy_variables = xrst.variables(
        array_strategy_fn=make_strategies_namespace(xp).arrays
    )
    numpy_variables.example()
