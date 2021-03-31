# Property-based tests using Hypothesis

This directory contains property-based tests using a library
called [Hypothesis](https://github.com/HypothesisWorks/hypothesis-python).

The property tests for xarray are a work in progress - more are always welcome.
They are stored in a separate directory because they tend to run more examples
and thus take longer, and so that local development can run a test suite
without needing to `pip install hypothesis`.

## Hang on, "property-based" tests?

Instead of making assertions about operations on a particular piece of
data, you use Hypothesis to describe a *kind* of data, then make assertions
that should hold for *any* example of this kind.

For example: "given a 2d ndarray of dtype uint8 `arr`,
`xr.DataArray(arr).plot.imshow()` never raises an exception".

Hypothesis will then try many random examples, and report a minimised
failing input for each error it finds.
[See the docs for more info.](https://hypothesis.readthedocs.io/en/master/)
