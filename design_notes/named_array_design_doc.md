# named-array Design Document

## Abstract

Despite the wealth of scientific libraries in the Python ecosystem, there is a gap for a lightweight, efficient array structure with named dimensions that can provide convenient broadcasting and indexing.

Existing solutions like Xarray's Variable, [Pytorch Named Tensor](https://github.com/pytorch/pytorch/issues/60832), [Levanter](https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html), and [Larray](https://larray.readthedocs.io/en/stable/tutorial/getting_started.html) have their own strengths and weaknesses. Xarray's Variable is an efficient data structure, but it depends on the relatively heavy-weight library Pandas, which limits its use in other projects. Pytorch Named Tensor offers named dimensions, but it lacks support for many operations, making it less user-friendly. Levanter is a powerful tool with a named tensor module (Haliax) that makes deep learning code easier to read, understand, and write, but it is not as lightweight or generic as desired. Larry offers labeled N-dimensional arrays, but it may not provide the level of seamless interoperability with other scientific Python libraries that some users need.

named-array aims to solve these issues by exposing the core functionality of Xarray's Variable class as a standalone package.

## Motivation and Scope

The Python ecosystem boasts a wealth of scientific libraries that enable efficient computations on large, multi-dimensional arrays. Libraries like PyTorch, Xarray, and NumPy have revolutionized scientific computing by offering robust data structures for array manipulations. Despite this wealth of tools, a gap exists in the Python landscape for a lightweight, efficient array structure with named dimensions that can provide convenient broadcasting and indexing.

Xarray internally maintains a data structure that meets this need, referred to as [`xarray.Variable`](https://docs.xarray.dev/en/latest/generated/xarray.Variable.html) . However, Xarray's dependency on Pandas, a relatively heavy-weight library, restricts other projects from leveraging this efficient data structure (<https://github.com/nipy/nibabel/issues/412>, <https://github.com/scikit-learn/enhancement_proposals/pull/18>, <https://github.com/scikit-learn/enhancement_proposals/pull/18#issuecomment-511991096>).

We propose the creation of a standalone Python package, "named-array". This package is envisioned to be a version of the `xarray.Variable` data structure, cleanly separated from the heavier dependencies of Xarray. named-array will provide a lightweight, user-friendly array-like data structure with named dimensions, facilitating convenient indexing and broadcasting. The package will use existing scientific Python community standards such as established array protocols and the new [Python array API standard](https://data-apis.org/array-api/latest), allowing users to wrap multiple duck-array objects, including, but not limited to, NumPy, Dask, Sparse, Pint, CuPy, and Pytorch.

The development of named-array is projected to meet a key community need and expected to broaden Xarray's user base. By making the core `xarray.Variable` more accessible, we anticipate an increase in contributors and a reduction in the developer burden on current Xarray maintainers.

### Goals

1. **Simple and minimal**: named-array will expose Xarray's [Variable class](https://docs.xarray.dev/en/stable/internals/variable-objects.html) as a standalone object (`NamedArray`) with named axes (dimensions) and arbitrary metadata (attributes) but without coordinate labels. This will make it a lightweight, efficient array data structure that allows convenient broadcasting and indexing.

2. **Interoperability**: named-array will follow established scientific Python community standards and in doing so, will allow it to wrap multiple duck-array objects, including but not limited to, NumPy, Dask, Sparse, Pint, CuPy, and Pytorch.

3. **Community Engagement**: By making the core `xarray.Variable` more accessible, we open the door to increased adoption of this fundamental data structure. As such, we hope to see an increase in contributors and reduction in the developer burden on current Xarray maintainers.

### Non-Goals

1. **Extensive Data Analysis**: named-array will not provide extensive data analysis features like statistical functions, data cleaning, or visualization. Its primary focus is on providing a data structure that allows users to use dimension names for descriptive array manipulations.

2. **Support for I/O**: named-array will not bundle file reading functions. Instead users will be expected to handle I/O and then wrap those arrays with the new named-array data structure.

## Backward Compatibility

The creation of named-array is intended to separate the `xarray.Variable` from Xarray into a standalone package. This allows it to be used independently, without the need for Xarray's dependencies, like Pandas. This separation has implications for backward compatibility.

Since the new named-array is envisioned to contain the core features of Xarray's variable, existing code using Variable from Xarray should be able to switch to named-array with minimal changes. However, there are several potential issues related to backward compatibility:

- **API Changes**: as the Variable is decoupled from Xarray and moved into named-array, some changes to the API may be necessary. These changes might include differences in function signature, etc. These changes could break existing code that relies on the current API and associated utility functions (e.g. `as_variable()`). The `xarray.Variable` object will subclass `NamedArray`, and provide the existing interface for compatibility.

## Detailed Description

named-array aims to provide a lightweight, efficient array structure with named dimensions, or axes, that enables convenient broadcasting and indexing. The primary component of named-array is a standalone version of the xarray.Variable data structure, which was previously a part of the Xarray library.
The xarray.Variable data structure in named-array will maintain the core features of its counterpart in Xarray, including:

- **Named Axes (Dimensions)**: Each axis of the array can be given a name, providing a descriptive and intuitive way to reference the dimensions of the array.

- **Arbitrary Metadata (Attributes)**: named-array will support the attachment of arbitrary metadata to arrays as a dict, providing a mechanism to store additional information about the data that the array represents.

- **Convenient Broadcasting and Indexing**: With named dimensions, broadcasting and indexing operations become more intuitive and less error-prone.

The named-array package is designed to be interoperable with other scientific Python libraries. It will follow established scientific Python community standards and use standard array protocols, as well as the new data-apis standard. This allows named-array to wrap multiple duck-array objects, including, but not limited to, NumPy, Dask, Sparse, Pint, CuPy, and Pytorch.

## Implementation

- **Decoupling**: making `variable.py` agnostic to Xarray internals by decoupling it from the rest of the library. This will make the code more modular and easier to maintain. However, this will also make the code more complex, as we will need to define a clear interface for how the functionality in `variable.py` interacts with the rest of the library, particularly the ExplicitlyIndexed subclasses used to enable lazy indexing of data on disk.
- **Move Xarray's internal lazy indexing classes to follow standard Array Protocols**: moving the lazy indexing classes like `ExplicitlyIndexed` to use standard array protocols will be a key step in decoupling. It will also potentially improve interoperability with other libraries that use these protocols, and prepare these classes [for eventual movement out](https://github.com/pydata/xarray/issues/5081) of the Xarray code base. However, this will also require significant changes to the code, and we will need to ensure that all existing functionality is preserved.
  - Use [https://data-apis.org/array-api-compat/](https://data-apis.org/array-api-compat/) to handle compatibility issues?
- **Leave lazy indexing classes in Xarray for now**
- **Preserve support for Dask collection protocols**: named-array will preserve existing support for the dask collections protocol namely the **dask\_\*\*\*** methods
- **Preserve support for ChunkManagerEntrypoint?** Opening variables backed by dask vs cubed arrays currently is [handled within Variable.chunk](https://github.com/pydata/xarray/blob/92c8b33eb464b09d6f8277265b16cae039ab57ee/xarray/core/variable.py#L1272C15-L1272C15). If we are preserving dask support it would be nice to preserve general chunked array type support, but this currently requires an entrypoint.

### Plan

1. Create a new baseclass for `xarray.Variable` to its own module e.g. `xarray.core.base_variable`
2. Remove all imports of internal Xarray classes and utils from `base_variable.py`. `base_variable.Variable` should not depend on anything in xarray.core
   - Will require moving the lazy indexing classes (subclasses of ExplicitlyIndexed) to be standards compliant containers.`
     - an array-api compliant container that provides **array_namespace**`
     - Support `.oindex` and `.vindex` for explicit indexing
     - Potentially implement this by introducing a new compliant wrapper object?
   - Delete the `NON_NUMPY_SUPPORTED_ARRAY_TYPES` variable which special-cases ExplicitlyIndexed and `pd.Index.`
     - `ExplicitlyIndexed` class and subclasses should provide `.oindex` and `.vindex` for indexing by `Variable.__getitem__.`: `oindex` and `vindex` were proposed in [NEP21](https://numpy.org/neps/nep-0021-advanced-indexing.html), but have not been implemented yet
     - Delete the ExplicitIndexer objects (`BasicIndexer`, `VectorizedIndexer`, `OuterIndexer`)
     - Remove explicit support for `pd.Index`. When provided with a `pd.Index` object, Variable will coerce to an array using `np.array(pd.Index)`. For Xarray's purposes, Xarray can use `as_variable` to explicitly wrap these in PandasIndexingAdapter and pass them to `Variable.__init__`.
3. Define a minimal variable interface that the rest of Xarray can use:

   1. `dims`: tuple of dimension names
   2. `data`: numpy/dask/duck arrays`
   3. `attrs``: dictionary of attributes

4. Implement basic functions & methods for manipulating these objects. These methods will be a cleaned-up subset (for now) of functionality on xarray.Variable, with adaptations inspired by the [Python array API](https://data-apis.org/array-api/2022.12/API_specification/index.html).
5. Existing Variable structures
   1. Keep Variable object which subclasses the new structure that adds the `.encoding` attribute and potentially other methods needed for easy refactoring.
   2. IndexVariable will remain in xarray.core.variable and subclass the new named-array data structure pending future deletion.
6. Docstrings and user-facing APIs will need to be updated to reflect the changed methods on Variable objects.

Further implementation details are in Appendix: [Implementation Details](#appendix-implementation-details).

## Plan for decoupling lazy indexing functionality from NamedArray

Today's implementation Xarray's lazy indexing functionality uses three private objects: `*Indexer`, `*IndexingAdapter`, `*Array`.
These objects are needed for two reason:

1. We need to translate from Xarray (NamedArray) indexing rules to bare array indexing rules.
   - `*Indexer` objects track the type of indexing - basic, orthogonal, vectorized
2. Not all arrays support the same indexing rules, so we need `*Indexing` adapters
   1. Indexing Adapters today implement `__getitem__` and use type of `*Indexer` object to do appropriate conversions.
3. We also want to support lazy indexing of on-disk arrays.
   1. These again support different types of indexing, so we have `explicit_indexing_adapter` that understands `*Indexer` objects.

### Goals

1. We would like to keep the lazy indexing array objects, and backend array objects within Xarray. Thus NamedArray cannot treat these objects specially.
2. A key source of confusion (and coupling) is that both lazy indexing arrays and indexing adapters, both handle Indexer objects, and both subclass `ExplicitlyIndexedNDArrayMixin`. These are however conceptually different.

### Proposal

1. The `NumpyIndexingAdapter`, `DaskIndexingAdapter`, and `ArrayApiIndexingAdapter` classes will need to migrate to Named Array project since we will want to support indexing of numpy, dask, and array-API arrays appropriately.
2. The `as_indexable` function which wraps an array with the appropriate adapter will also migrate over to named array.
3. Lazy indexing arrays will implement `__getitem__` for basic indexing, `.oindex` for orthogonal indexing, and `.vindex` for vectorized indexing.
4. IndexingAdapter classes will similarly implement `__getitem__`, `oindex`, and `vindex`.
5. `NamedArray.__getitem__` (and `__setitem__`) will still use `*Indexer` objects internally (for e.g. in `NamedArray._broadcast_indexes`), but use `.oindex`, `.vindex` on the underlying indexing adapters.
6. We will move the `*Indexer` and `*IndexingAdapter` classes to Named Array. These will be considered private in the long-term.
7. `as_indexable` will no longer special case `ExplicitlyIndexed` objects (we can special case a new `IndexingAdapter` mixin class that will be private to NamedArray). To handle Xarray's lazy indexing arrays, we will introduce a new `ExplicitIndexingAdapter` which will wrap any array with any of `.oindex` of `.vindex` implemented.
   1. This will be the last case in the if-chain that is, we will try to wrap with all other `IndexingAdapter` objects before using `ExplicitIndexingAdapter` as a fallback. This Adapter will be used for the lazy indexing arrays, and backend arrays.
   2. As with other indexing adapters (point 4 above), this `ExplicitIndexingAdapter` will only implement `__getitem__` and will understand `*Indexer` objects.
8. For backwards compatibility with external backends, we will have to gracefully deprecate `indexing.explicit_indexing_adapter` which translates from Xarray's indexing rules to the indexing supported by the backend.
   1. We could split `explicit_indexing_adapter` in to 3:
      - `basic_indexing_adapter`, `outer_indexing_adapter` and `vectorized_indexing_adapter` for public use.
   2. Implement fall back `.oindex`, `.vindex` properties on `BackendArray` base class. These will simply rewrap the `key` tuple with the appropriate `*Indexer` object, and pass it on to `__getitem__` or `__setitem__`. These methods will also raise DeprecationWarning so that external backends will know to migrate to `.oindex`, and `.vindex` over the next year.

THe most uncertain piece here is maintaining backward compatibility with external backends. We should first migrate a single internal backend, and test out the proposed approach.

## Project Timeline and Milestones

We have identified the following milestones for the completion of this project:

1. **Write and publish a design document**: this document will explain the purpose of named-array, the intended audience, and the features it will provide. It will also describe the architecture of named-array and how it will be implemented. This will ensure early community awareness and engagement in the project to promote subsequent uptake.
2. **Refactor `variable.py` to `base_variable.py`** and remove internal Xarray imports.
3. **Break out the package and create continuous integration infrastructure**: this will entail breaking out the named-array project into a Python package and creating a continuous integration (CI) system. This will help to modularize the code and make it easier to manage. Building a CI system will help ensure that codebase changes do not break existing functionality.
4. Incrementally add new functions & methods to the new package, ported from xarray. This will start to make named-array useful on its own.
5. Refactor the existing Xarray codebase to rely on the newly created package (named-array): This will help to demonstrate the usefulness of the new package, and also provide an example for others who may want to use it.
6. Expand tests, add documentation, and write a blog post: expanding the test suite will help to ensure that the code is reliable and that changes do not introduce bugs. Adding documentation will make it easier for others to understand and use the project.
7. Finally, we will write a series of blog posts on [xarray.dev](https://xarray.dev/) to promote the project and attract more contributors.
   - Toward the end of the process, write a few blog posts that demonstrate the use of the newly available data structure
   - pick the same example applications used by other implementations/applications (e.g. Pytorch, sklearn, and Levanter) to show how it can work.

## Related Work

1. [GitHub - deepmind/graphcast](https://github.com/deepmind/graphcast)
2. [Getting Started — LArray 0.34 documentation](https://larray.readthedocs.io/en/stable/tutorial/getting_started.html)
3. [Levanter — Legible, Scalable, Reproducible Foundation Models with JAX](https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html)
4. [google/xarray-tensorstore](https://github.com/google/xarray-tensorstore)
5. [State of Torch Named Tensors · Issue #60832 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/60832)
   - Incomplete support: Many primitive operations result in errors, making it difficult to use NamedTensors in Practice. Users often have to resort to removing the names from tensors to avoid these errors.
   - Lack of active development: the development of the NamedTensor feature in PyTorch is not currently active due a lack of bandwidth for resolving ambiguities in the design.
   - Usability issues: the current form of NamedTensor is not user-friendly and sometimes raises errors, making it difficult for users to incorporate NamedTensors into their workflows.
6. [Scikit-learn Enhancement Proposals (SLEPs) 8, 12, 14](https://github.com/scikit-learn/enhancement_proposals/pull/18)
   - Some of the key points and limitations discussed in these proposals are:
     - Inconsistency in feature name handling: Scikit-learn currently lacks a consistent and comprehensive way to handle and propagate feature names through its pipelines and estimators ([SLEP 8](https://github.com/scikit-learn/enhancement_proposals/pull/18),[SLEP 12](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep012/proposal.html)).
     - Memory intensive for large feature sets: storing and propagating feature names can be memory intensive, particularly in cases where the entire "dictionary" becomes the features, such as in NLP use cases ([SLEP 8](https://github.com/scikit-learn/enhancement_proposals/pull/18),[GitHub issue #35](https://github.com/scikit-learn/enhancement_proposals/issues/35))
     - Sparse matrices: sparse data structures present a challenge for feature name propagation. For instance, the sparse data structure functionality in Pandas 1.0 only supports converting directly to the coordinate format (COO), which can be an issue with transformers such as the OneHotEncoder.transform that has been optimized to construct a CSR matrix ([SLEP 14](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep014/proposal.html))
     - New Data structures: the introduction of new data structures, such as "InputArray" or "DataArray" could lead to more burden for third-party estimator maintainers and increase the learning curve for users. Xarray's "DataArray" is mentioned as a potential alternative, but the proposal mentions that the conversion from a Pandas dataframe to a Dataset is not lossless ([SLEP 12](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep012/proposal.html),[SLEP 14](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep014/proposal.html),[GitHub issue #35](https://github.com/scikit-learn/enhancement_proposals/issues/35)).
     - Dependency on other libraries: solutions that involve using Xarray and/or Pandas to handle feature names come with the challenge of managing dependencies. While a soft dependency approach is suggested, this means users would be able to have/enable the feature only if they have the dependency installed. Xarra-lite's integration with other scientific Python libraries could potentially help with this issue ([GitHub issue #35](https://github.com/scikit-learn/enhancement_proposals/issues/35)).

## References and Previous Discussion

- <code>[[Proposal] Expose Variable without Pandas dependency · Issue #3981 · pydata/xarray · GitHub](https://github.com/pydata/xarray/issues/3981) </code>
- <code>[https://github.com/pydata/xarray/issues/3981#issuecomment-985051449](https://github.com/pydata/xarray/issues/3981#issuecomment-985051449) </code>
- <code>[Lazy indexing arrays as a stand-alone package · Issue #5081 · pydata/xarray · GitHub](https://github.com/pydata/xarray/issues/5081) </code>

### Appendix: Engagement with the Community

We plan to publicize this document on :

- [x] `Xarray dev call`
- [ ] `Scientific Python discourse`
- [ ] `Xarray Github`
- [ ] `Twitter`
- [ ] `Respond to NamedTensor and Scikit-Learn issues?`
- [ ] `Pangeo Discourse`
- [ ] `Numpy, SciPy email lists?`
- [ ] `Xarray blog`

Additionally, We plan on writing a series of blog posts to effectively showcase the implementation and potential of the newly available functionality. To illustrate this, we will use the same example applications as other established libraries (such as Pytorch, sklearn), providing practical demonstrations of how these new data structures can be leveraged.

### Appendix: API Surface

Questions:

1. Document Xarray indexing rules
2. Document use of .oindex and .vindex protocols
3. Do we use `.mean` and `.nanmean` or `.mean(skipna=...)`?
   - Default behavior in named-array should mirror NumPy / the array API standard, not pandas.
   - nanmean is not (yet) in the [array API](https://github.com/pydata/xarray/pull/7424#issuecomment-1373979208). There are a handful of other key functions (e.g., median) that are are also missing. I think that should be OK, as long as what we support is a strict superset of the array API.
4. What methods need to be exposed on Variable?
   - `Variable.concat` classmethod: create two functions, one as the equivalent of `np.stack` and other for `np.concat`
   - `.rolling_window` and `.coarsen_reshape` ?
   - `named-array.apply_ufunc`: used in astype, clip, quantile, isnull, notnull`

#### methods to be preserved from xarray.Variable

```python
# Sorting
   Variable.argsort
   Variable.searchsorted

# NaN handling
   Variable.fillna
   Variable.isnull
   Variable.notnull

# Lazy data handling
   Variable.chunk # Could instead have accessor interface and recommend users use `Variable.dask.chunk` and `Variable.cubed.chunk`?
   Variable.to_numpy()
   Variable.as_numpy()

# Xarray-specific
   Variable.get_axis_num
   Variable.isel
   Variable.to_dict

# Reductions
   Variable.reduce
   Variable.all
   Variable.any
   Variable.argmax
   Variable.argmin
   Variable.count
   Variable.max
   Variable.mean
   Variable.median
   Variable.min
   Variable.prod
   Variable.quantile
   Variable.std
   Variable.sum
   Variable.var

# Accumulate
   Variable.cumprod
   Variable.cumsum

# numpy-like Methods
   Variable.astype
   Variable.copy
   Variable.clip
   Variable.round
   Variable.item
   Variable.where

# Reordering/Reshaping
   Variable.squeeze
   Variable.pad
   Variable.roll
   Variable.shift

```

#### methods to be renamed from xarray.Variable

```python
# Xarray-specific
   Variable.concat # create two functions, one as the equivalent of `np.stack` and other for `np.concat`

   # Given how niche these are, these would be better as functions than methods.
   # We could also keep these in Xarray, at least for now. If we don't think people will use functionality outside of Xarray it probably is not worth the trouble of porting it (including documentation, etc).
   Variable.coarsen # This should probably be called something like coarsen_reduce.
   Variable.coarsen_reshape
   Variable.rolling_window

   Variable.set_dims # split this into broadcast_to and expand_dims


# Reordering/Reshaping
   Variable.stack # To avoid confusion with np.stack, let's call this stack_dims.
   Variable.transpose # Could consider calling this permute_dims, like the [array API standard](https://data-apis.org/array-api/2022.12/API_specification/manipulation_functions.html#objects-in-api)
   Variable.unstack # Likewise, maybe call this unstack_dims?
```

#### methods to be removed from xarray.Variable

```python
# Testing
   Variable.broadcast_equals
   Variable.equals
   Variable.identical
   Variable.no_conflicts

# Lazy data handling
   Variable.compute # We can probably omit this method for now, too, given that dask.compute() uses a protocol. The other concern is that different array libraries have different notions of "compute" and this one is rather Dask specific, including conversion from Dask to NumPy arrays. For example, in JAX every operation executes eagerly, but in a non-blocking fashion, and you need to call jax.block_until_ready() to ensure computation is finished.
   Variable.load # Could remove? compute vs load is a common source of confusion.

# Xarray-specific
   Variable.to_index
   Variable.to_index_variable
   Variable.to_variable
   Variable.to_base_variable
   Variable.to_coord

   Variable.rank # Uses bottleneck. Delete? Could use https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html instead


# numpy-like Methods
   Variable.conjugate # .conj is enough
   Variable.__array_wrap__ # This is a very old NumPy protocol for duck arrays. We don't need it now that we have `__array_ufunc__` and `__array_function__`

# Encoding
    Variable.reset_encoding

```

#### Attributes to be preserved from xarray.Variable

```python
# Properties
   Variable.attrs
   Variable.chunks
   Variable.data
   Variable.dims
   Variable.dtype

   Variable.nbytes
   Variable.ndim
   Variable.shape
   Variable.size
   Variable.sizes

   Variable.T
   Variable.real
   Variable.imag
   Variable.conj
```

#### Attributes to be renamed from xarray.Variable

```python

```

#### Attributes to be removed from xarray.Variable

```python

   Variable.values # Probably also remove -- this is a legacy from before Xarray supported dask arrays. ".data" is enough.

# Encoding
   Variable.encoding

```

### Appendix: Implementation Details

- Merge in VariableArithmetic's parent classes: AbstractArray, NdimSizeLenMixin with the new data structure..

```python
class VariableArithmetic(
 ImplementsArrayReduce,
 IncludeReduceMethods,
 IncludeCumMethods,
 IncludeNumpySameMethods,
 SupportsArithmetic,
 VariableOpsMixin,
):
 __slots__ = ()
 # prioritize our operations over those of numpy.ndarray (priority=0)
 __array_priority__ = 50

```

- Move over `_typed_ops.VariableOpsMixin`
- Build a list of utility functions used elsewhere : Which of these should become public API?
  - `broadcast_variables`: `dataset.py`, `dataarray.py`,`missing.py`
    - This could be just called "broadcast" in named-array.
  - `Variable._getitem_with_mask` : `alignment.py`
    - keep this method/function as private and inside Xarray.
- The Variable constructor will need to be rewritten to no longer accept tuples, encodings, etc. These details should be handled at the Xarray data structure level.
- What happens to `duck_array_ops?`
- What about Variable.chunk and "chunk managers"?

  - Could this functionality be left in Xarray proper for now? Alternative array types like JAX also have some notion of "chunks" for parallel arrays, but the details differ in a number of ways from the Dask/Cubed.
  - Perhaps variable.chunk/load methods should become functions defined in xarray that convert Variable objects. This is easy so long as xarray can reach in and replace .data

- Utility functions like `as_variable` should be moved out of `base_variable.py` so they can convert BaseVariable objects to/from DataArray or Dataset containing explicitly indexed arrays.
