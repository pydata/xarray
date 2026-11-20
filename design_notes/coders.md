# dataset and variable coders

## current design

When opening datasets, xarray currently applies "coders" to the variables. The exact coders depend on the backend:

Code paths:

1. decode

- `open_dataset`
- `backend.open_dataset`
- `StoreBackendEntrypoint` → `conventions.decode_cf_variables`

2. encode

- `to_*` → `dump_to_store`
  → `encode_dataset_coordinates` (for cf-style coordinate storage)
  → `store.store` → `store.encode` → `store.encode_variable` → `encode_zarr_variable` → `encode_cf_variable`

All backends use `decode_cf_variable` (through the `StoreBackendEntrypoint`), while `zarr` is the only backend that also uses `encode_cf_variable`.

Within `encode_cf_variable`:

- cfdatetime coder
- cftimedelta coder
- cfscaleoffset coder
- cf mask coder
- native enum coder
- nonstring coder
- default fillvalue coder
- boolean coder
  Additionally, `decode_cf_variable` has:
- characterarray coder
- encoded string coder
- objectvlen string coder
- numpy2 string dtype coder
- endian coder

These coders try to apply their operations in a lazy way, such that the actual computation is only triggered when explicitly or implicitly requested.

## new: custom coders

### variable coders

Variable coders will follow a protocol (not a ABC), with two methods:

- `VariableCoder.encode(variable, *, **additional_metadata)`
- `VariableCoder.decode(variable, *, **additional_metadata)`

It will also need a heuristic to decide whether the coder should be applied. This could be:

- a function that, given metadata (dtype, attributes, encoding), decides whether to apply the coder
- the coder performs the check and returns `NotImplemented` if it doesn't fit.

### dataset coders

Dataset coders have a very similar API (still a protocol):

- `DatasetCoder.encode(dataset, **additional_metadata)`
- `DatasetCoder.decode(dataset, **additional_metadata)`

Just like with variable coders it might make sense to have a function that determines whether a coder is applicable given dataset structure and attributes.

### coder pipelines

Coder pipelines describe a set of coding operations of the same type.

TODO: defaults based on the existing attributes (for a CF coder variable and dataset pipeline).

## processing steps

### decoding

The `StoreBackendEntrypoint.open_dataset` method will be split up into different parts (as functions):

- load variables and attributes from the datastore
- apply variable coders (given a ordered list of coders)
- construct a backend dataset from variables, attrs, and the file object
- apply dataset coders (by default contains a CF `coordinates` dataset coder)

Where only the first two will stay in the `StoreBackendEntrypoint` (?).

Then there will be a constructor / object that, given the cf coder settings, constructs a list of variable coders that need to be applied. Backends can then filter these coders to only select those that apply.

### encoding

The steps from `decoding` can be inverted:

- apply dataset coders
- apply variable coders
- split dataset into variables and attrs
