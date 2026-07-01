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

## new: variable coders

Variable coders will follow a protocol (not a ABC), with two methods:

- `VariableCoder.encode(variable, *, **additional_metadata)`
- `VariableCoder.decode(variable, *, **additional_metadata)`

It will also need a heuristic to decide whether the coder should be applied. This could be:

- a function that, given metadata (dtype, attributes, encoding), decides whether to apply the coder
- the coder performs the check and returns `NotImplemented` if it doesn't fit.

### decoding

The `StoreBackendEntrypoint.open_dataset` method will be split up into different parts:

- load variables and attributes from the datastore
- determine coordinate names
- apply coders (given a ordered list of coders)
- split variables into coords and data vars (using the dimension name / coordinate names)
- construct a backend dataset from coords, data vars, attrs, encoding, and the file object

Then there will be a constructor / object that, given the cf coder settings, constructs a list of coders that need to be applied. Backends can then filter these coders to only select those that apply.

### encoding

## new: dataset coders

In
