import numpy as np
import inspect
import itertools

from xray.utils import FrozenOrderedDict
from xray.pycompat import iteritems
from xray.variable import Index
from collections import OrderedDict
import functools

NONE_VAR_NAME = '__values__'


def _encode_variable_name(name):
    if name is None:
        name = NONE_VAR_NAME
    return name


def _decode_variable_name(name):
    if name == NONE_VAR_NAME:
        name = None
    return name


def is_trivial_index(var):
    # if either attributes or encodings are defined
    # the index is not trival.
    if len(var.attrs) or len(var.encoding):
        return False
    # if the index is not a 1d integer array
    if var.ndim > 1 or not var.dtype.kind == 'i':
        return False
    if isinstance(var, Index):
        arange = np.arange(var.size, dtype=var.dtype)
        if np.any(var.values != arange):
            return False
    return True


class AbstractDataStore(object):

    def get_attrs(self):
        raise NotImplementedError

    def get_variables(self):
        raise NotImplementedError

    def get_dimensions(self):
        return list(itertools.chain(*[x.dimensions
                                      for x in self.get_variables().values()]))

    @property
    def variables(self):
        return FrozenOrderedDict((_decode_variable_name(k), v)
                                 for k, v in iteritems(self.get_variables()))

    @property
    def attrs(self):
        return FrozenOrderedDict(self.get_attrs())

    @property
    def dimensions(self):
        return self.get_dimensions()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, tracebook):
        self.close()


class AbstractWritableDataStore(AbstractDataStore):

    def set_dimension(self, d, l):
        raise NotImplementedError

    def set_attribute(self, k, v):
        raise NotImplementedError

    def set_variable(self, k, v):
        raise NotImplementedError

    def sync(self):
        pass

    def store(self, dataset):
        self.set_attributes(dataset.attrs)
        neccesary_dims = [[d for d in v.dimensions]
                          for v in dataset.variables.values()]
        neccesary_dims = set(itertools.chain(*neccesary_dims))
        # set all non-indexes and any index which is not trivial.
        variables = {k: v for k, v in dataset.variables.iteritems()
                     if not (k in neccesary_dims and is_trivial_index(v))}
        self.set_variables(variables)
        #self.set_variables(dataset.variables)

    def set_dimensions(self, dimensions):
        for d, l in iteritems(dimensions):
            self.set_dimension(d, l)

    def set_attributes(self, attributes):
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables):
        for vn, v in iteritems(variables):
            self.set_variable(_encode_variable_name(vn), v)
            self.set_necessary_dimensions(v)

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dimensions, variable.shape):
            if d not in self.dimensions:
                self.set_dimension(d, l)


class AbstractEncodedDataStore(AbstractWritableDataStore):
    """
    AbstractEncodedDataStore is an interface for making a
    DataStore which wraps another DataStore while first passing
    all input/output through an encoding/decoding layer.
    This allows more modular application of things such as
    conforming to CF Conventions.

    There are no explicity restrictions requiring an
    EncodedDataStore to be roundtrip-able, but when this is desired
    (probably often) consider passing implementing
    classes through test_backends:DatasetIOTestCases.

    Requires Implementation
    --------
    encode : function(self, datastore)


    decode : function(self, datastore)

    """
    def encode(self, datastore):
        """
        A function which takes an un-encoded datastore and returns
        a new DataStore (or Dataset) which has been encoded.  Returning
        an InMemoryDataStore for this is encouraged since it avoids
        the xray consistency checks making it faster / more flexible.

        """
        raise NotImplementedError

    def decode(self, datastore):
        """
        A function which takes an encoded datastore and returns
        a new DataStore which has been decoded.  Again consider
        using an InMemoryDataStore, though returning a Dataset
        will work perfectly fine in most situations.

        Also note that directly accessing variable data may cause
        remote DataStores to be loaded into memory.
        See conventions.decode_cf_variable for examples of wrapping
        computations to make them lazy.
        """
        raise NotImplementedError

    @property
    def decoded(self):
        if not hasattr(self, '_decoded'):
            self._decoded = self.decode(self.ds)
        return self._decoded

    def get_dimensions(self):
        return self.decoded.dimensions

    def get_variables(self):
        return self.decoded.variables

    def get_attrs(self):
        return self.decoded.attrs

    def store(self, dataset):
        self.ds.store(self.encode(dataset))
        self.ds.sync()

    def sync(self):
        self.ds.sync()

    def close(self):
        self.ds.close()


def encoding_decorator(encoder, decoder):
    """
    This is a Class decorating function which makes wrapping DataStores
    in additional encoding layers easier.

    Note that often times the encoders and decoders will require arguments
    at class creation time.  To handle this, the encoder and decoder args
    are first inspected.  Any arguments they require are used first, and
    any remaining arguments are passed onto the DataStore being wrapped.

    Parameters
    ----------
    encoder : function
        Takes a Datastore (or Dataset) and returns an encoded Datastore.
    decoder : function
        Takes a Datastore (or Dataset) and returns a decoded Datastore.

    Returns
    -------
    class_wrapper: A function which wraps a DataStore class and turns
        it into an EncodingWrappedDataStore.
    """

    def class_wrapper(cls):
        class EncodingWrappedDataStore(AbstractEncodedDataStore):

            def __init__(self, *args, **kwdargs):
                # NOTE: we assume that any arguments for the encoder
                # and decoder are keyword args.  All position arguments
                # are passed on to the DataStore.
                encoder_argnames = set(inspect.getargspec(encoder).args[1:])
                decoder_argnames = set(inspect.getargspec(decoder).args[1:])
                # make sure there aren't any argument collisions, that would
                # get pretty confusing.
                constructor_args = set(inspect.getargspec(cls.__init__)[1:])
                if constructor_args.intersection(encoder_argnames):
                    bad_args = constructor_args.intersection(encoder_argnames)
                    raise ValueError("encoder and class have overlapping args: %s"
                                     % ', '.join(bad_args))
                if constructor_args.intersection(decoder_argnames):
                    bad_args = constructor_args.intersection(decoder_argnames)
                    raise ValueError("decoder and class have overlapping args: %s"
                                     % ', '.join(bad_args))
                # create a set of keyword arguments for both the encoder and decoder
                encoder_args = {}
                decoder_args = {}
                for k in encoder_argnames.union(decoder_argnames):
                    if k in kwdargs:
                        v = kwdargs.pop(k)
                        if k in encoder_argnames:
                            encoder_args[k] = v
                        if k in decoder_argnames:
                            decoder_args[k] = v
                # create the data store.
                self.ds = cls(*args, **kwdargs)
                # set the encode and decode function using the provided args
                self.encode = functools.partial(encoder, **encoder_args)
                self.decode = functools.partial(decoder, **decoder_args)

        return EncodingWrappedDataStore

    return class_wrapper