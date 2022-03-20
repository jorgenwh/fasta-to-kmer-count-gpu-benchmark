from numbers import Number
import numpy as np
import cupy as cp
import time
from .raggedarray import RaggedArray

mempool = cp.get_default_memory_pool()

HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_function__ implementation for RaggedArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

class HashTable:
    """Enables `dict`-like lookup of values for a predefined set of integer keys

    Provides fast lookup for a predefined set of keys. The set of keys must be unique 
    values andcannot be modified after the creation of the `HashTable`. 
    This is in contrast to `dict`, where the set of keys is mutable.
    Indexing both with a single index, or an array_like index is supported. See examples

    Parameters
    ----------
    keys : array_like or `RaggedArray`
           The keys for the lookup
    values : array_like or `RaggedArray`
             The corresponding values
    mod : int, optional
          the modulo-value used to create the hashes
    key_dtype : optional
                the datatype to use for keys. (Must be integer-type)
    value_dtype : optional
                  the datatype to use for the values

    Attributes
    ----------

    Examples
    --------
    >>> table = HashTable([10, 19, 20, 100], [3.14, 2.87, 1.11, 0])
    >>> table[[19, 100]]
    array([2.87, 0.  ])
    """
    def __init__(self, keys, values, mod=None, key_dtype=None, value_dtype=None, safe_mode=True):
        self._is_cuda = False

        if isinstance(keys, RaggedArray):
            self._keys = keys
            self._mod = len(keys)
            self._values = values
            # assert isinstance(values, RaggedArray)
        else:
            keys = np.asanyarray(keys, dtype=key_dtype)
            self.dtype = keys.dtype.type
            if mod is None:
                mod = self._get_mod(keys)
            self._mod = mod
            hashes = self._get_hash(keys)
            args = np.argsort(hashes)
            hashes = hashes[args]
            keys = keys[args]
            self._keys = self._build_ragged_array(keys, hashes)
            if isinstance(values, Number):
                self._values = values
            else:
                values = np.asanyarray(values)
                self._values = RaggedArray(values[args], self._keys.shape)
        self._safe_mode = safe_mode
        self._value_dtype=value_dtype if isinstance(self._values, Number) else self._values.dtype
        self._key_dtype = self._keys.dtype

    def to_cuda(self):
        assert isinstance(self._keys, RaggedArray)
        self._keys.to_cuda()
        if isinstance(self._values, RaggedArray):
            self._values.to_cuda()
        self._is_cuda = True
        #print(f"Hashtable moved to CUDA. Allocated CUDA bytes: {mempool.used_bytes()}")

    def to_cpu(self):
        assert isinstance(self._keys, RaggedArray)
        self._keys.to_cpu()
        if isinstance(self._values, RaggedArray):
            self._values.to_cpu()
        self._is_cuda = False 

    def _get_indices(self, keys):
        xp = cp if self._is_cuda else np

        if isinstance(keys, Number):
            h = self._get_hash(keys)
            possible_keys = self._keys[h]
            offset = xp.flatnonzero(possible_keys==keys)
            return h, offset
        keys = xp.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        offsets = (possible_keys==keys[:, None]).nonzero()[1]
        assert offsets.size == keys.size, (offsets.size, keys.size)
        return hashes, offsets

    def __getitem__(self, keys):
        xp = cp if self._is_cuda else np

        if isinstance(self._values, Number):
            return self._values if isinstance(keys, Number) else xp.full(len(keys), self._values, dtype=self._value_dtype)
        return self._values[self._get_indices(keys)]

    def _fill_values(self):
        xp = cp if self._is_cuda else np

        if isinstance(self._values, Number):
            self._values = self._values * xp.ones_like(self._keys, dtype=self.value_dtype)
            self._values._safe_mode=False

    def __setitem__(self, key, value):
        self._fill_values()
        indices = self._get_indices(key)
        self._values[indices] = value

    def __repr__(self):
        v = self._values
        if isinstance(v, RaggedArray):
            v = self._values.ravel().tolist()
        return f"{self.__class__.__name__}({self._keys.ravel().tolist()}, {v})"

    def _get_mod(self, keys):
        return self.dtype(2 * keys.size - 1) # TODO: make prime

    def _get_hash(self, keys):
        return keys % self._mod

    def _build_ragged_array(self, keys, hashes):
        unique, counts = np.unique(hashes, return_counts=True)
        lengths = np.zeros(self._mod, dtype=int)
        lengths[unique] = counts
        ra = RaggedArray(keys, lengths)
        return ra

    def __eq__(self, other):
        t = np.all(self._keys == other._keys)
        t &= np.all(self._values == other._values)
        return t

    def __add__(self, other):
        if self._safe_mode and not self._keys.equals(other._keys):
            raise ValueError(f"Could not add hash tables with differing keys ({self._keys, other._keys})")
        return HashTable(self._keys, self._values + other._values)

    def __iadd__(self, other):
        if isinstance(other, Number):
            self._values += other
            return self
        if self._safe_mode and not self._keys.equals(other._keys):
            raise ValueError(f"Could not add hash tables with differing keys ({self._keys, other._keys})")
        if isinstance(self._values, Number) and not isinstance(other._values, Number):
            self._fill_values()
        self._values += other._values
        return self

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

@implements(np.zeros_like)
def zeros_like(hash_table, dtype=None):
    dtype = hash_table._value_dtype if dtype is None else dtype
    return hash_table.__class__(hash_table._keys, 0, value_dtype=dtype)

@implements(np.ones_like)
def ones_like(hash_table, dtype=None, shape=None):
    dtype = hash_table._value_dtype if dtype is None else dtype
    return hash_table.__class__(hash_table._keys, 1, value_dtype=dtype)


class Counter(HashTable):
    """HashTable-based counter to count occurances of a predefined set of integers

    Parameters
    ----------
    keys : array_like or `RaggedArray`
           The elements that are to be counted
    values : array_like or `RaggedArray`, default=0
             Initial counts for the elements

    Attributes
    ----------

    Examples
    --------
    >>> counter = Counter([1, 12, 123, 1234, 12345])
    >>> counter.count([1, 0, 123, 123, 123, 2, 12345])
    >>> counter
    Counter([1, 1234, 12, 123, 12345], [1, 0, 0, 3, 1])
    >>> counter.count([12, 12, 12, 12, 12])
    Counter([1, 1234, 12, 123, 12345], [1, 0, 5, 3, 1])
    """
    def __init__(self, keys, values=0, **kwargs):
        # value_dtype=int
        if not("value_dtype" in kwargs and kwargs["value_dtype"] is not None):
            kwargs["value_dtype"]=int
        super().__init__(keys, values, **kwargs)
        self._keys._safe_mode=False
        if isinstance(self._values, RaggedArray):
            self._values._safe_mode=False

    def count(self, keys):
        """ Count the occurances of the predefined set of integers.

        Updates the counts in the Counter with the number of occurances
        of each of its keys in `keys`.

        Parameters
        ----------
        keys : array_like
               The set of integers to count
        """
        #print(f"1 - {mempool.used_bytes()} bytes")
        xp = cp if self._is_cuda else np
        if self._is_cuda:
            assert xp == cp.get_array_module(keys)

        t = time.time()
        keys = xp.asanyarray(keys, dtype=self._key_dtype)
        #print(f"2 - {mempool.used_bytes()} bytes")
        hashes = self._get_hash(keys)
        #print(f"3 - {mempool.used_bytes()} bytes")
        view = self._keys.shape.view(hashes)
        #print(f"4 - {mempool.used_bytes()} bytes")
        mask = xp.flatnonzero(view.lengths)
        #print(f"5 - {mempool.used_bytes()} bytes")
        keys = keys[mask]
        #print(f"6 - {mempool.used_bytes()} bytes")
        hashes = hashes[mask]
        #print(f"7 - {mempool.used_bytes()} bytes")
        view = view[mask]
        #print(f"8 - {mempool.used_bytes()} bytes")
        view.empty_removed = True
        #print(f"9 - {mempool.used_bytes()} bytes")
        rows, offsets = (self._keys[view]==keys[:, None]).nonzero()
        #print(f"10 - {mempool.used_bytes()} bytes")
        if not rows.size:
            return 
        flat_indices = view.ravel_multi_index((rows, offsets))
        #print(f"11 - {mempool.used_bytes()} bytes")
        if isinstance(self._values, Number):
            if self._values==0:
                self._values = RaggedArray(
                    xp.bincount(flat_indices, minlength=self._keys.size),
                    self._keys.shape, dtype=self._value_dtype, safe_mode=False, is_cuda=self._is_cuda)
            else:
                self._values = RaggedArray(
                    self._values + xp.bincount(flat_indices, minlength=self._keys.size),
                    self._keys.shape, dtype=self._value_dtype, is_cuda=self._is_cuda)
        else:
            self._values.ravel()[:] += xp.bincount(flat_indices, minlength=self._values.size)
        #print("T:", time.time()-t)
