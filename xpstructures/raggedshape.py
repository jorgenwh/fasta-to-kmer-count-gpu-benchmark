from numbers import Number
import numpy as np
import cupy as cp

mempool = cp.get_default_memory_pool()

class ViewBase:
    def __init__(self, codes, lengths=None, is_cuda=False):
        self._is_cuda = is_cuda

        if lengths is None:
            self._codes = codes.view(np.int32)
        else:
            xp = cp if self._is_cuda else np
            starts = xp.asanyarray(codes, dtype=np.int32)
            lengths = xp.asanyarray(lengths, dtype=np.int32)
            if not lengths.size:
                self._codes = xp.array([], dtype=np.int32)
            else:
                self._codes = xp.hstack((starts[:, None], lengths[:, None])).flatten()

        #if self._is_cuda:
            #print(f"View placed on CUDA. Allocated CUDA bytes: {mempool.used_bytes()}")

    def __eq__(self, other):
        xp = cp if self._is_cuda else np
        assert xp == cp.get_array_module(other._codes), "Cannot compare numpy and cupy ViewBases."
        return xp.all(self._codes==other._codes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.starts}, {self.lengths})[CUDA={self._is_cuda}]"

    @property
    def lengths(self):
        """The row lengths"""
        return self._codes[1::2]

    @property
    def starts(self):
        """The start index of each row"""
        return self._codes[::2]

    @property
    def ends(self):
        """The end index of each row"""
        return self.starts + self.lengths

    @property
    def n_rows(self):
        """Number of rows"""
        if isinstance(self.starts, Number):
            return 1
        return self.starts.size

    @property
    def is_cuda(self):
        return self._is_cuda

    def to_cuda(self):
        self._codes = cp.asarray(self._codes)
        self._is_cuda = True
        #print(f"View moved to CUDA. Allocated CUDA bytes: {mempool.used_bytes()}")

    def to_cpu(self):
        self._codes = cp.asnumpy(self._codes)
        self._is_cuda = False 

    def empty_rows_removed(self):
        """Check wheter the `View` with certainty have no empty rows

        Returns
        -------
        bool
            Whether or not it is cerain that this view contins no empty rows
        """
        return hasattr(self, "empty_removed") and self.empty_removed
        
    def ravel_multi_index(self, indices):
        """Return the flattened indices of a set of array indices

        Parameters
        ----------
        indices : tuple
            Tuple containing the row- and column indices to ravel

        Returns
        -------
        array
            array containing the flattenened indices
        """
        xp = cp if self._is_cuda else np
        return self.starts[indices[0]] + xp.asanyarray(indices[1], dtype=np.int32)

    def unravel_multi_index(self, flat_indices):
        """Return array indices for a set of flat indices

        Parameters
        ----------
        indices : index_like
            flat indices to unravel

        Returns
        -------
        tuple
            tuple containing the unravelled row- and column indices
        """
        xp = cp if self._is_cuda else np
        starts = self.starts
        rows = xp.searchsorted(starts, flat_indices, side="right") - 1
        cols = flat_indices - starts[rows]
        return rows, cols

    def index_array(self):
        """Return an array of broadcasted row indices"""
        xp = cp if self._is_cuda else np
        diffs = xp.zeros(self.size, dtype=np.int32)
        diffs[self.starts[1:]] = 1
        return xp.cumsum(diffs)


class RaggedRow:
    def __init__(self, code, is_cuda=False):
        self._is_cuda = is_cuda
        xp = cp if self._is_cuda else np
        code = xp.atleast_1d(code).view(np.int32)
        self.starts = code[0]
        self.legths = code[1]
        self.ends = code[0] + code[1]

class RaggedShape(ViewBase):
    """ Class that represents the shape of a ragged array.
    
    Represents the same information as a list of row lengths.

    Parameters
    ----------
    codes : list or array_like
        Either a list of row lengths, or if ``is_coded=True`` an  array containing row-starts
        and row-lengths as 32-bit numbers.
    is_coded : bool, default=False
        if `False`, the `codes` are interpreted as row lengths.

    Attributes
    ----------
    starts
    lengths
    ends
    """
    def __init__(self, codes, is_coded=False, is_cuda=False):
        if is_coded: # isinstance(codes, np.ndarray) and codes.dtype==np.uint64:
            super().__init__(codes, is_cuda=is_cuda)
            self._is_coded = True
        else:
            xp = cp if is_cuda else np
            lengths = xp.asanyarray(codes, dtype=np.int32)

            if is_cuda:
                starts = xp.pad(xp.cumsum(lengths, dtype=np.int32)[:-1], pad_width=1, mode="constant")[:-1]
            else:
                starts = xp.insert(lengths.cumsum(dtype=np.int32)[:-1], 0, np.int32(0))
            super().__init__(starts, lengths, is_cuda=is_cuda)
            self._is_coded = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lengths})[CUDA={self._is_cuda}]"

    def __str__(self):
        return str(self.lengths)

    def __getitem__(self, index):
        if not isinstance(index, slice) or isinstance(index, Number):
            return NotImplemented
        if isinstance(index, Number):
            index = [index]
        new_codes = self._codes.view(np.uint64)[index].copy().view(np.int32)
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes, is_coded=True, is_cuda=self._is_cuda)

    @property
    def size(self):
        """The sum of the row lengths"""
        if not self.n_rows:
            return 0
        return self.starts[-1] + self.lengths[-1]

    def view(self, indices):
        """Return a view of a subset of rows

        Return a view with row information for the row given by `indices`
        
        Parameters
        ----------
        indices : index_like
            Used to index the rows

        Returns
        -------
        RaggedView
            RaggedView containing information to find the rows specified by `indices`
        """
        if isinstance(indices, Number):
            return RaggedRow(self._codes.view(np.uint64)[indices], is_cuda=self._is_cuda)
        return RaggedView(self._codes.view(np.uint64)[indices], is_cuda=self._is_cuda)

    def view_cols(self, col_slice):
        assert col_slice.step is None

        xp = cp if self._is_cuda else np

        starts = self.starts
        lengths = self.lengths
        ends = self.ends
        if col_slice.start is not None:
            if col_slice.start >= 0:
                starts = starts + xp.minimum(lengths, col_slice.start)
            else:
                starts = starts + xp.maximum(lengths + col_slice.start, 0)
        if col_slice.stop is not None:
            if col_slice.stop >= 0:
                ends = xp.minimum(self.starts + col_slice.stop, ends)
            else:
                ends = xp.maximum(self.ends + col_slice.stop, starts)
        return RaggedView(starts, xp.maximum(0, ends - starts), is_cuda=self._is_cuda)

    def to_dict(self):
        """Return a `dict` of all necessary variables"""
        return {"codes": self._codes}

    @classmethod
    def from_dict(cls, d):
        """Load a `Shape` object from a dict of necessary variables

        Paramters
        ---------
        d : dict
            `dict` containing all the variables needed to initialize a RaggedShape

        Returns
        -------
        RaggedShape
        """
        xp = cp if self._is_cuda else np
        if "offsets" in d:
            return cls(xp.diff(d["offsets"]))
        else:
            return cls(d["codes"])

    @classmethod
    def asshape(cls, shape):
        """Create a `Shape` from either a list of row lengths or a `Shape`
        
        If `shape` is already a `RaggedShape`, do nothing. Else construct a new
        `RaggedShape` object

        Parameters
        ----------
        shape : RaggedShape or array_like

        Returns
        -------
        RaggedShape
        """
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def broadcast_values(self, values, dtype=None):
        """Broadcast the values in a column vector to the data of a ragged array

        The resulting array is such that a `RaggedArray` with `self` as shape will
        have the rows filled with the values in `values. I.e. 
        ``RaggedArray(ret, self)[row, j] = values[row, 1]``

        Parameters
        ----------
        values : array_like
            column vectors with values to be broadcasted
        
        Returns
        -------
        array
            flat array with broadcasted values
        """
        xp = cp if self._is_cuda else np
        values = xp.asanyarray(values)
        assert values.shape == (self.n_rows, 1), (values.shape, (self.n_rows, 1))
        if self.empty_rows_removed():
            return self._broadcast_values_fast(values, dtype)
        values = values.ravel()
        broadcast_builder = xp.zeros(int(self.size) + 1, dtype=dtype)
        broadcast_builder[self.ends[::-1]] -= values[::-1]
        broadcast_builder[0] = 0 
        broadcast_builder[self.starts] += values

        if not self._is_cuda:
            func = xp.logical_xor if values.dtype==bool else xp.add
            return func.accumulate(broadcast_builder[:-1])
        else:
            if values.dtype == bool:
                broadcast_builder = xp.asnumpy(broadcast_builder[:-1])
                np.logical_xor.accumulate(broadcast_builder, out=broadcast_builder)
                return xp.asarray(broadcast_builder)
            else:
                return xp.cumsum(broadcast_builder[:-1])

    def _broadcast_values_fast(self, values, dtype=None):
        xp = cp if self._is_cuda else np
        values = values.ravel()
        broadcast_builder = xp.zeros(int(self.size), dtype=dtype)
        broadcast_builder[self.starts[1:]] = xp.diff(values) # np.diff?
        broadcast_builder[0] = values[0]

        if not self._is_cuda:
            func = np.logical_xor if values.dtype==bool else np.add
            func.accumulate(broadcast_builder, out=broadcast_builder)
            return broadcast_builder
        else:
            if values.dtype == bool:
                broadcast_builder = xp.asnumpy(broadcast_builder)
                np.logical_xor.accumulate(broadcast_builder, out=broadcast_builder)
                broadcast_builder = xp.asarray(broadcast_builder)
            else:
                xp.cumsum(broadcast_builder, out=broadcast_builder)
            return broadcast_builder

class RaggedView(ViewBase):
    """Class to represent a view onto subsets of rows

    Same as RaggedShape, except without the constraint that the rows 
    fill the whole data array. I.e. ``np.all(self.ends[:-1]==self.starts[1:])``
    does not necessarilty hold.

    Parameters
    ----------
    codes : array_like
        Either a list of row starts, or if `lengths` is provided an  array containing row-starts
        and row-lengths as 32-bit numbers.
    lengths : array_like, optional
        the lengths of the rows

    Attributes
    ----------
    starts
    lengths
    ends
    """
    def __getitem__(self, index):
        if isinstance(index, Number):
            return RaggedRow(self._codes.view(np.uint64)[index], is_cuda=self._is_cuda)

        return self.__class__(self._codes.view(np.uint64)[index], is_cuda=self._is_cuda)

    def get_shape(self):
        """ Return the shape of a ragged array containing the view's rows

        Returns
        -------
        RaggedShape
            The shape of a ragged array consisting of the rows in this view
        """
        if not self.n_rows:
            return RaggedShape(self._codes, is_coded=True, is_cuda=self._is_cuda)
        
        xp = cp if self._is_cuda else np
        codes = self._codes.copy()
        xp.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0
        return RaggedShape(codes, is_coded=True, is_cuda=self._is_cuda)

    def get_flat_indices(self):
        """Return the indices into a flattened array

        Return the indices of all the elements in all the
        rows in this view

        Returns
        -------
        array
        """
        xp = cp if self._is_cuda else np

        if not self.n_rows:
            return xp.ones(0, dtype=np.int32), self.get_shape()

        if self.empty_rows_removed():
            return self._get_flat_indices_fast()
        shape = self.get_shape()
        index_builder = xp.ones(int(shape.size) + 1, dtype=np.int32)
        index_builder[shape.ends[::-1]] = 1 - self.ends[::-1]
        index_builder[0] = 0
        index_builder[shape.starts] += self.starts
        xp.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

    def _get_flat_indices_fast(self):
        xp = cp if self._is_cuda else np

        shape = self.get_shape()
        index_builder = xp.ones(int(shape.size), dtype=np.int32)
        index_builder[shape.starts[1:]] = xp.diff(self.starts) - self.lengths[:-1] + 1
        index_builder[0] = self.starts[0]
        xp.cumsum(index_builder, out=index_builder)
        shape.empty_removed = True
        return index_builder, shape
