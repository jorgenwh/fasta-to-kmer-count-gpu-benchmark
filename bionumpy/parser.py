import gzip
import numpy as np
import cupy as cp
from xpstructures import RaggedArray, RaggedView
from bionumpy.sequences import Sequences
from bionumpy.encodings import BaseEncoding, ACTGTwoBitEncoding
NEWLINE = 10

class FileBuffer:
    _buffer_divisor = 1
    COMMENT = 0
    def __init__(self, data, new_lines, is_cuda=False):
        xp = cp if is_cuda else np
        self._data = xp.asanyarray(data)
        self._new_lines = xp.asanyarray(new_lines)
        self._is_validated = False
        self.size = self._data.size
        self._is_cuda = is_cuda

    @classmethod
    def from_raw_buffer(cls, chunk):
        raise NotImplemented

    def validate_if_not(self):
        if not self._is_validated:
            self._validate()

    def to_cuda(self):
        raise NotImplemented

    def to_cpu(self):
        raise NotImplemented

class OneLineBuffer(FileBuffer):
    n_lines_per_entry = 2
    _buffer_divisor = 32

    @classmethod
    def from_raw_buffer(cls, chunk, is_cuda=False):
        xp = cp if is_cuda else np
        new_lines = xp.flatnonzero(chunk==NEWLINE)
        n_lines = new_lines.size
        assert n_lines >= cls.n_lines_per_entry, "No complete entry in buffer"
        new_lines = new_lines[:n_lines - (n_lines % cls.n_lines_per_entry)]
        return cls(chunk[:new_lines[-1] + 1], new_lines, is_cuda=is_cuda)

    def get_sequences(self):
        xp = cp if self._is_cuda else np
        self.validate_if_not()
        sequence_starts = self._new_lines[::self.n_lines_per_entry] + 1
        sequence_lens = self._new_lines[1::self.n_lines_per_entry] - sequence_starts
        indices, shape = RaggedView(
                sequence_starts, 
                sequence_lens, 
                is_cuda=self._is_cuda).get_flat_indices()
        m = indices.size
        d = m % self._buffer_divisor
        seq = xp.empty(m - d + self._buffer_divisor, dtype=self._data.dtype)
        seq[:m] = self._data[indices]
        return Sequences(seq, shape, is_cuda=self._is_cuda)
    
    def _validate(self):
        xp = cp if self._is_cuda else np
        n_lines = self._new_lines.size
        assert n_lines % self.n_lines_per_entry == 0, "Wrong number of lines in buffer"
        header_idxs = self._new_lines[self.n_lines_per_entry - 1:-1:self.n_lines_per_entry] + 1
        assert xp.all(self._data[header_idxs]==self.HEADER)
        self._is_validated = True

    def to_cuda(self):
        self._is_cuda = True
        self._data = cp.asanyarray(self._data)
        self._new_lines = cp.asanyarray(self._new_lines)

    def to_cpu(self):
        if self._is_cuda:
            self._data = self._data.asnumpy()
            self._new_lines = self._new_lines.asnumpy()
            self._is_cuda = False

class OneLineFastaBuffer(OneLineBuffer):
    HEADER = 62
    n_lines_per_entry = 2

class FastQBuffer(OneLineBuffer):
    HEADER = 64
    n_lines_per_entry = 4
    _encoding = BaseEncoding

class BufferedNumpyParser:

    def __init__(self, file_obj, buffer_type, chunk_size=1000000):
        self._file_obj = file_obj
        self._chunk_size = chunk_size
        self._is_finished = False
        self._buffer_type = buffer_type

    @classmethod
    def from_filename(cls, filename, *args, **kwargs):
        if any(filename.endswith(suffix) for suffix in ("fasta", "fa", "fasta.gz", "fa.gz")):
            buffer_type = OneLineFastaBuffer
        elif any(filename.lower().endswith(suffix) for suffix in ("fastq", "fq", "fastq.gz", "fq.gz")):
            buffer_type = FastQBuffer
        else:
            raise NotImplemented
        if filename.endswith(".gz"):
            file_obj = gzip.open(filename, "rb")
        else:
            file_obj = open(filename, "rb")
        return cls(file_obj, buffer_type, *args, **kwargs)

    def get_chunk(self):
        a, bytes_read = self.read_raw_chunk()
        self._is_finished = bytes_read < self._chunk_size
        if bytes_read == 0:
            return None
        
        # Ensure that the last entry ends with newline. Makes logic easier later
        if self._is_finished and a[bytes_read - 1] != NEWLINE:
            a[bytes_read] = NEWLINE
            bytes_read += 1
        return a[:bytes_read]

    def read_raw_chunk(self):
        array = np.empty(self._chunk_size, dtype="uint8")
        bytes_read = self._file_obj.readinto(array)
        return array, bytes_read

    def remove_initial_comments(self):
        if self._buffer_type.COMMENT == 0:
            return 
        for line in self._file_obj:
            if line[0] != self._buffer_type.COMMENT:
                self._file_obj.seek(-len(line), 1)
                break

    def get_chunks(self):
        self.remove_initial_comments()
        chunk = self.get_chunk()
        buff = self._buffer_type.from_raw_buffer(chunk)
        while not self._is_finished:
            buff = self._buffer_type.from_raw_buffer(chunk)
            self._file_obj.seek(buff.size - self._chunk_size, 1)
            yield buff
            chunk = self.get_chunk()
        if chunk is not None and chunk.size:
            yield self._buffer_type.from_raw_buffer(chunk)
