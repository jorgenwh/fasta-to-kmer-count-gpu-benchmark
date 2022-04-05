import sys
import time
import argparse
import numpy as np
import cupy as cp

from shared_memory_wrapper import from_file

from bionumpy.parser import BufferedNumpyParser
from bionumpy.kmers import TwoBitHash
from xpstructures.raggedarray import RaggedArray
from xpstructures.hashtable import Counter

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Eeeeee")
    argument_parser.add_argument("-f", help="Fasta filename", type=str, required=True)
    argument_parser.add_argument("-k", help="length of kmers", type=int, default=31)
    argument_parser.add_argument("-mod", help="Modulo", type=int, default=100000033)
    argument_parser.add_argument("-chunksize", help="Chunk size", type=int, default=10000000)
    argument_parser.add_argument("--cuda", help="Use GPU", action="store_true")
    argument_parser.add_argument("-counter", help="Load counter object from .npz file.", type=str, default=None)
    args = argument_parser.parse_args()

    print(f"Arguments: k={args.k}, chunk-size={args.chunksize}, cuda={args.cuda}")
    
    if args.cuda:
        print("GPU enabled")
    else:
        print("GPU disabled. Run with --cuda to enable GPU")

    xp = cp if args.cuda else np

    #unique_kmers = np.load("uniquekmers.npy")

    parser = BufferedNumpyParser.from_filename(args.f, args.chunksize)
    hasher = TwoBitHash(k=args.k, is_cuda=args.cuda)

    if args.counter != None:
        print("Reading counter from file ...")
        c = from_file(args.counter).counter
        counter = Counter(keys=c._keys, values=c._values, key_dtype=c._key_dtype, value_dtype=c._value_dtype, mod=args.mod)
    else:
        print("Creating counter ...")
        counter = Counter(keys=unique_kmers, mod=args.mod)

    print("Counter ready.")

    if args.cuda:
        print("Moving counter to GPU ...")
        counter.to_cuda()
        print("Counter ready on GPU.")

    print("Counting kmers ...")
    start = time.time_ns()

    for i, chunk in enumerate(parser.get_chunks()):
        if args.cuda:
            chunk.to_cuda()

        sequences = chunk.get_sequences()
        kmers = hasher.get_kmer_hashes(sequences)

        counter.count(kmers)

        print(f"Chunks counted: {i+1}", end="\r")

    print(f"Chunks counted: {i+1}")

    elapsed = time.time_ns() - start
    print(f"Elapsed time: {elapsed / 1e9} seconds")
    print()

