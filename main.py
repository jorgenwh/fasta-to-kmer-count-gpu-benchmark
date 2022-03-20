import sys
import time
import argparse
import numpy as np
import cupy as cp

from bionumpy.parser import BufferedNumpyParser
from bionumpy.kmers import TwoBitHash
from xpstructures.raggedarray import RaggedArray
from xpstructures.hashtable import Counter

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Eeeeee")
    argument_parser.add_argument("-f", help="Fasta filename", type=str, required=True)
    argument_parser.add_argument("-k", help="length of kmers", type=int, default=31)
    argument_parser.add_argument("-chunksize", help="Chunk size", type=int, default=10000000)
    argument_parser.add_argument("--cuda", help="Use GPU", action="store_true")
    args = argument_parser.parse_args()

    if args.cuda:
        print("GPU enabled")
    else:
        print("GPU disabled. Run with --cuda to enable GPU")

    xp = cp if args.cuda else np

    start = time.time_ns()

    print("Setting up data structures ...")
    parser = BufferedNumpyParser.from_filename(args.f, args.chunksize)
    hasher = TwoBitHash(k=args.k, is_cuda=args.cuda)
    # counter = ...

    for i, chunk in enumerate(parser.get_chunks()):
        if args.cuda:
            chunk.to_cuda()

        sequences = chunk.get_sequences()
        kmers = hasher.get_kmer_hashes(sequences)

        # count kmers ...

        print(f"Chunks counted: {i+1}", end="\r")

    print(f"Chunks counted: {i+1}")

    elapsed = time.time_ns() - start
    print(f"Elapsed time: {elapsed / 1e9} seconds")
