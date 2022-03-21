import argparse
from matplotlib import pyplot as plt

def read_runtimes_file(filename):
    f = open(filename)
    k = None
    elapsed = []
    elapsed_cuda = []
    chunk_sizes = []
    prev_cuda = None

    for line in f:
        if line.isspace(): 
            continue
        line = line.strip()

        if line.startswith("Arguments"):
            data = line.split(",")
            cuda = "True" == data[2].split("=")[1]
            prev_cuda = cuda
            if k == None:
                k = int(data[0].split("=")[1])
            if cuda:
                chunk_sizes.append(int(data[1].split("=")[1]))

        if line.startswith("Elapsed"):
            data = line.split()
            if prev_cuda:
                elapsed_cuda.append(float(data[2]))
            else:
                elapsed.append(float(data[2]))

    f.close()
    return k, elapsed, elapsed_cuda, chunk_sizes

def plot(k, elapsed, elapsed_cuda, chunk_sizes, output_filename):
    assert len(elapsed) == len(elapsed_cuda) == len(chunk_sizes)
    plt.title(f"first 100m unique kmers, k={k}")
    plt.xlabel("chunk-size (million)")
    plt.ylabel("seconds")

    #plt.plot([i/1e6 for i in chunk_sizes])
    plt.plot([s/1e6 for s in chunk_sizes], [t for t in elapsed])
    plt.plot([s/1e6 for s in chunk_sizes], [t for t in elapsed_cuda])
    plt.scatter([s/1e6 for s in chunk_sizes], [t for t in elapsed], label="GPU")
    plt.scatter([s/1e6 for s in chunk_sizes], [t for t in elapsed_cuda], label="GPU")

    plt.annotate(f"{(chunk_sizes[-1]/1e6, round(elapsed[-1]))}", (chunk_sizes[-1]/1e6, elapsed[-1]))
    plt.annotate(f"{(chunk_sizes[-1]/1e6, round(elapsed_cuda[-1]))}", (chunk_sizes[-1]/1e6, elapsed_cuda[-1]))

    plt.grid()
    plt.legend()

    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Script to plot runtimes from a runtimes.txt file.")
    arg_parser.add_argument("-f", help="Input filename", type=str, required=True)
    arg_parser.add_argument("-o", help="Output filename", type=str, required=True)
    args = arg_parser.parse_args()
    
    k, elapsed, elapsed_cuda, chunk_sizes = read_runtimes_file(args.f)
    plot(k, elapsed, elapsed_cuda, chunk_sizes, args.o)

