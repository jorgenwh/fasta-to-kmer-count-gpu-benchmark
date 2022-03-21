filename=testreads20m.fa
k=31
chunksizes=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 11000000 12000000 13000000 14000000 15000000 16000000 17000000 18000000 19000000 20000000 21000000 22000000)

echo "Benchmarking CPU"
for size in "${chunksizes[@]}"; do
  echo "chunk-size=$size"
  python main.py -f "$filename" -k "$k" -chunksize "$size" >> runtimes.txt
done

echo "\nBenchmarking GPU"
for size in "${chunksizes[@]}"; do
  echo "chunk-size=$size"
  python main.py -f "$filename" -k "$k" -chunksize "$size" --cuda >> runtimes.txt
done

echo "\nBenchmarking completed"

