filename=testreads20m.fa
outputfilename=new_runtimes.txt
outputplotfilename=new_plot.png
k=31
modulos=(15485863 32452843 49979687 67867967 86028121)
chunksizes=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 11000000 12000000 13000000 14000000 15000000 16000000 17000000 18000000 19000000 20000000 21000000 22000000)

for mod in "${modulos[@]}"; do
  echo "" >> "$outputfilename"
  echo "Modulo=$mod" >> "$outputfilename"

  echo "" >> "$outputfilename"
  echo "Benchmarking CPU" >> "$outputfilename"
  for size in "${chunksizes[@]}"; do
    echo "chunk-size=$size"
    sleep 10
    python main.py -f "$filename" -k "$k" -chunksize "$size" >> "$outputfilename" 
  done

  echo "" >> "$outputfilename"
  echo "Benchmarking GPU" >> "$outputfilename"
  for size in "${chunksizes[@]}"; do
    echo "chunk-size=$size"
    sleep 10
    python main.py -f "$filename" -k "$k" -chunksize "$size" --cuda >> "$outputfilename" 
  done

  echo "" >> "$outputfilename"
  echo "Benchmarking completed" >> "$outputfilename"
done

#python plot.py -f "$outputfilename" -o "$outputplotfilename"
