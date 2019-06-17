#!/bin/bash  
for((i=2; i<64; i *=2));
do
    echo \n\n\n========BLOCKSIZE=$i========;
    nvcc jacobi.cu -o jacobi -D BLOCKSIZE=$i;
    nvprof ./jacobi;
done 