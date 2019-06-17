#!/bin/bash  
for((i=2; i<=32; i *=2));
do
    echo ========BLOCKSIZE=$i========;
    nvcc jacobi.cu -o jacobi -D BLOCKSIZE=$i;
    nvprof ./jacobi;
done 
