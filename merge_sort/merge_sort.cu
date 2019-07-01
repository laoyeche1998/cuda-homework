#include <cstdio>
#include <cmath>
#include <stdlib.h>
#include "error_checks_1.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG
#include <sys/time.h>


#define ARRAY_LEN  2048
void my_swap(int &a, int &b)
{ 
    int temp = a; 
    a = b; 
    b = temp; 
}

int src_arr[ARRAY_LEN];

void generate_array(int * const src_arr)
{
    for(int i = 0; i < ARRAY_LEN; i++)    
        src_arr[i] = i;
    for(int i = 0; i < ARRAY_LEN; i++)
        my_swap(src_arr[rand()%ARRAY_LEN], src_arr[i]);
}


__global__ void vector_add(double *C, const double *A, const double *B, int N /*,const double *hA*/)
{
    // Add the kernel code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void gpu_merge(int *a, int *temp, int sortedsize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // [index1,end1][index2,end2]
    int index1 = idx * 2 * sortedsize; 
    int end1 = index1 + sortedsize;
    int index2 = end1;
    int end2 = index2 + sortedsize;
    int tempIndex = idx * 2 * sortedsize;
    while ((index1 != end1) || (index2 != end2))
    {
        if ((index1 == end1) && (index2 < end2))
            temp[tempIndex++] = a[index2++];
        else if ((index2 == end2) && (index1 < end1))
            temp[tempIndex++] = a[index1++];
        else if (a[index1] < a[index2])
            temp[tempIndex++] = a[index1++];
        else
            temp[tempIndex++] = a[index2++];
    }
}




void cpu_merge_sort(int arr[], int len) {
    int *a = arr;
    int *b = (int *) malloc(len * sizeof(int));
    int seg, start;
    for (seg = 1; seg < len; seg += seg) {
        for (start = 0; start < len; start += seg * 2) {
            int low = start, mid = min(start + seg, len), high = min(start + seg * 2, len);
            int k = low;
            int start1 = low, end1 = mid;
            int start2 = mid, end2 = high;
            while (start1 < end1 && start2 < end2)
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
            while (start1 < end1)
                b[k++] = a[start1++];
            while (start2 < end2)
                b[k++] = a[start2++];
        }
        int *temp = a;
        a = b;
        b = temp;
    }
    if (a != arr) {
        int i;
        for (i = 0; i < len; i++)
            b[i] = a[i];
        b = a;
    }
    free(b);
}


int main(void)
{
    const int N = ARRAY_LEN;
    const int ThreadsInBlock = 1024;
    int *dA, *dB;
    int hA[N], hB[N], hC[N];
    timeval t1, t2; // Structs for timing   


    generate_array(hA);
    CUDA_CHECK(cudaMemcpy((void *)hB, (void *)hA, sizeof(int) * N, cudaMemcpyHostToHost));
    CUDA_CHECK(cudaMalloc((void **)&dA, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc((void **)&dB, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy((void *)dA, (void *)hA, sizeof(int) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void *)dB, (void *)hB, sizeof(int) * N, cudaMemcpyHostToDevice));

    int blockSize = ThreadsInBlock;
    int numBlocks = (N + blockSize - 1) / blockSize;
    dim3 grid(numBlocks), threads(blockSize);

    gettimeofday(&t1, NULL);
    int blocks = numBlocks / 2;
    int sortedsize = N;
    while (blocks > 0)
    {
        gpu_merge<<<grid, threads>>>(dA, dB, sortedsize);
        CUDA_CHECK(cudaMemcpy(dA, dB, N * sizeof(int), cudaMemcpyDeviceToDevice));
        blocks /= 2;
        sortedsize *= 2;
    }
    cudaMemcpy(hC, dA, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    printf("GPU merge sort: %g seconds\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.0e6);

    //// Copy back the results and free the device memory

    
    //#error Copy back the results and free the allocated memory

    gettimeofday(&t1, NULL);
    cpu_merge_sort(hA,N);
    gettimeofday(&t2, NULL);
    printf("CPU merge sort: %g seconds\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.0e6);


    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    return 0;
}