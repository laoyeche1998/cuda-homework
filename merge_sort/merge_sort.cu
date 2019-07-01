#include <cstdio>
#include <cmath>
#include <stdlib.h>
#include "error_checks_1.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG
#include <sys/time.h>


#define ARRAY_LEN  4096


void generate_array(int * ordered_arr,int * unordered_arr)
{
    for(int i = 0; i < ARRAY_LEN; i++) // 元素不重复
    {
        ordered_arr[i] = i; // 有序数组
        unordered_arr[i] = i;  // 无序数组
    }   
    for(int i = 0; i < ARRAY_LEN; i++) // 通过交换来打乱顺序
        std::swap(unordered_arr[rand()%ARRAY_LEN],unordered_arr[i]);

}


__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    // Add the kernel code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void GPU_merge(int *a, int *temp, int sortedsize)
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




void CPU_merge_sort(int arr[], int len) {
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

    int N = ARRAY_LEN;  // 数组长度
    printf("enter array length:\n");
    scanf("%d",&N);
    //const int ThreadsInBlock = 512;
    int *dA, *dB;
    int ordered_arr[N], unordered_arr[N], GPU_ans[N]; 
    // ordered_arr用来存放原始的有序数组，unordered_arr用来存储打乱顺序的数组
    timeval t1, t2; // 用于计时


    generate_array(ordered_arr,unordered_arr); // 生成打乱顺序的数组
    /* 检查生成的数组
    for(int i=0;i<min(N,32);i++)
    {
        printf("unordered_arr[%d]=%d\n",i,unordered_arr[i]);
    }
    */
    CUDA_CHECK(cudaMalloc((void **)&dA, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc((void **)&dB, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy((void *)dA, (void *)unordered_arr, sizeof(int) * N, cudaMemcpyHostToDevice));

    //*****************************************
    //***********GPU版本的merge sort************
    //*****************************************
    gettimeofday(&t1, NULL);
    int blocks = N / 2;
    int sortedsize = 1;
    while (sortedsize < N)
    {
        GPU_merge<<<blocks, 1>>>(dA, dB, sortedsize);
        CUDA_CHECK(cudaMemcpy((void*)dA, (void*)dB, N * sizeof(int), cudaMemcpyDeviceToDevice));
        blocks /= 2;
        sortedsize *= 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy((void*)GPU_ans, (void*)dA, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    gettimeofday(&t2, NULL);
    printf("GPU merge sort: %g seconds\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.0e6);
    //*****************************************
    //*****************************************


    //*****************************************
    //***********CPU版本的merge sort************
    //*****************************************
    gettimeofday(&t1, NULL);
    CPU_merge_sort(unordered_arr,N);
    gettimeofday(&t2, NULL);
    printf("CPU merge sort: %g seconds\n", t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.0e6);
    //*****************************************
    //*****************************************
    
    // 检查排序是否正确
    int GPU_wrong_count=0,CPU_wrong_count=0;
    for(int i=0;i<N;i++)
    {
        if(unordered_arr[i]!=ordered_arr[i])
        {
            CPU_wrong_count++;
            printf("CPU wrong at i[%d]=%d \n",i,unordered_arr[i]);
        }
        if(GPU_ans[i]!=ordered_arr[i])
        {
            GPU_wrong_count++;
            printf("GPU wrong at i[%d]=%d \n",i,GPU_ans[i]);
        }
    }
    printf("CPU wrong count = %d\n",CPU_wrong_count);
    printf("GPU wrong count = %d\n",GPU_wrong_count);

    /*  检查排序结果
    for(int i=0;i<min(N,32);i++)
    {
        printf("unordered_arr[%d]=%d  GPU_ans[%d]=%d\n",i,unordered_arr[i],i,GPU_ans[i]);
    }*/

    // 释放device的内存
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    return 0;
}