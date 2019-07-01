#include <cstdio>
#include <cmath>
#include <stdlib.h>
#include "error_checks_1.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG
#include <sys/time.h>



template<typename T>
void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

void generate_array(int * ordered_arr,int * unordered_arr, int ARRAY_LEN)
{
    for(int i = 0; i < ARRAY_LEN; i++) // 元素不重复
    {
        ordered_arr[i] = i; // 有序数组
        unordered_arr[i] = i;  // 无序数组
    }   
    for(int i = 0; i < ARRAY_LEN; i++) // 通过交换来打乱顺序
        swap(unordered_arr[rand()%ARRAY_LEN],unordered_arr[i]);

}



__global__ void GPU_merge(int *a, int *b, int seg, const int array_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 一个线程合并两个有序子序列
    int low = idx*2*seg; // 每个线程负责两个子序列，每个子序列的长度是2*seg
    int mid = min(low + seg, array_len);
    int high = min(low + seg * 2, array_len);
    // 最后子序列的长度可能不足seg，导致low+seg或者low+2*seg超出整个序列的长度
    //  所以要向下取整
    int k = low;  //  结果序列的索引值
    int start1 = low, end1 = mid;  // 序列1 [low,mid),mid不属于序列1，是不能取的，
    int start2 = mid, end2 = high;  // 序列2 [mid,high),high不属于序列2，是不能取的，
    while (start1 < end1 && start2 < end2)  // 两个序列都还没全部被归并
        b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];  // 小的排前面
    while (start1 < end1)
        b[k++] = a[start1++];  // 如果序列1非空，那么序列2空，所以把序列1按顺序排到结果序列
    while (start2 < end2)
        b[k++] = a[start2++];  // 如果序列2非空，那么序列1空，所以把序列2按顺序排到结果序列
    // 归并得到了结果序列中的有序序列,[low,high)
}




void CPU_merge_sort(int arr[], int len) {
    int *a = arr;
    int *b = (int *) malloc(len * sizeof(int));
    int seg, start;
    for (seg = 1; seg < len; seg += seg) { // 有序的子序列的长度从1开始增长，到大于等于len时，代表整个序列有序
        for (start = 0; start < len; start += seg * 2) { // 在循环中，每次合并两个有序子序列
            int low = start, mid = min(start + seg, len), high = min(start + seg * 2, len);
            // 最后子序列的长度可能不足seg，导致start+seg或者start+2*seg超出整个序列的长度
            //  所以要向下取整
            int k = low;  //  结果序列的索引值
            int start1 = low, end1 = mid;  // 序列1 [low,mid),mid不属于序列1，是不能取的，
            int start2 = mid, end2 = high;  // 序列2 [mid,high),high不属于序列2，是不能取的，
            while (start1 < end1 && start2 < end2)  // 两个序列都还没全部被归并
                b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];  // 小的排前面
            while (start1 < end1)
                b[k++] = a[start1++];  // 如果序列1非空，那么序列2空，所以把序列1按顺序排到结果序列
            while (start2 < end2)
                b[k++] = a[start2++];  // 如果序列2非空，那么序列1空，所以把序列2按顺序排到结果序列
            // 归并得到了结果序列中的有序序列,[low,high)
        }
        swap(a,b);  // 交换a,b，使得a指向本次merge的结果，b则作为存放下一次merge的结果的目的地址
    }
    if (a != arr) { // 如果a != arr，那么 b==arr，那么就要把结果存到arr
        int i;
        for (i = 0; i < len; i++)
            b[i] = a[i];
        b = a;
    }
    free(b);
}


int main(void)
{
    int N = 0; // 数组长度
    printf("enter array length:\n");
    scanf("%d",&N);
    int *dA, *dB;
    int ordered_arr[N], unordered_arr[N], GPU_ans[N]; 
    // ordered_arr用来存放原始的有序数组，unordered_arr用来存储打乱顺序的数组
    timeval t1, t2; // 用于计时


    generate_array(ordered_arr,unordered_arr,N); // 生成打乱顺序的数组
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
    gettimeofday(&t1, NULL); // 开始计时
    int threads = 16;
    //int blocks = (N+1) / 2;
    int blocks = (N+threads-1)/threads;
    int seg = 1;
    while (seg < N) //每个有序子序列的长度seg从1开始增长，到大于等于len时，代表整个序列有序
    { 
        GPU_merge<<<blocks, threads>>>(dA, dB, seg, N);
        swap(dA,dB);  // 交换dA,dB，使得dA指向本次merge的结果，dB则将作为存放下一次merge的结果的目的地址
        blocks = (blocks+(2-1))/2; // 每次归并后，有序子序列的长度*2，所需的进程数/2
        seg *= 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy((void*)GPU_ans, (void*)dA, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    gettimeofday(&t2, NULL);  // 结束计时
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
    printf("GPU wrong count = %d\n",GPU_wrong_count);
    printf("CPU wrong count = %d\n",CPU_wrong_count);

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