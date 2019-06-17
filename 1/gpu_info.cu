#include <iostream>
//#include "error_checks_1.h"

int main()
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "using GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM nums:" << devProp.multiProcessorCount << std::endl;
    std::cout << "shared mem in each block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "max threads per multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor / 32: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    return 0;

}