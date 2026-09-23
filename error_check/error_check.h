#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)
    do{
        cudaError_t err = (call);
        if (err != cudaSuccess){
            fprintf(stderr, "CUDA error %s at %s: %d %s\n", cudaGetErrorName(err), __FILE__, 
                    __LINE__, cudaGetErrorString(err));
            exit(1);
        }
    }while (0)

#define CUDA_CHECK_KERNEL(call)
    do{
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }while (0)