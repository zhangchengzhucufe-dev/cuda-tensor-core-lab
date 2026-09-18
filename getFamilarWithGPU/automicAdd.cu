#include <cuda_runtime.h>
#include <cstdio>

__global__ void atomic_add(int *i){
    atomicAdd(i, 1);
}

__global__ void nomal_add(int *i){
    *i += 1;
}


int main(){

    int *i1;
    cudaMallocManaged((void **)&i1, 4);
    cudaMemset(i1, 0, 4);
    atomic_add<<<3, 3>>>(i1);

    int *i2;
    cudaMallocManaged((void **)&i2, 4);
    cudaMemset(i2, 0, 4);
    nomal_add<<<3, 3>>>(i2);
    cudaDeviceSynchronize();
    printf("atomic add's result is %d\n", *i1);
    printf("nomal add's result is %d\n", *i2);


    cudaFree(i1);
    cudaFree(i2);
    return 0;
}