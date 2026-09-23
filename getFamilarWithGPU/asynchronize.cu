#include <stdio.h>

__global__ void whoami(int i){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Round %d i am NO.%d, from block %d, thread %d\n", i, tid, blockIdx.x, threadIdx.x);
}


int main(){
    for (int i = 0; i < 3; i++) whoami<<<4, 4>>>(i);
    cudaDeviceSynchronize();
    return 0;
}