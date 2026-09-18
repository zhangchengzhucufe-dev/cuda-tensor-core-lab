#include <cstdio>

__global__ void hello(){
    printf("hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    hello<<<2, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}