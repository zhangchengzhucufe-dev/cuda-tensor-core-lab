
#ifndef GPUTIMER_H
#define GPUTIMER_H

#include <cuda_runtime.h>

struct GPUTimer{
    cudaEvent_t start;
    cudaEvent_t end;

    GPUTimer(){
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    }

    ~GPUTimer(){
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    
    void Start(){
        
        cudaEventRecord(start, 0);
    }

    void End(){
        cudaEventRecord(end, 0);
    }

    float Elapsed(){
        float elapsed;
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, start, end);
        return elapsed;
    }


};

#endif