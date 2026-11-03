#pragma once

#include "cuda_check.h"

class GpuTimer {
    private:
        cudaEvent_t start_ev;
        cudaEvent_t stop_ev;
        float ms;

    public:
        GpuTimer(): ms(0) {
            CUDA_CHECK(cudaEventCreate(&start_ev));
            CUDA_CHECK(cudaEventCreate(&stop_ev));
        }

        GpuTimer(const GpuTimer& other) = delete;
        GpuTimer& operator=(const GpuTimer& other) = delete;

        GpuTimer(GpuTimer&& other) = delete;
        GpuTimer& operator=(GpuTimer&& other) = delete;

        ~GpuTimer() {
            CUDA_CHECK(cudaEventDestroy(start_ev));
            CUDA_CHECK(cudaEventDestroy(stop_ev));
        }

        void start() {
            CUDA_CHECK(cudaEventRecord(start_ev));
        }

        void stop() {
            CUDA_CHECK(cudaEventRecord(stop_ev));
            CUDA_CHECK(cudaEventSynchronize(stop_ev));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        }

        float time() const { return ms; }
};
