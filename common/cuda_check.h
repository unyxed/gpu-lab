#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

#define CUDA_CHECK(call)                                                                                                         \
    do {                                                                                                                         \
        cudaError_t err = (call);                                                                                                \
        if (err != cudaSuccess) {                                                                                                \
            fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorString(err), cudaGetErrorName(err));  \
            exit(EXIT_FAILURE);                                                                                                  \
        }                                                                                                                        \
    } while (0)                                                                                                                  \
