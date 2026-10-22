#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include "cuda_check.h"

bool vec_approx_equality_check(float *A, float *B, int vec_length, float epsilon=0.00001f) {
    for (int i{0}; i < vec_length; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void init_arr(float *v, int vec_length) {
    std::srand(std::time({}));
    for (int i{0}; i < vec_length; i++) {
        v[i] = rand()/ (float)RAND_MAX;
    }
}

__global__ void parallel_vec_add(float* A, float* B, float* C, int vec_length) {
    // calculate which element this thread is responsible for computing
    int work_index = blockDim.x * blockIdx.x + threadIdx.x;

    // perform computation
    if (work_index < vec_length) {
        C[work_index] = A[work_index] + B[work_index];
        printf("C[%d] = %f | A[%d] = %f | B[%d] = %f\n", work_index, C[work_index], work_index , A[work_index], work_index, B[work_index]);
    }
}

void serial_vec_add(float* A, float* B, float* C, int vec_length) {
    for (int i{0}; i < vec_length; i++) {
        C[i] = A[i] + B[i];
    }
}

void unified_memory_example(int vec_length) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparison_result = (float*)malloc(vec_length*sizeof(float));

    CUDA_CHECK(cudaMallocManaged((void**) &A, vec_length*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void**) &B, vec_length*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void**) &C, vec_length*sizeof(float)));

    init_arr(A, vec_length);
    init_arr(B, vec_length);

    int threads = 256;
    int blocks  = cuda::ceil_div(vec_length, threads);

    parallel_vec_add<<<blocks, threads>>>(A, B, C, vec_length);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    serial_vec_add(A, B, comparison_result, vec_length);

    if (vec_approx_equality_check(C, comparison_result, vec_length)) {
        printf("Unified Memory: CPU and GPU answers match\n");
    } else {
        printf("Unified Memory: Error- CPU and GPU answers do not match\n");
    }

    CUDA_CHECK(cudaFree((void*) A));
    CUDA_CHECK(cudaFree((void*) B));
    CUDA_CHECK(cudaFree((void*) C));
    free(comparison_result);
}

int main() {

    int vec_length = 1024; // the number of components in the vector

    unified_memory_example(vec_length);

    return 0;
}
