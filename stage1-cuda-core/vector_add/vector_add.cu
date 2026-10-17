#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__
void vec_add_kernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vec_add(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vec_add_kernel<<<ceil(n/256.0f), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 1 << 20;

    std::vector<float> A_h;
    std::vector<float> B_h;
    std::vector<float> C_h;

    for (int i{0}; i < N; i++) {
        A_h.push_back(1.0f);
        B_h.push_back(2.0f);
        C_h.push_back(0.0f);
    }

    vec_add(A_h.data(), B_h.data(), C_h.data(), N);

    cudaDeviceSynchronize();

    std::cout << "Result at index 0: " << C_h[0] << " (Expected: 3)" << std::endl;

    return 0;
}
