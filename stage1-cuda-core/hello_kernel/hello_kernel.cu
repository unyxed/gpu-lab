#include <cstdio>

__global__ void hello() {
    printf("Hello Kernel from device (GPU)\n");
    printf("blockIdx.x: %d | blockDim.x: %d | threadIdx.x: %d | global index: %d\n", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    printf("Hello World from host (CPU)\n");

    hello<<<2, 5>>>();

    cudaDeviceSynchronize();

    return 0;
}
