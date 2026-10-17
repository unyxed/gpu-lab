#include <cstdio>

__global__ void hello() {
    int thread_id = threadIdx.x;
    printf("Hello World from device (GPU) thread %d!\n", thread_id);
}

int main() {
    printf("Hello Word from Host (CPU)");

    hello<<<1, 5>>>();

    cudaDeviceSynchronize();

    return 0;
}
