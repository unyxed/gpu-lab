#include <cuda_runtime.h>

constexpr int RGB_CHANNELS = 3;

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < width * height) {
        int rgb_offset = i * RGB_CHANNELS;
        float red   = input[rgb_offset];
        float green = input[rgb_offset + 1];
        float blue  = input[rgb_offset + 2];
        float gray  = 0.299f * red + 0.587f * green + 0.114f * blue;
        output[i] = gray;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
