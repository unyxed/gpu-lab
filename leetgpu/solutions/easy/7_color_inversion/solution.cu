#include <cuda_runtime.h>

constexpr int RGBA_CHANNELS = 4;

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < width * height) {
        const int rgba_offset = i * RGBA_CHANNELS;
        image[rgba_offset]     = 255 - image[rgba_offset];
        image[rgba_offset + 1] = 255 - image[rgba_offset + 1];
        image[rgba_offset + 2] = 255 - image[rgba_offset + 2];
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
