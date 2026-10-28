#include "cuda_check.h"
#include "device_buffer.h"
#include "image.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

constexpr int RGB_CHANNELS = 3;
constexpr int GRAYSCALE_CHANNELS = 1;

__global__
void color_to_grayscale(uint8_t *out, uint8_t *in, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int gray_offset = row * width + col;

        int rgb_offset = gray_offset * RGB_CHANNELS;
        uint8_t r = in[rgb_offset];
        uint8_t g = in[rgb_offset + 1];
        uint8_t b = in[rgb_offset + 2];

        out[gray_offset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main(int argc, char *argv[]) {
    // 0. handle command line arguments
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <input filename> <output filename> " << std::endl;
        return 1;
    }
    std::vector<std::string> args(argv + 1, argv + argc);

    try {
        // 1. load the image into the ram (host side)
        Image image_in = Image(args[0], RGB_CHANNELS);

        int W = image_in.width();
        int H = image_in.height();

        // 2. create the output image buffer on the ram (host side)
        std::vector<uint8_t> hbuffer_out(W * H);

        // 3. create the input and output image buffer on the gpu ram (device side)
        DeviceBuffer<uint8_t> dbuffer_in(W * H * RGB_CHANNELS);
        DeviceBuffer<uint8_t> dbuffer_out(W * H *GRAYSCALE_CHANNELS);

        // 4. send image data from host to device (device side)
        dbuffer_in.sendDataToDevice(image_in.data());

        // 5. define block and gird dimensions for running the kernel (device side)
        dim3 dimBlock(16, 16);
        dim3 dimGrid(ceil(W / 16.0f), ceil(H/ 16.0f));

        // 6. call the kernel
        color_to_grayscale<<<dimGrid, dimBlock>>>(
            dbuffer_out.getDevicePointer(),
            dbuffer_in.getDevicePointer(),
            W, H
        );

        CUDA_CHECK(cudaGetLastError());

        // 7. recieve the grayscale image from device to host (device side)
        dbuffer_out.receiveDataFromDevice(hbuffer_out.data());

        // 8. write the output grayscale version
        stbi_write_png(args[1].c_str(), W, H, 1, hbuffer_out.data(), W * GRAYSCALE_CHANNELS);

    } catch (const std::runtime_error& e) {
       std::cerr << "error: " << e.what() << std::endl;
       return 2;
    }
    return 0;
}
