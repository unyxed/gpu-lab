#pragma once

#include "cuda_check.h"

template<typename T>
class DeviceBuffer {
    private:
        T *devPtr = nullptr;      // device pointer
        size_t bytes = 0;          // buffer size in bytes

        void release() {
            CUDA_CHECK(cudaFree(static_cast<void*>(devPtr)));
            devPtr = nullptr;
            bytes = 0;
        }

    public:
        // default ctor
        DeviceBuffer() = delete;

        // value ctor
        explicit DeviceBuffer(size_t length): bytes(length * sizeof(T)) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devPtr), bytes));
        }

        // copy ctor
        DeviceBuffer(const DeviceBuffer& other) = delete;

        // copy assignment
        DeviceBuffer& operator=(const DeviceBuffer& other) = delete;

        // move ctor
        DeviceBuffer(DeviceBuffer&& other) noexcept: devPtr(other.devPtr), bytes(other.bytes) {
            other.devPtr = nullptr;
            other.bytes = 0;
        }

        // move assignment
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (&other != this) {
                this->release();
                bytes  = other.bytes;
                devPtr = other.devPtr;

                other.bytes  = 0;
                other.devPtr = nullptr;
            }
            return *this;
        }

        // dtor
        ~DeviceBuffer() {
            this->release();
        }

        // get devPtr
        T* getDevicePointer() { return devPtr; }

        // send data from host to device
        void sendDataToDevice(const T* dataPtr) {
            CUDA_CHECK(cudaMemcpy(devPtr, dataPtr, bytes, cudaMemcpyHostToDevice));
        }

        // get data from device to host
        void receiveDataFromDevice(T* hostPtr) {
            CUDA_CHECK(cudaMemcpy(hostPtr, devPtr, bytes, cudaMemcpyDeviceToHost));
        }
};
