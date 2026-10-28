#pragma once

#include <string>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include "stb_image.h"
#include "stb_image_write.h"

class Image {
    private:
        int w;
        int h;
        int channels_in_file;
        int desired_channels;
        uint8_t *pixels = nullptr;

    public:
        // defalut ctor
        Image() = delete;

        // value ctor
        explicit Image(const std::string& fpath, int channels): desired_channels(channels) {
            std::cout << "image value constructor called" << std::endl;
            pixels = stbi_load(fpath.c_str(), &w, &h, &channels_in_file, desired_channels);
            if (pixels == nullptr) {
                throw std::runtime_error(stbi_failure_reason());
            } else {
                std::cout << "image loaded successfully" << std::endl;
            }
        }

        // copy ctor
        Image(const Image& other) = delete;

        // copy assignment
        Image& operator=(const Image& other) = delete;

        // dtor
        ~Image() {
            std::cout << "image destructor called" << std::endl;
            stbi_image_free(pixels);
        }

        int width() const { return w; }
        int height() const { return h; }
        int channels_file() const { return channels_in_file; }
        int channels_desired() const { return desired_channels; }
        size_t size_bytes() const { return (size_t)w * h * desired_channels; }
        uint8_t* data() const { return pixels; }
};
