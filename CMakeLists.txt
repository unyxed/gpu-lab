cmake_minimum_required(VERSION 3.24)
project(gpu_lab LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -lineinfo lets Nsight map SASS back to source; -G (device debug) only in Debug.
string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# stb's function bodies, compiled once and linked into every exercise.
add_library(stb_impl STATIC third_party/stb/stb_impl.cpp)
target_include_directories(stb_impl SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party/stb)

# One executable per stage*/<exercise>/ folder that contains .cu files.
# leetgpu/ is deliberately not included (built by leetgpu/lgpu.py).
file(GLOB exercise_dirs CONFIGURE_DEPENDS LIST_DIRECTORIES true
     ${CMAKE_SOURCE_DIR}/stage*/*)
foreach(dir IN LISTS exercise_dirs)
  if(NOT IS_DIRECTORY ${dir})
    continue()
  endif()
  file(GLOB cu_files CONFIGURE_DEPENDS ${dir}/*.cu)
  if(NOT cu_files)
    continue()
  endif()
  file(GLOB cpp_files CONFIGURE_DEPENDS ${dir}/*.cpp)
  get_filename_component(name ${dir} NAME)
  add_executable(${name} ${cu_files} ${cpp_files})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/common)
  # Brings in the stb include path too: #include <stb_image.h>
  target_link_libraries(${name} PRIVATE stb_impl)
endforeach()
