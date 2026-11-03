# gpu-lab

A learning repository for CUDA and GPU performance engineering. It works through the stack from the bottom up: execution model, memory hierarchy, optimized kernels, then Triton/PyTorch integration and profiling. Every exercise is checked against a CPU or PyTorch reference, and optimizations are recorded as measured benchmark tables.

## Authorship

All kernels, C++ exercise code, helpers in `common/` and LeetGPU solutions are written by hand.

An AI assistant (Claude Code) is used in a deliberately narrow role:

- It writes build and tooling files: `CMakeLists.txt`, `.gitignore`, `pyproject.toml` and the offline LeetGPU runner `leetgpu/lgpu.py`.
- It sometimes drafts commit messages.
- On request, it explains concepts, gives hints and reviews code that is already written.

It does not write or edit kernels, exercises or solutions. This rule is enforced through the assistant's project instructions, and the git history reflects it: commits that add exercise code are authored by me.

## Requirements

- An NVIDIA GPU with a current driver, and the CUDA toolkit (`nvcc`). Blackwell (`sm_120`) needs 12.8 or newer.
- CMake 3.24+ and Ninja
- A C++20 host compiler
- [`uv`](https://docs.astral.sh/uv/) for the Python tooling. PyTorch is used only as a test bench and reference, never in the CUDA code.

## Build and run

```bash
cmake -S . -B build -G Ninja                  # once, or after adding an exercise folder
cmake --build build                           # build everything
cmake --build build --target vector_add       # or a single exercise
./build/bin/vector_add
```

- Every `stageN-*/<exercise>/` folder with `.cu` files becomes an executable named after the folder. Adding an exercise means creating a folder and re-running configure.
- The default build is Release with `-lineinfo`, so Nsight can map metrics back to source lines. `-G` is only added in Debug.
- Code is built as C++20 for the GPU installed in the machine (`CMAKE_CUDA_ARCHITECTURES=native`).
- `compile_commands.json` is exported for clang-based tooling such as clangd.

## Layout

```
gpu-lab/
  CMakeLists.txt         one target per exercise folder, found automatically
  common/                small helpers (CUDA_CHECK, DeviceBuffer, GpuTimer)
  stage1-cuda-core/      execution model, first kernels
  stage2-memory/         shared memory, reductions, scan
  stage3-kernels/        matrix-multiply ladder, ML-flavored kernels
  stage4-transformers/   Python / PyTorch / Triton work
  results/               benchmark tables, one .md per topic
  notes/                 short write-ups
  leetgpu/               offline LeetGPU runner and solutions
```

Folders are named by stage and topic, never by date.

## Plan

| Stage           | Topics                                                                                                                                                                         | Status      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| 1. CUDA core    | host/device, thread/block/grid indexing, `cudaMalloc`/`cudaMemcpy`, bounds checks, event timing, effective bandwidth, RAII wrappers (`DeviceBuffer`, `GpuTimer`, `CUDA_CHECK`) | in progress |
| 2. Memory       | shared memory, `__syncthreads`, warps and divergence, parallel reduction (naive, shared memory, warp shuffle), scan                                                            | planned     |
| 3. Kernels      | matmul ladder (naive, coalesced, tiled, register-tiled) benchmarked against cuBLAS; softmax and LayerNorm/RMSNorm, including a fused softmax                                   | planned     |
| 4. Transformers | Triton versions of the same kernels, calling a custom kernel from PyTorch, profiling with Nsight, roofline analysis                                                            | planned     |

Possible follow-ups: a FlashAttention-style attention kernel, int8/FP8 quantization, and a small LLM inference engine (kept in a separate repo).

## LeetGPU practice

`leetgpu/` is an offline runner for [LeetGPU](https://leetgpu.com) challenges. It compiles a `solution.cu` with `nvcc`, loads it with `ctypes`, and grades it with each challenge's own reference implementation, tests and tolerances. It also has a `--perf` mode, and a `--t4` flag that catches code that would not compile on the site's older T4 GPU.

The challenge set is third-party content under CC BY-NC-ND, so it is gitignored and not distributed here. Only the solutions are committed. Details are in [`leetgpu/README.md`](leetgpu/README.md).

## Results

Benchmark tables are added to [`results/`](results/) as each exercise is optimized, using [`results/TEMPLATE.md`](results/TEMPLATE.md). Each table states the input size and the number of runs averaged.

## References

- _Programming Massively Parallel Processors_ (Hwu, Kirk, El Hajj)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
