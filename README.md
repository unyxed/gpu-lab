# gpu-lab

This is my learning repo for CUDA and GPU programming/performance engineering. I'm a junior game developer, and I want to become a strong HPC/GPU performance engineer, so I'm working through the stack from the bottom up.

## I Write All The Following By Hand So I Could Actually Learn

- I write all the kernels
- C++ exercise code
- helpers in `common/`
- LeetGPU solutions

I do use an AI assistant (Claude Code), but only in a narrow role:

- It writes the build and tooling files: `CMakeLists.txt`, `.gitignore`, `pyproject.toml`, the offline LeetGPU runner `leetgpu/lgpu.py`.
- It sometimes writes my commit messages.
- When I ask, it explains a concept, gives me a hint, or reviews code I already wrote. (as a mentor)

It doesn't write or edit my kernels, exercises or solutions, and I don't ask it for complete solutions. I've set this up as a rule in the assistant's project instructions (a local `CLAUDE.md`), so it isn't just a good intention. The git history shows it: the commits that add exercise code are mine.

## Build and run

```bash
cmake -S . -B build -G Ninja                  # once, or after adding an exercise folder
cmake --build build                           # build everything
cmake --build build --target vector_add       # or a single exercise
./build/bin/vector_add
```

- Every `stageN-*/<exercise>/` folder with `.cu` files becomes an executable named after the folder. To add an exercise I create a folder and re-run configure.
- The default build is Release with `-lineinfo`, so Nsight can map metrics back to source lines. `-G` is only added in Debug.
- `compile_commands.json` is exported so clangd works in my editor.

## Layout

```
gpu-lab/
  CMakeLists.txt         one target per exercise folder, found automatically
  common/                small helpers I write myself (CUDA_CHECK, DeviceBuffer, GpuTimer)
  stage1-cuda-core/      execution model, first kernels
  stage2-memory/         shared memory, reductions, scan
  stage3-kernels/        matrix-multiply ladder, ML-flavored kernels
  stage4-transformers/   Python / PyTorch / Triton work
  results/               benchmark tables, one .md per topic
  notes/                 my explain-it write-ups
  leetgpu/               offline LeetGPU runner and my solutions
```

Folders are named by stage and topic, never by date.

## Plan

| Stage           | What I'm learning                                                                                                                                                              | Status      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| 1. CUDA core    | host/device, thread/block/grid indexing, `cudaMalloc`/`cudaMemcpy`, bounds checks, event timing, effective bandwidth, RAII wrappers (`DeviceBuffer`, `GpuTimer`, `CUDA_CHECK`) | in progress |
| 2. Memory       | shared memory, `__syncthreads`, warps and divergence, parallel reduction (naive, shared memory, warp shuffle), scan                                                            | planned     |
| 3. Kernels      | matmul ladder (naive, coalesced, tiled, register-tiled) benchmarked against cuBLAS; softmax and LayerNorm/RMSNorm, including a fused softmax                                   | planned     |
| 4. Transformers | Triton versions of the same kernels, calling my own kernel from PyTorch, profiling with Nsight, roofline analysis                                                              | planned     |

If I continue after that: a FlashAttention-style attention kernel, int8/FP8 quantization, and a small LLM inference engine (which lives in its own repo, not here).

## LeetGPU practice

`leetgpu/` is an offline runner for [LeetGPU](https://leetgpu.com) challenges, so I can solve them locally in my editor. It compiles my `solution.cu` with `nvcc`, loads it with `ctypes`, and grades it with each challenge's own reference implementation, tests and tolerances. It also has a `--perf` mode, and a `--t4` flag that catches code that would not compile on the site's older T4 GPU.

The challenge set is third-party content under CC BY-NC-ND, so it is gitignored and never pushed. Only my solutions are committed. Details are in [`leetgpu/README.md`](leetgpu/README.md).

## Results

I'll add benchmark tables to [`results/`](results/) as I optimize each exercise, using [`results/TEMPLATE.md`](results/TEMPLATE.md). Each table states the input size and how many runs I averaged.

## References

- _Programming Massively Parallel Processors_ (Hwu, Kirk, El Hajj)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
