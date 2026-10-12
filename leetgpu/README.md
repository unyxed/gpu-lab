# leetgpu: offline LeetGPU practice

Solve [LeetGPU](https://leetgpu.com) challenges locally, in Zed, with no internet. A solution is graded with the challenge's own `challenge.py`: the same reference implementation, example, functional tests and tolerances that the site uses.

```
leetgpu/
  lgpu.py       the runner (one file, Python + PyTorch)
  solutions/    YOUR code, committed: solutions/<difficulty>/<id>_<name>/solution.{cu,triton.py,pytorch.py}
  problems/     gitignored clone of AlphaGPU/leetgpu-challenges (CC BY-NC-ND, never pushed)
  .cache/       gitignored: compiled .so files, scan results, pass/fail status
```

## One-time setup (needs internet)

From `gpu-lab/` in WSL:

```bash
uv sync                          # .venv with PyTorch (CUDA 13.2 wheels) + Triton, ~3 GB
uv run leetgpu/lgpu.py sync      # clone the challenge set into leetgpu/problems/
uv run leetgpu/lgpu.py scan      # measure every perf test once, flag the heavy ones
```

After that, everything works offline. Run `sync` (and then `scan` again) whenever you're online and want new challenges.

## Daily loop

```bash
alias lgpu='uv run leetgpu/lgpu.py'    # (fish: alias lgpu 'uv run leetgpu/lgpu.py')

lgpu list --difficulty easy --todo     # pick one
lgpu show 1                            # read the problem (the full HTML is in problems/.../challenge.html)
lgpu new 1 --lang cuda                 # copies the starter to solutions/easy/1_vector_add/solution.cu
# ... edit in Zed ...
lgpu run 1 --example                   # just the small example
lgpu run 1                             # example + all functional tests (stops at first failure)
lgpu run 1 --perf                      # + benchmark the perf test vs the PyTorch reference
lgpu run 1 --file solutions/easy/1_vector_add/v2.cu --perf   # benchmark another version
```

Languages: `--lang cuda` (default), `triton`, `pytorch`.

## How grading works

1. For each test case, `challenge.py` creates the input tensors on the GPU.
2. `reference_impl` runs on copies of the output tensors, giving the expected result.
3. Your `solve` runs. For CUDA, `solution.cu` is compiled with `nvcc -O3 -arch=native -lineinfo -shared` and loaded with `ctypes`, and each tensor is passed as a raw device pointer (`tensor.data_ptr()`). For Triton and PyTorch, `solve` is imported and called with the tensors.
4. Outputs are compared with `torch.isclose(atol, rtol)`, using the challenge's tolerances.

PyTorch is only the test bench. Your `.cu` file stays plain CUDA C++.

## Heavy challenges

`scan` runs each perf test's generator and reference once, in a child process. It applies a timeout and an RSS watchdog (WSL has no swap), and records peak VRAM, peak RAM and runtime in `.cache/scan.json`. Flags are judged against the current machine every time, so raising WSL's memory lifts the RAM flags without a rescan.

| flag | meaning |
| ---- | ------- |
| `vram X G` | perf test needs more than 85% of VRAM (8 GB on the RTX 5050) |
| `ram X G` | the process peaked above 60% of WSL's RAM |
| `slow` | reference above 5 s, or input generation above 60 s |
| `timeout`, `ram exceeded`, `vram exceeded`, `probe error` | the probe itself didn't finish |

For a flagged challenge, functional tests still run locally (they're small). Only `--perf` refuses to run, unless you pass `--force`. Benchmark those on leetgpu.com instead. `lgpu list --heavy` shows them all.

## Debugging

```bash
compute-sanitizer --tool memcheck .venv/bin/python leetgpu/lgpu.py run 1          # out-of-bounds, misaligned
compute-sanitizer --tool racecheck .venv/bin/python leetgpu/lgpu.py run 4         # shared-memory races
ncu --set full -k regex:my_kernel .venv/bin/python leetgpu/lgpu.py run 1 --perf --iters 1 --warmup 0
```

- `printf` from kernels works. Use `--example` so the output stays small.
- A wrong answer prints the first mismatching index, the value you produced and the expected value. Shrink the problem until you can reason about that index by hand.

## Differences from leetgpu.com

- CUDA compiles with `-arch=native` (`sm_120`) by default. The T4 is `sm_75`, so code that uses newer features (`cp.async`, bf16/FP8 math, more than 64 KB of shared memory) can pass here and fail on the site. `lgpu run <id> --t4` compiles PTX for `compute_75` only, so those features fail to compile locally too. The driver JIT-compiles that PTX for your GPU, so the first run is slower.
- Timings are from your RTX 5050 (Blackwell, `sm_120`), not the site's Tesla T4. Compare your versions with each other and with the PyTorch reference, not with the site's leaderboard.
- Perf inputs are random, so a fixed `--seed` (default 0) keeps runs comparable.
- Mojo, CuTe and JAX starters exist in `problems/`, but the runner doesn't support them.
