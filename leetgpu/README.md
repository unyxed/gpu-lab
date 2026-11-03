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

## TUI

`lgpu` with no arguments opens a terminal UI (`tui.py`, built on [Textual](https://textual.textualize.io)). It is a front end over the same data as the CLI. `run` and `new` execute as subprocesses of `lgpu.py` with their output streamed live, so the UI never freezes during nvcc or GPU work. The CLI stays dependency-free, and only `tui.py` needs `textual`.

| key | action |
| --- | ------ |
| `↑` `↓` | pick a challenge; the problem statement shows on the right |
| `space` / `b` | scroll the problem down / up |
| `enter` | run example + functional tests (if there's no solution yet, it tells you to press `n`) |
| `e` / `p` | example only / full run plus `--perf` |
| `n` | copy the starter into `solutions/` (`new`) |
| `l` / `t` | cycle language (cuda, triton, pytorch) / toggle `--t4` |
| `/` | show the filter bar and search by id or name. `enter` or `↓` returns to the list and keeps the filter; `esc` clears it and hides the bar |
| `d` / `h` / `o` | cycle difficulty / heavy only / todo only |
| `c` | change the theme; the choice is saved in `.cache/tui.json` and restored on the next launch |
| `x` | cancel the running job (only shown while one runs) |
| `?` | key help |
| `q` | quit |

The layout stacks vertically when the terminal is narrower than 120 columns. While a job runs, the output pane shows a live timer, and the run keys are greyed out.

Progress column: `cu✔` passed, `cu✘` failed, `cu·` not run yet. `⚠` marks a heavy perf test, `?` a challenge that hasn't been scanned. `sync` and `scan` stay CLI-only.

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

## Problem statements

`show` and the TUI turn each `challenge.html` into plain text. The statements are written in LaTeX (the site renders them with MathJax), so `lgpu.py` has a small built-in LaTeX-to-Unicode renderer:

- Symbols and Greek letters become Unicode (`\times` → ×, `\alpha` → α, `\le` → ≤). Fractions, roots and sub/superscripts get plain-text forms: `x/2`, `√(d)`, `x²`, `xᵢ`, `W_qkv`, `∑_(j=0)^(N-1)`.
- `cases`, `matrix` and `align` blocks become aligned text blocks. `<sub>`, `<sup>` and HTML tables are handled too, and paragraphs are reflowed.
- A command the renderer doesn't know prints as its bare name, never as a raw backslash. `lgpu sync` lists any unknown commands in the challenge set after pulling, so you know when to add one to the `TEX_*` tables in `lgpu.py`.
- If a statement is ever hard to read as text, open `challenge.html` in a browser (`show` prints the path).

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
