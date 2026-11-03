# leetgpu: offline LeetGPU runner

Solve [LeetGPU](https://leetgpu.com) challenges locally, with no internet. A solution is graded with the challenge's own `challenge.py`: the same reference implementation, example, functional tests and tolerances that the site uses.

```
leetgpu/
  lgpu.py       the runner (one file, Python + PyTorch)
  tui.py        the terminal UI (needs textual); `lgpu` with no arguments opens it
  solutions/    solutions, committed: solutions/<difficulty>/<id>_<name>/solution.{cu,triton.py,pytorch.py}
  history/      local benchmark record, gitignored: every version measured, its runs and ratings (see below)
  problems/     gitignored clone of AlphaGPU/leetgpu-challenges (CC BY-NC-ND, never pushed)
  .cache/       gitignored: compiled .so files, scan results, pass/fail status, measured GPU ceilings, TUI theme
```

## One-time setup (needs internet)

From the repository root:

```bash
uv sync                          # .venv with PyTorch (CUDA 13.2 wheels) + Triton, ~3 GB
uv run leetgpu/lgpu.py sync      # clone the challenge set into leetgpu/problems/
uv run leetgpu/lgpu.py scan      # measure every perf test once, flag the heavy ones
```

After that, everything works offline. Run `sync` (and then `scan` again) whenever you're online and want new challenges.

## TUI

`lgpu` with no arguments opens a terminal UI (`tui.py`, built on [Textual](https://textual.textualize.io)). It is a front end over the same data as the CLI. `run` and `new` execute as subprocesses of `lgpu.py` with their output streamed live, so the UI never freezes during nvcc or GPU work. The CLI stays dependency-free, and only `tui.py` needs `textual`.

**It works like a website.** Unless the terminal is wide, you only ever see one page:

1. **List page**: the challenges, nothing else. `enter` or a click opens one.
2. **Detail page**: the problem statement and the run output together, no list. Run things from here.

Go back and forward like in a browser: `esc`, `backspace`, `alt+←` or the `‹ back` link in the status line go back; `alt+→` goes forward. Opening a different challenge drops the forward history, as a browser would.

On a wide terminal (120+ columns, 16+ rows) there are no pages: the list, the problem and the output are all on screen at once.

| key | action |
| --- | ------ |
| `enter` / click | open the challenge (wide view: move into the problem pane) |
| `esc` / `backspace` / `alt+←` | back (first closes the filter bar if it is open) |
| `alt+→` | forward |
| `↑` `↓` | pick a challenge on the list page, scroll the problem on the detail page |
| `space` / `b` | scroll the problem down / up |
| `r` | run example + functional tests. From the list page this opens the detail page first. If there's no solution yet, it tells you to press `n` |
| `e` / `p` | example only / full run plus `--perf` (the benchmark is recorded as a version and rated) |
| `v` | versions: swaps the problem pane for every benchmarked version of the solution (time, throughput, rating, speedup vs previous and vs PyTorch). `●` is the file in `solutions/` now, `★` the fastest. `enter` on a row edits its note. From the list page it opens the detail page on the versions. Press `v` again for the problem |
| `n` | copy the starter into `solutions/` (`new`) |
| `l` / `t` | cycle language (cuda, triton, pytorch) / toggle `--t4` |
| `/` | show the filter bar and search by id or name (from a detail page it goes back to the list first). `enter` or `↓` returns to the list and keeps the filter; `esc` clears it and hides the bar |
| `d` / `h` / `o` | cycle difficulty / heavy only / todo only |
| `c` | change the theme; the choice is saved in `.cache/tui.json` and restored on the next launch |
| `x` | cancel the running job (only shown while one runs) |
| `tab` | on a small detail page, switch between problem and output. Elsewhere it moves focus, so you can scroll each pane with the keyboard |
| `?` | key help |
| `q` | quit |

**Status line.** The line under the header always shows whether the selected challenge is solved (`✔ solved`, `✘ failing`, `· not run yet`, `○ not started`), the challenge, difficulty, a mark per language (`cu✔ tr– pt–`, current language underlined; `–` means no solution file) and the heavy flag. It is the first thing kept when the terminal shrinks.

**The detail page adapts to the size.**

| terminal | detail page |
| -------- | ----------- |
| 90+ columns | problem and output side by side |
| 40–89 columns, 26+ rows | problem over output |
| 40–89 columns, fewer rows | one at a time; `tab` switches. Starting a job jumps to the output |
| under 40×10 | a "too small" message |

The list table drops columns as it narrows (difficulty and per-language marks first) and shortens long names with `…`. Output text re-wraps when its pane changes width. Under 24 rows the header hides, and the footer keeps the essential keys first.

While a job runs, the output pane shows a live timer, and the run keys are greyed out.

Progress column: `cu✔` passed, `cu✘` failed, `cu·` not run yet. `⚠` marks a heavy perf test, `?` a challenge that hasn't been scanned. `sync` and `scan` stay CLI-only.

## Daily loop

```bash
alias lgpu='uv run leetgpu/lgpu.py'    # fish: alias lgpu 'uv run leetgpu/lgpu.py'

lgpu list --difficulty easy --todo     # pick one
lgpu show 1                            # read the problem (the full HTML is in problems/.../challenge.html)
lgpu new 1 --lang cuda                 # copies the starter to solutions/easy/1_vector_add/solution.cu
# ... edit solution.cu ...
lgpu run 1 --example                   # just the small example
lgpu run 1                             # example + all functional tests (stops at first failure)
lgpu run 1 --perf                      # + benchmark vs the PyTorch reference; recorded as a version and rated
lgpu run 1 --perf --note "float4 loads"   # same, and say what changed
lgpu history 1                         # all versions side by side
lgpu run 1 --file solutions/easy/1_vector_add/v2.cu --perf   # benchmark another file (also recorded)
```

Languages: `--lang cuda` (default), `triton`, `pytorch`.

## Problem statements

`show` and the TUI turn each `challenge.html` into plain text. The statements are written in LaTeX (the site renders them with MathJax), so `lgpu.py` has a small built-in LaTeX-to-Unicode renderer:

- Symbols and Greek letters become Unicode (`\times` → ×, `\alpha` → α, `\le` → ≤). Fractions, roots and sub/superscripts get plain-text forms: `x/2`, `√(d)`, `x²`, `xᵢ`, `W_qkv`, `∑_(j=0)^(N-1)`.
- `cases`, `matrix` and `align` blocks become aligned text blocks. `<sub>`, `<sup>` and HTML tables are handled too, and paragraphs are reflowed.
- A command the renderer doesn't know prints as its bare name, never as a raw backslash. `lgpu sync` lists any unknown commands in the challenge set after pulling, so you know when to add one to the `TEX_*` tables in `lgpu.py`.
- If a statement is ever hard to read as text, open `challenge.html` in a browser (`show` prints the path).

## Benchmark history and rating

Every `--perf` run whose tests pass is saved, so you can change a kernel, measure again and compare. The TUI does this with `p`.

```bash
lgpu run 1 --perf --note "shared-memory tiling"   # measure + record (--no-record to skip)
lgpu history 1                    # versions: time, throughput, rating (A-E), speedup vs previous, vs PyTorch
lgpu history 1 --md               # the same as a markdown table in the results/TEMPLATE.md columns
lgpu note 1 v2 "float4 loads"     # set or change what a version did
lgpu diff 1                       # previous vs latest version (lgpu diff 1 v1 v3, or `current` for the file on disk)
lgpu calibrate                    # re-measure this GPU's ceilings (done automatically the first time)
```

**Versions.** A version is a hash of the solution file. Benchmarking new code gives `v2`; benchmarking the same code again adds another run to the same version, which shows how noisy the measurement is (`history` warns when runs of the same code differ by 3% or more, so don't trust speedups smaller than that). A copy of each version's source is kept in `history/<challenge>/<lang>/` so you can diff or restore an old one. Records are in `history/<challenge>/<lang>.json`. `--t4` builds are not recorded.

**Rating.** The rating says how close a run got to what this GPU can actually do for the work the challenge requires:

- *memory-bound* (copies, elementwise, reductions, transposes): the minimum bytes the kernel must move (each tensor once, in-out tensors twice) divided by the time, against the best copy bandwidth measured on the local GPU.
- *compute-bound* (matmul, convolution, attention): the reference's FLOPs as PyTorch counts them, against the best fp32 matmul measured on the local GPU.
- It takes whichever limit is slower (the roofline). Ceilings are measured with `lgpu calibrate`, never taken from spec sheets.
- You see a grade, not a percentage: **A** at the limit, nothing worth chasing · **B** close, a little left to gain · **C** room to improve · **D** slow, worth a look at the access pattern · **E** far off, something big is wrong. (The cut-offs are 80%, 60%, 40% and 20% of the ceiling.) The percentage is stored in the history file but not shown, because it wobbles a few points between identical runs and can't be exactly 100%; `lgpu history 1 --details` adds it.

Read it as an estimate. Some kernels can never reach 100%, and the byte model is the algorithmic minimum, so an algorithm that must read data several times scores low through no fault of yours. A result shows `n/a` instead of a grade when it can't be rated honestly: the kernel is faster than 0.05 ms (launch overhead dominates), the work is compute-bound but not fp32 (only an fp32 compute ceiling is modelled), or there is no tensor traffic. Before every timed call the L2 cache is overwritten with a scratch buffer, so a small input can't look faster than DRAM allows. This changed `--perf` timings compared with before.

## How grading works

1. For each test case, `challenge.py` creates the input tensors on the GPU.
2. `reference_impl` runs on copies of the output tensors, giving the expected result.
3. The solution's `solve` runs. For CUDA, `solution.cu` is compiled with `nvcc -O3 -arch=native -lineinfo -shared` and loaded with `ctypes`, and each tensor is passed as a raw device pointer (`tensor.data_ptr()`). For Triton and PyTorch, `solve` is imported and called with the tensors.
4. Outputs are compared with `torch.isclose(atol, rtol)`, using the challenge's tolerances.

PyTorch is only the test bench. The `.cu` file stays plain CUDA C++.

## Heavy challenges

`scan` runs each perf test's generator and reference once, in a child process. It applies a timeout and an RSS watchdog (WSL has no swap, so exceeding RAM would stall the machine), and records peak VRAM, peak RAM and runtime in `.cache/scan.json`. Flags are judged against the current machine every time, so raising the available memory (for example, WSL's limit) lifts the RAM flags without a rescan.

| flag | meaning |
| ---- | ------- |
| `vram X G` | perf test needs more than 85% of the GPU's VRAM |
| `ram X G` | the process peaked above 60% of system RAM |
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
- A wrong answer prints the first mismatching index, the value produced and the expected value. Shrink the problem until you can reason about that index by hand.

## Differences from leetgpu.com

- CUDA compiles with `-arch=native` (the local GPU's architecture) by default. The T4 is `sm_75`, so code that uses newer features (`cp.async`, bf16/FP8 math, more than 64 KB of shared memory) can pass here and fail on the site. `lgpu run <id> --t4` compiles PTX for `compute_75` only, so those features fail to compile locally too. The driver JIT-compiles that PTX for the local GPU, so the first run is slower.
- Timings come from the local GPU, not the site's Tesla T4. Compare versions with each other and with the PyTorch reference, not with the site's leaderboard.
- Perf inputs are random, so a fixed `--seed` (default 0) keeps runs comparable.
- Mojo, CuTe and JAX starters exist in `problems/`, but the runner doesn't support them.
