#!/usr/bin/env python3
"""Offline LeetGPU runner: grade solutions locally with the challenges' own tests and references.

Usage (from gpu-lab/, via `uv run`):
    uv run leetgpu/lgpu.py                      no arguments: open the TUI (leetgpu/tui.py, needs textual)
    uv run leetgpu/lgpu.py sync                 clone/update the challenge set into leetgpu/problems/
    uv run leetgpu/lgpu.py scan                 measure each perf test's memory/time -> heavy flags
    uv run leetgpu/lgpu.py list [--heavy]       list challenges with flags and your progress
    uv run leetgpu/lgpu.py show 1               print a problem statement
    uv run leetgpu/lgpu.py new 1 --lang cuda    copy the starter into leetgpu/solutions/
    uv run leetgpu/lgpu.py run 1 [--perf]       compile + run example and functional tests (+ benchmark)
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = ROOT / "problems"  # gitignored clone of the upstream challenge repo
CHALLENGES = PROBLEMS / "challenges"
SOLUTIONS = ROOT / "solutions"
CACHE = ROOT / ".cache"
SCAN_FILE = CACHE / "scan.json"
STATUS_FILE = CACHE / "status.json"
UPSTREAM_URL = "https://github.com/AlphaGPU/leetgpu-challenges.git"
LOCAL_UPSTREAM = ROOT.parent.parent / "leetgpu-challenges"

DIFFICULTIES = ("easy", "medium", "hard")
LANGS = {
    "cuda": ("solution.cu", "starter.cu"),
    "triton": ("solution.triton.py", "starter.triton.py"),
    "pytorch": ("solution.pytorch.py", "starter.pytorch.py"),
}

# Heavy-flag thresholds, judged against the current machine so raising WSL memory updates them.
CUDA_CONTEXT_GB = 0.6  # CUDA context + cuBLAS workspace, not counted by torch's allocator
VRAM_FRACTION = 0.85  # leave headroom for your solution's own scratch buffers
RAM_FRACTION = 0.6
SLOW_REF_MS = 5000
SLOW_GEN_S = 60

GB = 1024**3


# ---------------------------------------------------------------- challenge discovery


def all_challenges():
    out = []
    for diff in DIFFICULTIES:
        for d in sorted((CHALLENGES / diff).glob("*_*")):
            m = re.match(r"(\d+)_", d.name)
            if m and (d / "challenge.py").exists():
                out.append({"id": int(m.group(1)), "diff": diff, "dir": d, "slug": d.name})
    return sorted(out, key=lambda c: c["id"])


def find_challenge(query):
    if not CHALLENGES.exists():
        sys.exit("No problems found. Run: uv run leetgpu/lgpu.py sync")
    chs = all_challenges()
    if query.isdigit():
        hits = [c for c in chs if c["id"] == int(query)]
    else:
        hits = [c for c in chs if query.lower() in c["slug"].lower()]
    if len(hits) != 1:
        names = ", ".join(c["slug"] for c in hits[:10]) or "none"
        sys.exit(f"'{query}' matched {len(hits)} challenges: {names}")
    return hits[0]


def solution_dir(ch):
    return SOLUTIONS / ch["diff"] / ch["slug"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_challenge(ch, device="cuda"):
    if str(CHALLENGES) not in sys.path:
        sys.path.insert(0, str(CHALLENGES))
    mod = load_module(ch["dir"] / "challenge.py", f"challenge_{ch['id']}")
    return mod.Challenge(device=device)


def as_list(cases):
    return [cases] if isinstance(cases, dict) else list(cases)


# ---------------------------------------------------------------- tensors

LAZY = {"RandTensor", "RandnTensor", "RandIntTensor", "FullTensor", "OutTensor"}


def materialize(v, device):
    """Turn the lazy input specs from core/challenge_base.py into real tensors."""
    import torch

    kind = type(v).__name__
    if kind not in LAZY:
        return v
    dt = getattr(torch, v.dtype)
    if kind == "RandTensor":
        return torch.empty(v.shape, device=device, dtype=dt).uniform_(v.low, v.high)
    if kind == "RandnTensor":
        return torch.empty(v.shape, device=device, dtype=dt).normal_(v.mean, v.std)
    if kind == "RandIntTensor":
        return torch.randint(v.low, v.high, v.shape, device=device, dtype=dt)
    if kind == "FullTensor":
        return torch.full(v.shape, v.value, device=device, dtype=dt)
    return torch.empty(v.shape, device=device, dtype=dt)


def case_bytes(case):
    """Bytes of all inputs/outputs in a test case, without allocating anything."""
    import torch

    total = 0
    for v in case.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
        elif type(v).__name__ in LAZY:
            n = 1
            for s in v.shape:
                n *= s
            total += n * getattr(torch, v.dtype).itemsize
    return total


def meta_perf_bytes(ch):
    """Size of the perf test computed on the 'meta' device (shapes only, no memory)."""
    try:
        c = load_challenge(ch, device="meta")
        return sum(case_bytes(case) for case in as_list(c.generate_performance_test()))
    except Exception:
        return None  # generator does data-dependent work; the real probe will measure it


def out_keys(sig):
    return [k for k, (_, d) in sig.items() if d in ("out", "inout")]


# ---------------------------------------------------------------- solutions


def build_cuda(src, t4=False):
    """Compile solution.cu to a shared library; unique name per content so reloads work.

    t4: PTX for compute_75 only (the site's Tesla T4), JIT-compiled by the driver on this GPU,
    so sm_80+ features (cp.async, bf16 math, ...) fail to compile here like they would there.
    """
    arch = ["-arch=compute_75", "-code=compute_75"] if t4 else ["-arch=native"]
    digest = hashlib.sha1(src.read_bytes() + str(arch).encode()).hexdigest()[:12]
    out = CACHE / "build" / f"{src.parent.name}_{digest}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["nvcc", "-O3", *arch, "-lineinfo", "-shared", "-Xcompiler", "-fPIC"]
        cmd += ["-o", str(out), str(src)]
        print("$", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            sys.exit("compile failed")
    return out


def load_solve(src, lang, sig, t4=False):
    """Return solve(case_dict) that calls the user's solve with args in signature order."""
    import torch

    names = list(sig)
    if lang == "cuda":
        lib = ctypes.CDLL(str(build_cuda(src, t4)))
        lib.solve.argtypes = [t for t, _ in sig.values()]
        lib.solve.restype = None

        def to_c(v, ctype):
            if isinstance(v, torch.Tensor):
                assert v.is_contiguous(), "non-contiguous tensor passed to CUDA solve"
                return ctypes.cast(ctypes.c_void_p(v.data_ptr()), ctype)
            return ctype(v)

        return lambda case: lib.solve(*(to_c(case[n], sig[n][0]) for n in names))

    mod = load_module(src, f"solution_{lang}")
    return lambda case: mod.solve(*(case[n] for n in names))


# ---------------------------------------------------------------- grading


def compare(got, exp, atol, rtol):
    import torch

    if got.shape != exp.shape:
        return f"shape {tuple(got.shape)} != expected {tuple(exp.shape)}"
    if got.dtype == torch.bool or exp.dtype == torch.bool:
        bad = got != exp
    else:
        bad = ~torch.isclose(got, exp.to(got.dtype), atol=atol, rtol=rtol)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return None
    idx = tuple(int(i) for i in bad.nonzero()[0])
    msg = f"{n_bad}/{got.numel()} mismatches, first at {idx}: got {got[idx].item()}, "
    msg += f"expected {exp[idx].item()}"
    if got.dtype.is_floating_point:
        msg += f", max abs diff {(got.float() - exp.float()).abs().nan_to_num(float('inf')).max().item():.3g}"
    return msg


def run_case(c, sig, case, solve):
    """Reference on cloned outputs first (so a buggy solution can't corrupt its inputs), then yours."""
    import torch

    ref = dict(case)
    for k in out_keys(sig):
        ref[k] = case[k].clone()
    c.reference_impl(**ref)
    solve(case)
    torch.cuda.synchronize()
    errors = []
    for k in out_keys(sig):
        err = compare(case[k], ref[k], c.atol, c.rtol)
        if err:
            errors.append(f"{k}: {err}")
    return errors


def bench(fn, warmup, iters):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def describe_case(case):
    import torch

    parts = []
    for k, v in case.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}{list(v.shape)}")
        elif not isinstance(v, (list, dict)):
            parts.append(f"{k}={v}")
    return " ".join(parts)


# ---------------------------------------------------------------- heavy flags


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True))


def machine():
    """Total VRAM and RAM in bytes, without importing torch (used by `list`)."""
    vram = 8 * GB
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        vram = int(out.split()[0]) * 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return vram, meminfo("MemTotal")


def meminfo(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0


def assess(entry, vram_total, ram_total):
    """Reasons a challenge's perf test is risky on this machine (empty list = safe)."""
    if not entry:
        return []
    reasons = []
    status = entry.get("status", "ok")
    if status != "ok":
        reasons.append(status)
    vram = entry.get("perf_vram")
    if vram is None:
        vram = entry.get("meta_bytes")
    if vram is not None and vram / GB + CUDA_CONTEXT_GB > VRAM_FRACTION * vram_total / GB:
        reasons.append(f"vram {vram / GB + CUDA_CONTEXT_GB:.1f}G")
    rss = entry.get("perf_rss")
    if rss and rss > RAM_FRACTION * ram_total:
        reasons.append(f"ram {rss / GB:.1f}G")
    if entry.get("ref_ms", 0) > SLOW_REF_MS or entry.get("gen_s", 0) > SLOW_GEN_S:
        reasons.append(f"slow (ref {entry.get('ref_ms', 0):.0f} ms, gen {entry.get('gen_s', 0):.0f} s)")
    return reasons


def flag_for(ch):
    scan = read_json(SCAN_FILE, {})
    return assess(scan.get(ch["slug"]), *machine())


def file_hash(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def probe(ch):
    """Child process: run the perf test generator + reference once and report peak usage."""
    import resource

    import torch

    c = load_challenge(ch)
    sig = c.get_solve_signature()
    t0 = time.perf_counter()
    cases = [
        {k: materialize(v, "cuda") for k, v in case.items()}
        for case in as_list(c.generate_performance_test())
    ]
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    ref_ms = 0.0
    for case in cases:
        ref = dict(case)
        for k in out_keys(sig):
            ref[k] = case[k].clone()
        for _ in range(2):  # time the 2nd call: the 1st includes cuBLAS/kernel loading
            t0 = time.perf_counter()
            c.reference_impl(**ref)
            torch.cuda.synchronize()
        ref_ms = max(ref_ms, (time.perf_counter() - t0) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(
        json.dumps(
            {
                "perf_vram": torch.cuda.max_memory_reserved(),
                "perf_rss": rss,
                "gen_s": round(gen_s, 2),
                "ref_ms": round(ref_ms, 1),
            }
        )
    )


def run_probe(ch, timeout, rss_limit):
    """Run probe() in a subprocess with a timeout and an RSS watchdog (WSL has no swap)."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "_probe", str(ch["id"])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    killed = {}

    def watchdog():
        while proc.poll() is None:
            try:
                status = Path(f"/proc/{proc.pid}/status").read_text()
                rss = int(re.search(r"VmRSS:\s+(\d+)", status).group(1)) * 1024
            except (OSError, AttributeError):
                rss = 0
            if rss > rss_limit:
                killed["rss"] = rss
                proc.kill()
            time.sleep(0.2)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return {"status": f"timeout >{timeout}s"}
    if "rss" in killed:
        return {"status": "ram exceeded", "perf_rss": killed["rss"]}
    if proc.returncode != 0:
        last = (err.strip().splitlines() or ["?"])[-1]
        if "OutOfMemoryError" in err or "out of memory" in err:
            return {"status": "vram exceeded"}
        if proc.returncode < 0:
            return {"status": f"killed (signal {-proc.returncode})"}
        return {"status": f"probe error: {last[:120]}"}
    return {"status": "ok", **json.loads(out.strip().splitlines()[-1])}


# ---------------------------------------------------------------- commands


def cmd_sync(args):
    if (PROBLEMS / ".git").exists():
        r = subprocess.run(["git", "-C", str(PROBLEMS), "pull", "--ff-only"])
        if r.returncode != 0:
            print("pull failed (offline?) - keeping the existing copy")
    else:
        src = str(LOCAL_UPSTREAM) if (LOCAL_UPSTREAM / ".git").exists() else UPSTREAM_URL
        print(f"cloning {src} -> {PROBLEMS}")
        if subprocess.run(["git", "clone", src, str(PROBLEMS)]).returncode != 0:
            sys.exit("clone failed")
        subprocess.run(["git", "-C", str(PROBLEMS), "remote", "set-url", "origin", UPSTREAM_URL])
    chs = all_challenges()
    TEX_UNKNOWN.clear()
    for ch in chs:  # safety net: new challenges may use LaTeX the text converter doesn't know yet
        problem_text(ch)
    if TEX_UNKNOWN:
        print(f"note: unknown LaTeX commands in some statements: {', '.join(sorted(TEX_UNKNOWN))}")
        print("      they show as bare words until added to TEX_SYMBOLS (or a sibling table) in lgpu.py")
    print(f"{len(chs)} challenges. Next: uv run leetgpu/lgpu.py scan")


def cmd_scan(args):
    chs = [find_challenge(q) for q in args.ids] if args.ids else all_challenges()
    scan = read_json(SCAN_FILE, {})
    vram_total, _ = machine()
    rss_limit = int(meminfo("MemAvailable") * 0.8)
    for i, ch in enumerate(chs, 1):
        h = file_hash(ch["dir"] / "challenge.py")
        old = scan.get(ch["slug"])
        if old and old.get("hash") == h and not args.force:
            continue
        entry = {"hash": h, "meta_bytes": meta_perf_bytes(ch)}
        mb = entry["meta_bytes"]
        if mb is not None and mb * 2 / GB + CUDA_CONTEXT_GB > vram_total / GB:
            entry["status"] = "skipped: inputs alone exceed vram"
        else:
            entry.update(run_probe(ch, args.timeout, rss_limit))
        scan[ch["slug"]] = entry
        write_json(SCAN_FILE, scan)  # save as we go so an interrupted scan keeps its progress
        flag = ", ".join(assess(entry, *machine())) or "ok"
        print(f"[{i}/{len(chs)}] {ch['slug']:<40} {flag}")


def cmd_list(args):
    scan = read_json(SCAN_FILE, {})
    status = read_json(STATUS_FILE, {})
    vram_total, ram_total = machine()
    print(f"{'id':>4}  {'diff':<6} {'challenge':<42} {'progress':<22} perf-test flag")
    for ch in all_challenges():
        if args.difficulty and ch["diff"] != args.difficulty:
            continue
        entry = scan.get(ch["slug"])
        reasons = assess(entry, vram_total, ram_total)
        if args.heavy and not reasons:
            continue
        langs = [lang for lang, (f, _) in LANGS.items() if (solution_dir(ch) / f).exists()]
        if args.todo and langs:
            continue
        prog = " ".join(f"{lang}:{status.get(ch['slug'], {}).get(lang, '-')}" for lang in langs)
        flag = ("HEAVY: " + ", ".join(reasons)) if reasons else ("" if entry else "? (not scanned)")
        print(f"{ch['id']:>4}  {ch['diff']:<6} {ch['slug'][:42]:<42} {prog:<22} {flag}")


# ---------------------------------------------------------------- LaTeX -> plain text
# Challenge statements are written in LaTeX (MathJax on the site). `tex_to_text` renders it as readable
# Unicode for `show` and the TUI. A command it doesn't know degrades to its bare name (never a raw
# backslash) and is recorded in TEX_UNKNOWN, which `sync` reports so it can be added to the tables here.

NB = "\xa0"  # deliberate spacing; ordinary runs of spaces are collapsed at the end
ROW, COL = "\x01", "\x00"  # row / column separators inside environments

_GREEK_LOWER = (
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho "
    "sigma tau upsilon phi varphi chi psi omega"
)
TEX_SYMBOLS = {
    **dict(zip(_GREEK_LOWER.split(), "αβγδεεζηθϑικλμνξπρστυφφχψω")),
    **dict(zip("Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split(), "ΓΔΘΛΞΠΣΥΦΨΩ")),
    # operators and relations
    "times": "×", "cdot": "·", "cdots": "⋯", "ldots": "…", "dots": "…", "pm": "±", "mp": "∓", "div": "÷",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠", "approx": "≈", "sim": "∼",
    "equiv": "≡", "propto": "∝", "ll": "≪", "gg": "≫",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cap": "∩", "cup": "∪", "setminus": "∖",
    "odot": "⊙", "oplus": "⊕", "otimes": "⊗", "circ": "∘", "ast": "∗", "star": "⋆",
    "mid": "|", "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "Leftrightarrow": "⇔", "mapsto": "↦", "uparrow": "↑", "downarrow": "↓",
    "forall": "∀", "exists": "∃", "wedge": "∧", "vee": "∨", "neg": "¬", "emptyset": "∅",
    "infty": "∞", "partial": "∂", "nabla": "∇", "ell": "ℓ", "top": "ᵀ", "bmod": " mod ",
    "sum": "∑", "prod": "∏", "int": "∫", "bigcup": "⋃", "bigcap": "⋂",
    # delimiters
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "langle": "⟨", "rangle": "⟩",
    "lVert": "‖", "rVert": "‖", "lvert": "|", "rvert": "|", "vert": "|", "Vert": "‖",
}
TEX_FUNCS = set("log ln exp sin cos tan tanh sinh cosh max min sup inf lim arg det".split())
TEX_LIMITS = {"sum", "prod", "int", "bigcup", "bigcap", "max", "min", "lim", "sup", "inf"}  # take under/over limits
TEX_SIZERS = set(
    "left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr".split()
)  # size hints: the delimiter that follows is kept, the command dropped
TEX_TEXTLIKE = set("text texttt textbf textit mathrm mathbf mathit mathsf mathcal mathtt mathscr operatorname boldsymbol bm".split())
TEX_NOOP = {"displaystyle", "textstyle", "scriptstyle", "limits", "nolimits"}
TEX_ACCENTS = {
    "hat": "̂", "widehat": "̂", "tilde": "̃", "widetilde": "̃",
    "bar": "̄", "overline": "̄", "vec": "⃗", "dot": "̇", "ddot": "̈",
}
TEX_SPACING = {",": " ", ";": " ", ":": " ", "!": "", " ": " ", "quad": NB * 2, "qquad": NB * 4}
TEX_BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"}
TEX_BRACKETS = {"matrix": ("", ""), "pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": ("{", "}"),
                "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖")}
TEX_UNKNOWN = set()

_SUP_DIGITS = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
_SUB_DIGITS = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUP_LETTERS = {"n": "ⁿ", "i": "ⁱ", "T": "ᵀ", "ᵀ": "ᵀ"}
_SUB_LETTERS = dict(zip("aehijklmnoprstuvx", "ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))


def _simple(s):
    """True if s can follow / or √ or _ without parentheses."""
    return len(s) <= 1 or re.fullmatch(r"[\w.]+", s) is not None or (s[0] == "√" and _simple(s[1:]))


def _wrap(s):
    return s if _simple(s) else f"({s})"


def tex_script(kind, arg, explicit=False):
    """Sub/superscript: real Unicode ones when the whole argument has them (x², xᵢ, 10⁻⁶), else x_(a,b) / x^(n+1).
    explicit=True (limits of ∑, ∫, ...) always uses the _() / ^() form, which reads better than stacked glyphs."""
    arg = arg.replace("−", "-").strip(" ")
    if not arg:
        return ""
    if not explicit:
        if all(ch in "0123456789+-=()" for ch in arg):
            return arg.translate(_SUP_DIGITS if kind == "^" else _SUB_DIGITS)
        letters = _SUP_LETTERS if kind == "^" else _SUB_LETTERS
        if arg in letters:
            return letters[arg]
    return kind + (arg if _simple(arg) else f"({arg})")


class _Tex:
    """Small recursive-descent renderer for the LaTeX subset the challenges use."""

    def __init__(self, src):
        self.s = re.sub(r"\s+", " ", src.strip())
        self.i = 0
        self.limits = False  # the previous token was ∑/∫/max/...: its scripts are limits

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def seq(self, brace=False, env=False):
        out, s = [], self.s
        while self.i < len(s):
            c = s[self.i]
            if (brace and c == "}") or (env and s.startswith("\\end", self.i)):
                break
            if c not in "_^ ":
                self.limits = False
            if c == "\\":
                out.append(self.command())
            elif c == "{":
                self.i += 1
                out.append(self.seq(brace=True))
                self.i += 1
            elif c in "_^":
                self.i += 1
                limits = self.limits
                out.append(tex_script(c, self.atom(), explicit=limits))
                self.limits = limits
            elif c == "&":
                self.i += 1
                out.append(COL)
            elif c == "~":
                self.i += 1
                out.append(NB)
            else:
                self.i += 1
                out.append(c)
        return "".join(out)

    def atom(self):
        """One argument: a {group}, a single command, or a single character."""
        while self.peek() == " ":
            self.i += 1
        c = self.peek()
        if c in ("", "}"):
            return ""
        if c == "{":
            self.i += 1
            r = self.seq(brace=True)
            self.i += 1
            return r
        if c == "\\":
            return self.command()
        self.i += 1
        return c

    def command(self):
        s = self.s
        self.i += 1
        if self.i >= len(s):
            return ""
        if not s[self.i].isalpha():
            ch = s[self.i]
            self.i += 1
            if ch == "\\":  # row break, optionally followed by a spacing like [4pt]
                m = re.match(r"\[\s*-?[\d.]+\s*[a-z]{2}\s*\]", s[self.i :])
                if m:
                    self.i += m.end()
                return ROW
            return TEX_SPACING.get(ch, "‖" if ch == "|" else ch)  # \_ \{ \} \% ... -> the character itself
        m = re.compile(r"[A-Za-z]+").match(s, self.i)
        name = m.group()
        self.i = m.end()

        if name in TEX_SPACING:
            return TEX_SPACING[name]
        if name in TEX_SIZERS:
            if name in ("left", "right") and self.peek() == ".":
                self.i += 1  # invisible delimiter
            return ""
        if name in TEX_NOOP:
            return ""
        if name in ("frac", "tfrac", "dfrac"):
            a, b = self.atom(), self.atom()
            after = " " if re.match(r"[\\(\w]", self.s[self.i :]) else ""  # x/2 (1 + ..) rather than x/2(1 + ..)
            return f"{_wrap(a)}/{_wrap(b)}{after}"
        if name == "sqrt":
            if self.peek() == "[":  # \sqrt[n]{x}: keep it simple, drop the index
                self.i = max(self.i, s.find("]", self.i) + 1)
            return "√" + _wrap(self.atom())
        if name in TEX_ACCENTS:
            a = self.atom()
            return a + TEX_ACCENTS[name] if len(a) == 1 else f"{name}({a})"
        if name == "mathbb":
            return "".join(TEX_BLACKBOARD.get(ch, ch) for ch in self.atom())
        if name in TEX_TEXTLIKE:
            return self.atom()
        if name == "begin":
            return self.environment()
        if name == "end":
            self.atom()
            return ""
        if name in TEX_LIMITS:
            self.limits = True
        if name in TEX_SYMBOLS:
            return TEX_SYMBOLS[name]
        if name in TEX_FUNCS:  # log, max, ...: upright name, separated from a following letter/command
            nxt = re.match(r"\\([A-Za-z]+)|[A-Za-z0-9]", s[self.i :])
            gap = nxt and not (nxt.group(1) in TEX_SIZERS)
            return name + (" " if gap else "")
        TEX_UNKNOWN.add(name)
        return name

    def environment(self):
        name = self.atom().strip()
        if name == "array":
            self.atom()  # column spec
        body = self.seq(env=True)
        if self.s.startswith("\\end", self.i):
            self.i += 4
            self.atom()
        rows = [[c.strip(" ") for c in r.split(COL)] for r in body.split(ROW)]
        while rows and not any(rows[-1]):
            rows.pop()
        if not rows:
            return ""
        base = name.rstrip("*")
        w = [max(len(r[k]) for r in rows if k < len(r)) for k in range(max(map(len, rows)))]

        def line(r, sep):
            return sep.join(c.ljust(w[k], NB) for k, c in enumerate(r)).rstrip(NB)

        if base in TEX_BRACKETS:
            lb, rb = TEX_BRACKETS[base]
            lines = [NB * 2 + lb + NB + line(r, NB * 2) + NB + rb for r in rows]
        elif base == "cases":
            lines = [NB * 2 + "{" + NB + line(r, NB * 2) for r in rows]
        elif len(rows) == 1:
            return " ".join(rows[0])
        else:  # align, aligned, gather, ...: the first column lines up, the rest follows
            lines = [line(r, " ") if len(r) > 1 else r[0] for r in rows]
        return "\n" + "\n".join(lines) + "\n"


def tex_to_text(src, display=False):
    out = _Tex(src).seq().replace(ROW, "\n").replace(COL, " ")
    lines = []
    for ln in out.split("\n"):
        ln = re.sub(r" {2,}", " ", ln).strip(" ").replace(NB, " ")
        if ln.strip():
            lines.append(ln)
    if display:
        return "\n" + "\n".join("    " + ln for ln in lines) + "\n\n"
    return " ".join(lines)


_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\] *|\$\$(.*?)\$\$ *", re.S)  # display math eats the space after it


def convert_math(text):
    """Replace every \\(..\\), \\[..\\] and $$..$$ span with its plain-text rendering."""

    def sub(m):
        if m.group(1) is not None:
            return tex_to_text(m.group(1))
        return tex_to_text(m.group(2) if m.group(2) is not None else m.group(3), display=True)

    return _MATH.sub(sub, text)


# ---------------------------------------------------------------- HTML -> plain text


class _HtmlToText(HTMLParser):
    BLOCK = {"p", "h1", "h2", "h3", "pre", "ul", "ol", "div", "br", "table", "tr"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.pre = 0  # >0 inside <pre>, where whitespace is kept as written
        self.script = None  # (tag, index into parts) while inside <sub>/<sup>
        self.table = None  # rows of cell texts while inside <table>
        self.cell = None  # index into parts where the current <td>/<th> started
        self.heads = 0  # leading rows that are header rows

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.pre += 1
        if tag in ("sub", "sup"):
            self.script = (tag, len(self.parts))
            return
        if self.table is not None:
            if tag == "tr":
                self.table.append([])
            elif tag in ("td", "th"):
                self.cell = len(self.parts)
                if tag == "th" and not self.table[-1]:
                    self.heads += 1
            return
        if tag == "table":
            self.table, self.heads = [], 0
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("\n  - ")
        if tag in ("h2", "h3"):
            self.parts.append("## ")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre = max(0, self.pre - 1)
        if tag in ("sub", "sup") and self.script:
            _, idx = self.script
            arg = "".join(self.parts[idx:])
            del self.parts[idx:]
            self.parts.append(tex_script("^" if tag == "sup" else "_", arg))
            self.script = None
            return
        if self.table is not None:
            if tag in ("td", "th") and self.cell is not None:
                self.table[-1].append("".join(self.parts[self.cell :]))
                del self.parts[self.cell :]
                self.cell = None
            elif tag == "table":
                self.parts.append(self.render_table())
                self.table = None
            return
        if tag in self.BLOCK:
            self.parts.append("\n")

    def render_table(self):
        rows = [[convert_math(re.sub(r"\s+", " ", c)).strip() for c in r] for r in self.table if r]
        if not rows:
            return ""
        w = [max(len(r[k]) for r in rows if k < len(r)) for k in range(max(map(len, rows)))]
        lines = ["  ".join(c.ljust(w[k]) for k, c in enumerate(r)).rstrip() for r in rows]
        if self.heads:
            lines.insert(self.heads, "  ".join("─" * n for n in w))
        return "\n" + "\n".join(lines) + "\n"

    def handle_data(self, data):
        if not self.pre:  # reflow paragraphs: the HTML source's line breaks and indentation mean nothing
            data = re.sub(r"\s+", " ", data)
            if not self.parts or self.parts[-1][-1:].isspace():
                data = data.lstrip()
        self.parts.append(data)


def _stray_tex(m):
    name = m.group(1)
    return TEX_SPACING.get(name) or TEX_SYMBOLS.get(name) or m.group(0)


def problem_text(ch):
    """Problem statement as plain text (shared by `show` and the TUI)."""
    p = _HtmlToText()
    page = (ch["dir"] / "challenge.html").read_text()
    p.feed(re.sub(r"<svg.*?</svg>", "[diagram: open challenge.html]", page, flags=re.S))
    p.close()
    text = convert_math("".join(p.parts))
    text = re.sub(r"\\([A-Za-z]+)", _stray_tex, text).replace(NB, " ")  # commands left outside math delimiters
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n\s*\n(\s*- )", r"\n\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def cmd_show(args):
    ch = find_challenge(args.id)
    print(f"# {ch['id']}. {ch['slug']} ({ch['diff']})\n")
    print(problem_text(ch))
    print(f"\nfiles:    {ch['dir']}")
    print(f"solution: {solution_dir(ch)}")
    reasons = flag_for(ch)
    if reasons:
        print(f"\nHEAVY perf test on this machine: {', '.join(reasons)}")
        print("Functional tests still run locally; benchmark this one on leetgpu.com.")


def cmd_new(args):
    ch = find_challenge(args.id)
    sol_name, starter_name = LANGS[args.lang]
    dst = solution_dir(ch) / sol_name
    if dst.exists():
        sys.exit(f"{dst} already exists")
    if not (ch["dir"] / "starter" / starter_name).exists():
        have = ", ".join(p.name for p in (ch["dir"] / "starter").iterdir())
        sys.exit(f"this challenge has no {args.lang} starter (available: {have})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ch["dir"] / "starter" / starter_name, dst)
    print(f"created {dst}")
    reasons = flag_for(ch)
    if reasons:
        print(f"note: HEAVY perf test ({', '.join(reasons)}) - benchmark on leetgpu.com")


def cmd_run(args):
    import torch

    ch = find_challenge(args.id)
    src = Path(args.file) if args.file else solution_dir(ch) / LANGS[args.lang][0]
    if not src.exists():
        sys.exit(f"{src} not found. Create it with: uv run leetgpu/lgpu.py new {ch['id']} --lang {args.lang}")
    torch.manual_seed(args.seed)
    c = load_challenge(ch)
    sig = c.get_solve_signature()
    solve = load_solve(src, args.lang, sig, args.t4)

    groups = [("example", as_list(c.generate_example_test()))]
    if not args.example:
        groups.append(("functional", as_list(c.generate_functional_test())))
    passed = True
    for group, cases in groups:
        for i, case in enumerate(cases):
            case = {k: materialize(v, "cuda") for k, v in case.items()}
            errors = run_case(c, sig, case, solve)
            mark = "PASS" if not errors else "FAIL"
            print(f"[{mark}] {group} {i}: {describe_case(case)}")
            for e in errors:
                print(f"       {e}")
            passed &= not errors
            if errors and not args.keep_going:
                break
        if not passed:
            break

    if not args.file and not args.example:  # progress tracks only the full run of solutions/
        status = read_json(STATUS_FILE, {})
        status.setdefault(ch["slug"], {})[args.lang] = "pass" if passed else "fail"
        write_json(STATUS_FILE, status)
    if not passed:
        sys.exit(1)
    if args.perf:
        run_perf(ch, c, sig, solve, args)


def run_perf(ch, c, sig, solve, args):
    import torch

    reasons = flag_for(ch)
    if reasons and not args.force:
        print(f"\nperf test skipped, flagged HEAVY: {', '.join(reasons)}")
        print("Benchmark it on leetgpu.com, or pass --force if you raised WSL memory / rescanned.")
        return
    entry = read_json(SCAN_FILE, {}).get(ch["slug"], {})
    need = entry.get("perf_vram") or entry.get("meta_bytes") or meta_perf_bytes(ch) or 0
    free, _ = torch.cuda.mem_get_info()
    if need > free and not args.force:
        print(f"\nperf test needs ~{need / GB:.1f}G VRAM, only {free / GB:.1f}G free right now. Close other GPU apps.")
        return
    for case in as_list(c.generate_performance_test()):
        case = {k: materialize(v, "cuda") for k, v in case.items()}
        print(f"\nperf: {describe_case(case)}")
        errors = run_case(c, sig, case, solve)
        if errors:
            print("[FAIL] perf case is wrong:\n       " + "\n       ".join(errors))
            sys.exit(1)
        med, best = bench(lambda: solve(case), args.warmup, args.iters)
        ref_med, _ = bench(lambda: c.reference_impl(**case), args.warmup, args.iters)
        print(f"  yours:     {med:9.3f} ms median, {best:9.3f} ms best ({args.iters} runs)")
        print(f"  reference: {ref_med:9.3f} ms median (PyTorch, {ref_med / med:.2f}x yours)")
        print(f"  ({torch.cuda.get_device_name(0)}; leetgpu.com times on a T4, so expect different numbers)")


def main():
    ap = argparse.ArgumentParser(description="Offline LeetGPU runner")
    sub = ap.add_subparsers(dest="cmd")  # no command -> TUI

    sub.add_parser("sync", help="clone or update the challenge set")

    p = sub.add_parser("scan", help="measure perf-test memory/time and flag heavy challenges")
    p.add_argument("ids", nargs="*")
    p.add_argument("--force", action="store_true", help="rescan unchanged challenges")
    p.add_argument("--timeout", type=int, default=300)

    p = sub.add_parser("list", help="list challenges")
    p.add_argument("--difficulty", choices=DIFFICULTIES)
    p.add_argument("--heavy", action="store_true", help="only flagged challenges")
    p.add_argument("--todo", action="store_true", help="only challenges without a solution file")

    p = sub.add_parser("show", help="print a problem statement")
    p.add_argument("id")

    p = sub.add_parser("new", help="copy a starter into solutions/")
    p.add_argument("id")
    p.add_argument("--lang", choices=LANGS, default="cuda")

    p = sub.add_parser("run", help="test a solution")
    p.add_argument("id")
    p.add_argument("--lang", choices=LANGS, default="cuda")
    p.add_argument("--file", help="solution file to test instead of the default one")
    p.add_argument("--example", action="store_true", help="only the example test")
    p.add_argument("--t4", action="store_true", help="CUDA: compile for the site's T4 (sm_75) features")
    p.add_argument("--perf", action="store_true", help="also run the performance benchmark")
    p.add_argument("--force", action="store_true", help="run perf even if flagged heavy")
    p.add_argument("--keep-going", action="store_true", help="don't stop at the first failure")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("_probe")
    p.add_argument("id")

    args = ap.parse_args()
    if args.cmd is None:
        try:
            import tui  # lazy: the CLI itself stays dependency-free
        except ImportError as e:
            ap.print_help()
            sys.exit(f"\nTUI unavailable ({e}). Run `uv sync`, or use a subcommand.")
        return tui.main()
    if args.cmd == "_probe":
        return probe(find_challenge(args.id))
    {"sync": cmd_sync, "scan": cmd_scan, "list": cmd_list, "show": cmd_show, "new": cmd_new, "run": cmd_run}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
