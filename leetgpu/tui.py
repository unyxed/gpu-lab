"""Textual front end for lgpu. Opens when you run lgpu.py with no arguments.

Reads the same data as the CLI (problems/, solutions/, .cache/) and runs `run` / `new`
as subprocesses of lgpu.py, streaming their output, so compiling and GPU work never block the UI.
"""

import asyncio
import os
import re
import sys
import time

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

import lgpu

DIFF_STYLE = {"easy": "green", "medium": "yellow", "hard": "red"}
LANG_TAG = {"cuda": "cu", "triton": "tr", "pytorch": "pt"}
DIFF_CYCLE = [None, *lgpu.DIFFICULTIES]
NARROW_BELOW = 120  # columns; under this the panes stack vertically
RUN_ACTIONS = {"run", "run_example", "run_perf", "new"}
PREFS_FILE = lgpu.CACHE / "tui.json"  # gitignored with the rest of .cache/

HELP = """\
Move       ↑ ↓          pick a challenge
           space / b    scroll the problem down / up
Filter     /            show the filter bar and search by id or name
           enter / ↓    back to the list, keep the filter
           esc          clear the filter and hide the bar
           d  h  o      difficulty · heavy only · todo only
Work       n            create the solution file from the starter
           enter        run example + functional tests
           e            example only
           p            tests + performance benchmark
Options    l            language (cuda, triton, pytorch)
           t            compile for the T4 (sm_75)
Other      c            change the theme (remembered next launch)
           x            cancel the running job
           q            quit          ?  this help
"""


def line_style(line):
    if line.startswith("[PASS]"):
        return "bold green"
    if line.startswith("[FAIL]") or line.startswith("Traceback") or "error" in line.lower():
        return "bold red"
    if line.startswith("$"):
        return "dim"
    if line.startswith("       "):
        return "red"
    if line.startswith("perf:"):
        return "bold"
    if line.startswith("  yours") or line.startswith("  reference"):
        return "cyan"
    return ""


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "Close")]
    CSS = """
    HelpScreen { align: center middle; }
    #help { width: 72; height: auto; border: round $accent; background: $surface; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        box = Static(HELP, id="help", markup=False)
        box.border_title = "keys"
        box.border_subtitle = "esc to close"
        yield box


class LgpuApp(App):
    TITLE = "lgpu"

    CSS = """
    #left { width: 45%; min-width: 56; }
    #right { width: 1fr; }
    #table { height: 1fr; border: round $primary; }
    #problem-box { height: 1fr; border: round $primary; padding: 0 1; }
    #log { height: 45%; border: round $secondary; padding: 0 1; }
    #left:focus-within #table { border: round $accent; }
    DataTable, VerticalScroll, RichLog {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-color: $primary 35%;
        scrollbar-color-hover: $primary 70%;
        scrollbar-color-active: $primary;
    }
    #main.narrow { layout: vertical; }
    #main.narrow #left { width: 100%; height: 45%; min-width: 0; }
    #main.narrow #right { width: 100%; height: 1fr; }
    """

    BINDINGS = [
        Binding("enter", "run", "Run"),
        Binding("e", "run_example", "Example"),
        Binding("p", "run_perf", "Perf"),
        Binding("n", "new", "New"),
        Binding("l", "cycle_lang", "Lang"),
        Binding("t", "toggle_t4", "T4"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("d", "cycle_diff", "Diff"),
        Binding("h", "toggle_heavy", "Heavy"),
        Binding("o", "toggle_todo", "Todo"),
        Binding("x", "cancel", "Cancel"),
        Binding("c", "change_theme", "Theme"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
        Binding("space", "problem_down", show=False),
        Binding("b", "problem_up", show=False),
        Binding("escape", "close_filter", show=False),
        Binding("down", "focus_table", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.lang = "cuda"
        self.t4 = False
        self.diff = None
        self.heavy_only = False
        self.todo_only = False
        self.running = False
        self.proc = None
        self.challenges = []
        self.by_slug = {}
        self.flags = {}
        self.scanned = set()
        self.cells = {}  # slug -> table row cells; rebuilt only when status/solutions change
        self.has_sol = {}
        self.text_cache = {}
        self.filter_timer = None

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Input(placeholder="filter by id or name   ( / )", id="filter")
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                with VerticalScroll(id="problem-box"):
                    yield Static(id="problem")
                yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self):
        table = self.query_one("#table", DataTable)
        table.border_title = "challenges"
        table.add_column("id", width=4)
        table.add_column("diff", width=6)
        table.add_column("challenge")
        table.add_column("progress", width=14)
        table.add_column("", width=2)
        self.query_one("#problem-box").border_title = "problem"
        self.query_one("#filter", Input).display = False  # shown by /
        log =self.query_one("#log", RichLog)
        log.border_title = "output"
        log.write(Text("enter runs the tests · n creates the solution file · ? shows all keys", style="dim"))

        if not lgpu.CHALLENGES.exists():
            self.query_one("#problem", Static).update("No problems found. Run:  lgpu sync")
        else:
            self.challenges = lgpu.all_challenges()
            self.by_slug = {c["slug"]: c for c in self.challenges}
            scan = lgpu.read_json(lgpu.SCAN_FILE, {})
            vram, ram = lgpu.machine()
            self.scanned = set(scan)
            self.flags = {c["slug"]: lgpu.assess(scan.get(c["slug"]), vram, ram) for c in self.challenges}
        self.update_subtitle()
        self.reload()
        table.focus()

        saved = lgpu.read_json(PREFS_FILE, {}).get("theme")
        if saved in self.available_themes:  # ignore a theme name from a different Textual version
            self.theme = saved
        self.theme_changed_signal.subscribe(self, self.save_theme)  # subscribe after restoring, so it isn't re-saved

    def save_theme(self, theme):
        prefs = lgpu.read_json(PREFS_FILE, {})
        prefs["theme"] = theme.name
        lgpu.write_json(PREFS_FILE, prefs)

    def on_resize(self, event: events.Resize):
        self.query_one("#main").set_class(event.size.width < NARROW_BELOW, "narrow")

    def on_unmount(self):
        if self.proc and self.proc.returncode is None:
            self.proc.kill()

    def check_action(self, action, parameters):
        if action == "cancel":
            return self.running  # only shown while a job runs
        if action in RUN_ACTIONS and self.running:
            return None  # shown greyed out
        return True

    # ------------------------------------------------------------ table

    def update_subtitle(self):
        bits = [f"lang {self.lang}", f"T4 {'on' if self.t4 else 'off'}", f"diff {self.diff or 'all'}"]
        if self.heavy_only:
            bits.append("heavy only")
        if self.todo_only:
            bits.append("todo only")
        self.sub_title = "  ·  ".join(bits)

    def reload(self):
        """Re-read progress + solution files, then redraw. Called at start and after every job."""
        status = lgpu.read_json(lgpu.STATUS_FILE, {})
        for ch in self.challenges:
            slug = ch["slug"]
            prog = Text()
            self.has_sol[slug] = False
            for lang, (fname, _) in lgpu.LANGS.items():
                if not (lgpu.solution_dir(ch) / fname).exists():
                    continue
                self.has_sol[slug] = True
                result = status.get(slug, {}).get(lang)
                mark, style = ("✔", "green") if result == "pass" else ("✘", "red") if result == "fail" else ("·", "dim")
                prog.append(f"{LANG_TAG[lang]}{mark} ", style=style)
            if self.flags[slug]:
                flag = Text("⚠", style="bold yellow")
            elif slug not in self.scanned:
                flag = Text("?", style="dim")
            else:
                flag = Text("")
            self.cells[slug] = (
                str(ch["id"]),
                Text(ch["diff"], style=DIFF_STYLE[ch["diff"]]),
                re.sub(r"^\d+_", "", slug),
                prog,
                flag,
            )
        self.refresh_table()

    def visible(self):
        q = self.query_one("#filter", Input).value.strip().lower()
        for ch in self.challenges:
            slug = ch["slug"]
            if self.diff and ch["diff"] != self.diff:
                continue
            if self.heavy_only and not self.flags[slug]:
                continue
            if self.todo_only and self.has_sol[slug]:
                continue
            if q and q not in slug.lower() and q != str(ch["id"]):
                continue
            yield slug

    def refresh_table(self):
        table = self.query_one("#table", DataTable)
        keep = self.current()
        slugs = list(self.visible())
        with self.batch_update():
            table.clear()
            for slug in slugs:
                table.add_row(*self.cells[slug], key=slug)
            if keep and keep["slug"] in slugs:
                table.move_cursor(row=slugs.index(keep["slug"]), animate=False)
        table.border_subtitle = f"{len(slugs)} / {len(self.challenges)}"
        if not slugs:
            self.query_one("#problem", Static).update(Text("no challenges match the filter", style="dim"))

    def current(self):
        table = self.query_one("#table", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return self.by_slug.get(key)

    @on(DataTable.RowHighlighted)
    def show_problem(self, event: DataTable.RowHighlighted):
        ch = self.by_slug.get(event.row_key.value)
        if not ch:
            return
        slug = ch["slug"]
        if slug not in self.text_cache:
            out = Text()
            for line in lgpu.problem_text(ch).splitlines():
                out.append(line + "\n", style="bold cyan" if line.startswith("## ") else "")
            reasons = self.flags[slug]
            if reasons:
                out.append(f"\nHEAVY perf test on this machine: {', '.join(reasons)}\n", style="yellow")
                out.append("Functional tests run locally; benchmark on leetgpu.com.\n", style="yellow")
            self.text_cache[slug] = out
        box = self.query_one("#problem-box", VerticalScroll)
        box.border_title = f"{ch['id']}. {re.sub(r'^\d+_', '', slug)}  ·  {ch['diff']}"
        self.query_one("#problem", Static).update(self.text_cache[slug])
        box.scroll_home(animate=False)

    @on(Input.Changed, "#filter")
    def filter_changed(self):
        if self.filter_timer:
            self.filter_timer.stop()
        self.filter_timer = self.set_timer(0.08, self.refresh_table)  # debounce fast typing

    @on(Input.Submitted, "#filter")
    def filter_submitted(self):
        if not self.query_one("#filter", Input).value:
            self.action_close_filter()  # nothing typed: don't leave an empty bar open
        else:
            self.query_one("#table", DataTable).focus()  # keep the bar so the active filter stays visible

    @on(DataTable.RowSelected)
    def row_selected(self):
        self.action_run()

    # ------------------------------------------------------------ toggles / navigation

    def action_toggle_t4(self):
        self.t4 = not self.t4
        self.update_subtitle()
        self.notify(f"T4 compile {'on (sm_75 only)' if self.t4 else 'off (native arch)'}", timeout=2)

    def action_cycle_lang(self):
        langs = list(lgpu.LANGS)
        self.lang = langs[(langs.index(self.lang) + 1) % len(langs)]
        self.update_subtitle()
        self.notify(f"language: {self.lang}", timeout=2)

    def action_cycle_diff(self):
        self.diff = DIFF_CYCLE[(DIFF_CYCLE.index(self.diff) + 1) % len(DIFF_CYCLE)]
        self.update_subtitle()
        self.refresh_table()

    def action_toggle_heavy(self):
        self.heavy_only = not self.heavy_only
        self.update_subtitle()
        self.refresh_table()

    def action_toggle_todo(self):
        self.todo_only = not self.todo_only
        self.update_subtitle()
        self.refresh_table()

    def action_focus_filter(self):
        box = self.query_one("#filter", Input)
        box.display = True
        box.focus()

    def action_close_filter(self):
        """Esc: clear the filter, hide the bar, back to the list."""
        box = self.query_one("#filter", Input)
        if box.value:
            box.value = ""
            self.refresh_table()
        box.display = False
        self.query_one("#table", DataTable).focus()

    def action_focus_table(self):
        self.query_one("#table", DataTable).focus()

    def action_problem_down(self):
        self.query_one("#problem-box", VerticalScroll).scroll_page_down()

    def action_problem_up(self):
        self.query_one("#problem-box", VerticalScroll).scroll_page_up()

    def action_help(self):
        self.push_screen(HelpScreen())

    # ------------------------------------------------------------ running

    def run_args(self, *extra):
        ch = self.current()
        if not ch:
            return None
        if not (lgpu.solution_dir(ch) / lgpu.LANGS[self.lang][0]).exists():
            self.notify(f"no {self.lang} solution yet. Press n to create it from the starter.", severity="warning")
            return None
        args = ["run", str(ch["id"]), "--lang", self.lang, *extra]
        if self.t4 and self.lang == "cuda":
            args.append("--t4")
        return args

    def action_run(self):
        args = self.run_args()
        if args:
            self.run_cli(args)

    def action_run_example(self):
        args = self.run_args("--example")
        if args:
            self.run_cli(args)

    def action_run_perf(self):
        args = self.run_args("--perf")
        if args:
            self.run_cli(args)

    def action_new(self):
        ch = self.current()
        if ch and not self.running:
            self.run_cli(["new", str(ch["id"]), "--lang", self.lang])

    def action_cancel(self):
        if self.running and self.proc and self.proc.returncode is None:
            self.cancelled = True
            self.proc.terminate()

    @work
    async def run_cli(self, args):
        if self.running:
            self.notify("a job is already running (x to cancel)", severity="warning")
            return
        log = self.query_one("#log", RichLog)
        self.running, self.cancelled = True, False
        self.refresh_bindings()
        log.clear()
        log.border_title = "lgpu " + " ".join(args)
        t0 = time.perf_counter()

        def tick():
            log.border_subtitle = f"⏳ {time.perf_counter() - t0:.1f}s · x to cancel"

        tick()
        timer = self.set_interval(0.1, tick)
        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable,
                lgpu.__file__,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"},
                limit=2**20,
            )
            async for raw in self.proc.stdout:
                line = raw.decode(errors="replace").rstrip("\n")
                log.write(Text(line, style=line_style(line)))
            rc = await self.proc.wait()
        finally:
            timer.stop()
            log.border_subtitle = ""
            self.running = False
            self.refresh_bindings()
        elapsed = time.perf_counter() - t0
        if self.cancelled:
            log.write(Text(f"\n── cancelled after {elapsed:.1f}s ──", style="bold yellow"))
        elif rc == 0:
            log.write(Text(f"\n── done in {elapsed:.1f}s ──", style="bold green"))
        else:
            log.write(Text(f"\n── exit {rc} after {elapsed:.1f}s ──", style="bold red"))
        self.reload()


def main():
    LgpuApp().run()
