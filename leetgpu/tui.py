"""Textual front end for lgpu. Opens when you run lgpu.py with no arguments.

Reads the same data as the CLI (problems/, solutions/, .cache/) and runs `run` / `new`
as subprocesses of lgpu.py, streaming their output, so compiling and GPU work never block the UI.

Two views, chosen by terminal size:
  wide   (>=120 cols, >=16 rows)  everything at once: list | problem over output
  paged  (anything smaller)       like a website: a list page, and a detail page (problem + output, no list)
                                  opened with enter or a click; esc / backspace / alt+left go back,
                                  alt+right goes forward
A status line under the header always shows whether the selected challenge is solved.
"""

import asyncio
import os
import re
import sys
import time

from rich.style import Style
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
RUN_ACTIONS = {"run", "run_example", "run_perf", "new"}
PREFS_FILE = lgpu.CACHE / "tui.json"  # gitignored with the rest of .cache/

# state of a challenge: label + style for the status line, mark + style for the table
STATE = {
    "solved": ("✔ solved", "bold green", "✔", "green"),
    "failing": ("✘ failing", "bold red", "✘", "red"),
    "untested": ("· not run yet", "yellow", "·", "yellow"),
    "none": ("○ not started", "dim", "", "dim"),
}
RESULT_MARK = {"pass": ("✔", "green"), "fail": ("✘", "red"), None: ("·", "yellow"), "missing": ("–", "dim")}

GRADE_STYLE = {"A": "bold green", "B": "green", "C": "yellow", "D": "dark_orange", "E": "bold red"}

HINT = "r runs the tests · e example only · p benchmark (recorded) · v versions · n creates the solution file · ? all keys"

HELP = """\
Open       enter / click  open a challenge: its problem and the run output
Navigate   esc / backspace / alt+←   back (like a browser)
           alt+→                     forward
           ↑ ↓                       pick a challenge, or scroll the problem
           space / b                 scroll the problem down / up
           tab                       small detail page: switch problem / output
Filter     /            show the filter bar and search by id or name
           enter / ↓    back to the list, keep the filter
           esc          clear the filter and hide the bar
           d  h  o      difficulty · heavy only · todo only
Work       r            run example + functional tests
           e            example only
           p            tests + performance benchmark: recorded as a version and rated
           v            versions: every benchmarked version of your solution with its
                        time, throughput, rating and speedup; enter edits its note
           n            create the solution file from the starter
Options    l            language (cuda, triton, pytorch)
           t            compile for the T4 (sm_75)
Other      c            change the theme (remembered next launch)
           x            cancel the running job
           q            quit          ?  this help
Wide terminals (120+ columns) show the list, problem and output at once.
"""


def line_style(line):
    m = re.match(r"  rating:\s+([A-E]) ", line)
    if m:
        return GRADE_STYLE[m.group(1)]
    if line.startswith(("recorded as", "re-measured")):
        return "bold magenta"
    if line.startswith(("  traffic:", "  work:")):
        return "cyan"
    if line.startswith(("not recorded", "  rating:")):
        return "yellow"
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


def clip(s, n):
    n = max(n, 3)
    return s if len(s) <= n else s[: n - 1] + "…"


def short_name(slug):
    return re.sub(r"^\d+_", "", slug)


def link(label, action):
    """Clickable text in the status line."""
    return label, Style(color="cyan", underline=True, meta={"@click": action})


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "Close")]
    CSS = """
    HelpScreen { align: center middle; }
    #help { width: 90%; max-width: 78; height: auto; max-height: 100%; border: round $accent;
            background: $surface; padding: 0 2; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help") as box:
            box.border_title = "keys"
            box.border_subtitle = "esc to close"
            yield Static(HELP, markup=False)


class NoteScreen(ModalScreen):
    """Small dialog: type what changed in a version."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    NoteScreen { align: center middle; }
    #note-box { width: 90%; max-width: 72; height: auto; border: round $accent; background: $surface; padding: 1 2; }
    """

    def __init__(self, title, value):
        super().__init__()
        self.title_text = title
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="note-box") as box:
            box.border_title = self.title_text
            box.border_subtitle = "enter saves · esc cancels"
            yield Input(value=self.value, placeholder="e.g. shared-memory tiling, float4 loads", id="note-input")

    def on_mount(self):
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def submit(self, event: Input.Submitted):
        self.dismiss(event.value)

    def action_cancel(self):
        self.dismiss(None)


class VersionsTable(DataTable):
    """Reports its own width so the app can pick the columns that fit."""

    def on_resize(self, event: events.Resize):
        self.app.versions_resized(event.size.width)


class JobLog(RichLog):
    """RichLog wraps a line once, when it is written. Keep the lines and re-flow them when the pane is resized
    (the same pane is 40 columns wide beside the problem and 100 when stacked)."""

    def __init__(self, **kwargs):
        # min_width=1: by default RichLog renders every line at least 78 columns wide, so in a narrower pane the text
        # would be clipped instead of wrapped
        super().__init__(wrap=True, markup=False, highlight=False, min_width=1, **kwargs)
        self.entries = []
        self.flowed_width = 0

    def add(self, text):
        self.entries.append(text)
        self.write(text)

    def reset(self):
        self.entries.clear()
        self.clear()

    def on_resize(self, event: events.Resize):
        if event.size.width == self.flowed_width or not self.entries:
            self.flowed_width = event.size.width
            return
        self.flowed_width = event.size.width
        self.clear()
        for text in self.entries:
            self.write(text)


class ChallengeTable(DataTable):
    """Reports its own width (Resize events don't bubble) and opens a challenge on a single click."""

    BINDINGS = [Binding("enter", "select_cursor", "Open")]  # same key as DataTable's, but shown in the footer

    def on_resize(self, event: events.Resize):
        self.app.table_resized(event.size.width)

    def on_click(self, event: events.Click):
        if self.app.view_mode != "paged":
            return  # wide view: a click just highlights; the problem is already on screen
        row = event.style.meta.get("row")
        if row is None or row < 0:
            return  # header
        self.move_cursor(row=row, animate=False)
        self.app.call_after_refresh(self.app.action_activate)


class LgpuApp(App):
    TITLE = "lgpu"

    CSS = """
    #status { height: 1; padding: 0 1; background: $boost; }
    #left { width: 50%; min-width: 60; }
    #right { width: 1fr; }
    #table { height: 1fr; border: round $primary; }
    #info { height: 1fr; }
    #problem-box, #history-box { height: 1fr; border: round $primary; }
    #problem-box { padding: 0 1; }
    #history-box { display: none; }
    #versions { height: 1fr; }
    #versions-empty { height: auto; padding: 1 2; color: $text-muted; }
    #log { height: 40%; min-height: 5; border: round $secondary; padding: 0 1; }
    #left:focus-within #table { border: round $accent; }
    #toosmall { display: none; width: 100%; height: 1fr; content-align: center middle; text-align: center; }

    /* paged view: one page at a time */
    #main.paged { layout: vertical; }
    #main.paged #left, #main.paged #right { width: 100%; height: 1fr; min-width: 0; }
    #main.paged.page-list #right { display: none; }
    #main.paged.page-detail #left { display: none; }
    #main.paged #filter { height: 1; border: none; padding: 0 1; }

    /* detail page arrangements: problem and output side by side, stacked, or one at a time */
    #main.paged.detail-side #right { layout: horizontal; }
    #main.paged.detail-side #info { width: 3fr; height: 100%; }
    #main.paged.detail-side #log { width: 2fr; height: 100%; min-height: 0; }
    #main.paged.detail-stack #info { height: 1fr; }
    #main.paged.detail-stack #log { height: 40%; min-height: 5; }
    #main.paged.detail-single #info, #main.paged.detail-single #log { height: 100%; }
    #main.paged.detail-single.pane-problem #log { display: none; }
    #main.paged.detail-single.pane-output #info { display: none; }

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
    """

    BINDINGS = [
        # footer order = priority: on a narrow terminal the tail gets clipped, so the essentials come first
        Binding("escape", "back", "Back"),
        Binding("backspace", "back", show=False),
        Binding("alt+left", "back", show=False),
        Binding("alt+right", "forward", "Forward"),
        Binding("tab", "next_pane", "Switch pane", priority=True),
        Binding("shift+tab", "next_pane", show=False, priority=True),
        Binding("r", "run", "Run"),
        Binding("enter", "activate", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("e", "run_example", "Example"),
        Binding("p", "run_perf", "Perf"),
        Binding("v", "toggle_versions", "Versions"),
        Binding("n", "new", "New"),
        Binding("l", "cycle_lang", "Lang"),
        Binding("t", "toggle_t4", "T4"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("d", "cycle_diff", "Diff", show=False),  # less used: listed in ? help, keeps the footer short
        Binding("h", "toggle_heavy", "Heavy", show=False),
        Binding("o", "toggle_todo", "Todo", show=False),
        Binding("c", "change_theme", "Theme", show=False),
        Binding("x", "cancel", "Cancel"),
        Binding("q", "quit", "Quit"),
        Binding("space", "problem_down", show=False),
        Binding("b", "problem_up", show=False),
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
        self.cancelled = False
        self.proc = None
        self.view_mode = "wide"  # "wide" | "paged"
        self.detail = "side"  # paged detail arrangement: "side" | "stack" | "single"
        self.page = "list"  # paged: "list" | "detail"
        self.pane = "problem"  # detail-single: which of problem / output is showing
        self.nav = [("list", None)]  # browser-style history of (page, slug)
        self.nav_pos = 0
        self.table_key = None  # (tier, name width) the table columns were last built for
        self.show_versions = False  # the info pane shows the versions table instead of the problem
        self.versions_key = None  # (tier, note width) the versions columns were last built for
        self.ver_rows = []
        self.perf = {}  # slug -> {lang: row of the fastest recorded version}
        self.challenges = []
        self.by_slug = {}
        self.flags = {}
        self.scanned = set()
        self.info = {}  # slug -> {"langs": {lang: result|None}, "state": ...}; rebuilt by reload()
        self.text_cache = {}
        self.filter_timer = None

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status", markup=False)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Input(placeholder="filter by id or name   ( / )", id="filter")
                yield ChallengeTable(id="table", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                with Vertical(id="info"):  # the problem, or (key v) the versions of your solution
                    with VerticalScroll(id="problem-box"):
                        yield Static(id="problem")
                    with Vertical(id="history-box"):
                        yield VersionsTable(id="versions", cursor_type="row", zebra_stripes=True)
                        yield Static(id="versions-empty", markup=False)
                yield JobLog(id="log")
        yield Static("Too small\n(need 40×10)", id="toosmall")
        yield Footer()

    def on_mount(self):
        table = self.query_one("#table", DataTable)
        table.border_title = "challenges"
        self.query_one("#problem-box").border_title = "problem"
        self.query_one("#filter", Input).display = False  # shown by /
        log = self.query_one("#log", JobLog)
        log.border_title = "output"
        log.add(Text(HINT, style="dim"))

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
        self.apply_table_layout(self.size.width * 45 // 100 - 3)
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
        w, h = event.size.width, event.size.height
        too_small = w < 40 or h < 10
        old = self.view_mode
        self.view_mode = "wide" if (w >= 120 and h >= 16) else "paged"
        self.detail = "side" if w >= 90 else ("stack" if h >= 26 else "single")
        if self.view_mode == "paged" and old == "wide":
            # entering the paged view starts on the list page, unless a job is running (then its output matters)
            ch = self.current()
            self.nav, self.nav_pos = [("list", None)], 0
            self.page = "list"
            if self.running and ch:
                self.nav.append(("detail", ch["slug"]))
                self.nav_pos, self.page = 1, "detail"
        self.query_one("#main").display = not too_small
        self.query_one("#status").display = not too_small
        self.query_one("#toosmall").display = too_small
        self.query_one(Header).display = h >= 24 and not too_small
        footer = self.query_one(Footer)
        footer.display = not too_small
        footer.compact = w < 100
        footer.show_command_palette = w >= 100  # on narrow terminals the keys matter more
        self.apply_view(focus=self.view_mode != old)

    def on_unmount(self):
        if self.proc and self.proc.returncode is None:
            self.proc.kill()

    def check_action(self, action, parameters):
        paged = self.view_mode == "paged"
        if action == "cancel":
            return self.running  # only shown while a job runs
        if action in RUN_ACTIONS and self.running:
            return None  # shown greyed out
        if action == "back":
            if self.query_one("#filter", Input).display:
                return True
            if paged:
                return self.nav_pos > 0
            return not isinstance(self.focused, DataTable)  # wide view: esc returns from the problem pane to the list
        if action == "forward":
            return paged and self.nav_pos < len(self.nav) - 1
        if action == "next_pane":
            return paged and self.page == "detail" and self.detail == "single"  # elsewhere tab keeps its focus job
        return True

    # ------------------------------------------------------------ pages (browser-style navigation)

    def apply_view(self, focus=True):
        """Set the CSS classes for the current page/arrangement, then focus the right widget."""
        main = self.query_one("#main")
        paged = self.view_mode == "paged"
        main.set_class(paged, "paged")
        for p in ("list", "detail"):
            main.set_class(paged and self.page == p, f"page-{p}")
        for d in ("side", "stack", "single"):
            main.set_class(self.detail == d, f"detail-{d}")
        for p in ("problem", "output"):
            main.set_class(self.pane == p, f"pane-{p}")
        self.query_one("#problem-box").display = not self.show_versions
        self.query_one("#history-box").display = self.show_versions
        if self.show_versions:
            self.refresh_versions()
        self.refresh_bindings()
        self.update_status()
        if focus:
            self.call_after_refresh(self.focus_view)

    def info_target(self):
        return "#versions" if self.show_versions else "#problem-box"

    def focus_view(self):
        if self.view_mode == "wide" or self.page == "list":
            target = "#table"
        elif self.detail == "single" and self.pane == "output":
            target = "#log"
        else:
            target = self.info_target()
        self.query_one(target).focus()

    def go(self, page, slug=None):
        """Navigate to a new page: drops any 'forward' history, like clicking a link."""
        entry = (page, slug)
        if self.nav[self.nav_pos] != entry:
            del self.nav[self.nav_pos + 1 :]
            self.nav.append(entry)
            self.nav_pos += 1
        self.show_entry(entry)

    def show_entry(self, entry):
        page, slug = entry
        self.page = page
        if page == "detail" and slug:
            self.select_slug(slug)
        self.apply_view()

    def select_slug(self, slug):
        table = self.query_one("#table", DataTable)
        slugs = [c["slug"] for c in self.visible()]
        if slug not in slugs:  # the filter hides it: drop the filter
            box = self.query_one("#filter", Input)
            box.value = ""
            box.display = False
            self.refresh_table()
            slugs = [c["slug"] for c in self.visible()]
        if slug in slugs:
            table.move_cursor(row=slugs.index(slug), animate=False)

    def ensure_detail(self):
        """Running something from the list page opens the detail page, where the output is."""
        if self.view_mode != "paged":
            return
        ch = self.current()
        if self.page != "detail" and ch:
            self.go("detail", ch["slug"])
        if self.detail == "single":
            self.pane = "output"
            self.apply_view()

    def action_activate(self):
        """enter / click: open the challenge. In the wide view it is already open, so move into the problem."""
        ch = self.current()
        if not ch:
            return
        if self.view_mode == "paged":
            if self.page == "list":
                self.show_versions = False  # a freshly opened page starts on the problem
                self.go("detail", ch["slug"])
            else:
                self.action_run()
        else:
            self.query_one(self.info_target()).focus()

    def action_back(self):
        box = self.query_one("#filter", Input)
        if box.display:  # first close the filter bar: clear it and hide it
            if box.value:
                box.value = ""
                self.refresh_table()
            box.display = False
            self.refresh_bindings()
            self.query_one("#table", DataTable).focus()
        elif self.view_mode == "paged" and self.nav_pos > 0:
            self.nav_pos -= 1
            self.show_entry(self.nav[self.nav_pos])
        else:
            self.query_one("#table", DataTable).focus()

    def action_forward(self):
        if self.view_mode == "paged" and self.nav_pos < len(self.nav) - 1:
            self.nav_pos += 1
            self.show_entry(self.nav[self.nav_pos])

    def action_next_pane(self):
        self.pane = "output" if self.pane == "problem" else "problem"
        self.apply_view()

    # ------------------------------------------------------------ status line + table

    def update_subtitle(self):
        bits = [f"lang {self.lang}", f"T4 {'on' if self.t4 else 'off'}", f"diff {self.diff or 'all'}"]
        if self.heavy_only:
            bits.append("heavy only")
        if self.todo_only:
            bits.append("todo only")
        self.sub_title = "  ·  ".join(bits)

    def update_status(self):
        """One line, always visible: most important first, so truncation only cuts the tail."""
        if not self.is_mounted:
            return
        bar = self.query_one("#status", Static)
        ch = self.current()
        t = Text(no_wrap=True, overflow="ellipsis")
        if self.view_mode == "paged" and self.page == "detail":
            t.append(*link("‹ back", "app.back"))
            t.append("  ", style="dim")
        if not ch:
            t.append("no challenge selected", style="dim")
            bar.update(t)
            return
        slug = ch["slug"]
        inf = self.info[slug]
        label, style, _, _ = STATE[inf["state"]]
        t.append(label, style=style)
        t.append("  │  ", style="dim")
        t.append(f"{ch['id']}. {short_name(slug)}", style="bold")
        t.append(f"  {ch['diff']}", style=DIFF_STYLE[ch["diff"]])
        t.append("  │  ", style="dim")
        for lang in lgpu.LANGS:
            mark, mstyle = RESULT_MARK[inf["langs"].get(lang, "missing")]
            if lang == self.lang:
                mstyle += " bold underline"
            t.append(f"{LANG_TAG[lang]}{mark}", style=mstyle)
            t.append(" ")
        best = self.perf.get(slug, {}).get(self.lang)
        if best:  # the fastest recorded version of the current language
            if best["grade"]:
                t.append(f"best {best['grade']} {best['eff'] * 100:.0f}%", style=GRADE_STYLE[best["grade"]])
            else:
                t.append(f"best {best['ms']:.2f} ms", style="dim")
            t.append(f" v{best['n']}", style="dim")
        if self.t4:
            t.append(" T4", style="bold magenta")
        if self.flags[slug]:
            t.append("  ⚠ heavy perf test", style="yellow")
        bar.update(t)

    def table_resized(self, width):
        self.apply_table_layout(width - 3)  # borders + scrollbar

    def apply_table_layout(self, tw):
        """Pick the columns that fit: full -> drop diff/progress detail -> name only. Names are ellipsised to fit."""
        if tw >= 56:
            tier, name_w = "full", tw - 39
        elif tw >= 34:
            tier, name_w = "mid", tw - 12
        else:
            tier, name_w = "tiny", tw - 6
        key = (tier, max(name_w, 8))
        if key == self.table_key:
            return
        self.table_key = key
        table = self.query_one("#table", DataTable)
        keep = self.current()
        table.clear(columns=True)
        if tier == "full":
            table.add_column("id", width=4)
            table.add_column("diff", width=6)
            table.add_column("challenge", width=key[1])
            table.add_column("progress", width=11)
            table.add_column("perf", width=5)
            table.add_column("", width=1)
        elif tier == "mid":
            table.add_column("id", width=4)
            table.add_column("challenge", width=key[1])
            table.add_column("", width=2)
        else:
            table.add_column("challenge", width=key[1])
            table.add_column("", width=2)
        self.refresh_table(keep)

    def reload(self):
        """Re-read progress + solution files, then redraw. Called at start and after every job."""
        status = lgpu.read_json(lgpu.STATUS_FILE, {})
        for ch in self.challenges:
            slug = ch["slug"]
            langs = {}
            for lang, (fname, _) in lgpu.LANGS.items():
                if (lgpu.solution_dir(ch) / fname).exists():
                    langs[lang] = status.get(slug, {}).get(lang)  # "pass" | "fail" | None (never run)
            results = list(langs.values())
            if "pass" in results:
                state = "solved"
            elif "fail" in results:
                state = "failing"
            else:
                state = "untested" if langs else "none"
            self.info[slug] = {"langs": langs, "state": state}
            self.perf[slug] = {}
            for lang in langs:
                if lgpu.history_path(slug, lang).exists():
                    rows = lgpu.history_rows(slug, lang)
                    if rows:
                        self.perf[slug][lang] = min(rows, key=lambda r: r["ms"])
        self.refresh_table()
        if self.show_versions:
            self.refresh_versions()

    def row_cells(self, ch):
        slug = ch["slug"]
        inf = self.info[slug]
        tier, name_w = self.table_key
        _, _, mark, mstyle = STATE[inf["state"]]
        if self.flags[slug]:
            flag = Text("⚠", style="bold yellow")
        elif slug not in self.scanned:
            flag = Text("?", style="dim")
        else:
            flag = Text("")
        if tier == "full":
            prog = Text()
            for lang, result in inf["langs"].items():
                m, s = RESULT_MARK[result]
                prog.append(f"{LANG_TAG[lang]}{m} ", style=s)
            best = self.perf.get(slug, {}).get(self.lang)
            if best and best["grade"]:
                perf = Text(f"{best['grade']} {best['eff'] * 100:.0f}%", style=GRADE_STYLE[best["grade"]])
            else:
                perf = Text("")
            return (str(ch["id"]), Text(ch["diff"], style=DIFF_STYLE[ch["diff"]]), clip(short_name(slug), name_w), prog, perf, flag)
        state = Text(mark or " ", style=mstyle) + flag
        if tier == "mid":
            return (str(ch["id"]), clip(short_name(slug), name_w), state)
        return (clip(f"{ch['id']} {short_name(slug)}", name_w), state)

    def visible(self):
        q = self.query_one("#filter", Input).value.strip().lower()
        for ch in self.challenges:
            slug = ch["slug"]
            if self.diff and ch["diff"] != self.diff:
                continue
            if self.heavy_only and not self.flags[slug]:
                continue
            if self.todo_only and self.info[slug]["langs"]:
                continue
            if q and q not in slug.lower() and q != str(ch["id"]):
                continue
            yield ch

    def refresh_table(self, keep=None):
        if not self.table_key or not self.info:
            return  # columns or per-challenge info not ready yet (a resize can arrive before reload())
        table = self.query_one("#table", DataTable)
        keep = keep or self.current()
        chs = list(self.visible())
        slugs = [c["slug"] for c in chs]
        with self.batch_update():
            table.clear()
            for ch in chs:
                table.add_row(*self.row_cells(ch), key=ch["slug"])
            if keep and keep["slug"] in slugs:
                table.move_cursor(row=slugs.index(keep["slug"]), animate=False)
        table.border_subtitle = f"{len(slugs)} / {len(self.challenges)}"
        if not slugs:
            self.query_one("#problem", Static).update(Text("no challenges match the filter", style="dim"))
        self.update_status()

    def current(self):
        table = self.query_one("#table", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return self.by_slug.get(key)

    @on(DataTable.RowHighlighted, "#table")
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
        box.border_title = f"{ch['id']}. {short_name(slug)}  ·  {ch['diff']}"
        self.query_one("#problem", Static).update(self.text_cache[slug])
        box.scroll_home(animate=False)
        if self.show_versions:
            self.refresh_versions()
        self.update_status()

    @on(Input.Changed, "#filter")
    def filter_changed(self):
        if self.filter_timer:
            self.filter_timer.stop()
        self.filter_timer = self.set_timer(0.08, self.refresh_table)  # debounce fast typing

    @on(Input.Submitted, "#filter")
    def filter_submitted(self):
        if not self.query_one("#filter", Input).value:
            self.action_back()  # nothing typed: don't leave an empty bar open
        else:
            self.query_one("#table", DataTable).focus()  # keep the bar so the active filter stays visible

    @on(DataTable.RowSelected, "#table")
    def row_selected(self):
        self.action_activate()

    # ------------------------------------------------------------ toggles

    def action_toggle_t4(self):
        self.t4 = not self.t4
        self.update_subtitle()
        self.update_status()
        self.notify(f"T4 compile {'on (sm_75 only)' if self.t4 else 'off (native arch)'}", timeout=2)

    def action_cycle_lang(self):
        langs = list(lgpu.LANGS)
        self.lang = langs[(langs.index(self.lang) + 1) % len(langs)]
        self.update_subtitle()
        self.refresh_table()  # the perf column follows the language
        if self.show_versions:
            self.refresh_versions()
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
        if self.view_mode == "paged" and self.page != "list":
            self.go("list")  # the filter belongs to the list page
        box = self.query_one("#filter", Input)
        box.display = True
        self.refresh_bindings()
        self.call_after_refresh(box.focus)

    def action_focus_table(self):
        self.query_one("#table", DataTable).focus()

    def action_problem_down(self):
        if not self.show_versions:
            self.query_one("#problem-box", VerticalScroll).scroll_page_down()

    def action_problem_up(self):
        if not self.show_versions:
            self.query_one("#problem-box", VerticalScroll).scroll_page_up()

    # ------------------------------------------------------------ versions (benchmark history)

    def action_toggle_versions(self):
        ch = self.current()
        if not ch:
            return
        if self.view_mode == "paged" and self.page == "list":
            self.show_versions = True  # from the list: open the detail page straight on the versions
            self.go("detail", ch["slug"])
            return
        self.show_versions = not self.show_versions
        self.apply_view()

    def versions_resized(self, width):
        tw = width - 1  # scrollbar
        if tw >= 76:
            key = ("full", tw - 68)
        elif tw >= 46:
            key = ("mid", tw - 39)
        else:
            key = ("tiny", 0)
        if key == self.versions_key:
            return
        self.versions_key = key
        table = self.query_one("#versions", DataTable)
        table.clear(columns=True)
        table.add_column("version", width=7)
        table.add_column("ms", width=8)
        if key[0] == "full":
            table.add_column("throughput", width=12)
        table.add_column("roofline", width=8)
        if key[0] in ("full", "mid"):
            table.add_column("vs prev", width=7)
        if key[0] == "full":
            table.add_column("vs torch", width=8)
            table.add_column("runs", width=4)
        if key[0] in ("full", "mid"):
            table.add_column("note", width=max(key[1], 5))
        self.refresh_versions()

    @staticmethod
    def speed_cell(x):
        if x is None:
            return Text("-", style="dim")
        if abs(x - 1) < 0.03:
            return Text("~same", style="dim")
        return Text(f"{x:.2f}x", style="green" if x > 1 else "red")

    def refresh_versions(self, keep=None):
        """Fill the versions table for the selected challenge and the current language."""
        if not self.versions_key or not self.is_mounted:
            return
        ch = self.current()
        table = self.query_one("#versions", DataTable)
        empty = self.query_one("#versions-empty", Static)
        box = self.query_one("#history-box")
        table.clear()
        self.ver_rows = []
        if not ch:
            table.display, empty.display = False, True
            empty.update("no challenge selected")
            return
        slug = ch["slug"]
        rows = lgpu.history_rows(slug, self.lang)
        self.ver_rows = rows
        box.border_title = f"versions · {ch['id']}. {short_name(slug)} · {self.lang}"
        if not rows:
            table.display, empty.display = False, True
            box.border_subtitle = ""
            if (lgpu.solution_dir(ch) / lgpu.LANGS[self.lang][0]).exists():
                empty.update(
                    f"No benchmark recorded for {self.lang} yet.\n\nPress p to run the tests and the performance benchmark.\n"
                    "Every measured version is saved here, so you can compare them."
                )
            else:
                empty.update(f"No {self.lang} solution yet. Press n to create it from the starter.")
            return
        table.display, empty.display = True, False
        cur = lgpu.current_hash(ch, self.lang)
        tier, note_w = self.versions_key
        for r in rows:
            ver = Text(f"v{r['n']}", style="bold")
            if r["hash"] == cur:
                ver.append(" ●", style="cyan")
            if r["best"] and len(rows) > 1:
                ver.append(" ★", style="yellow")
            roof = Text(lgpu.fmt_roofline(r), style=GRADE_STYLE.get(r["grade"], "dim"))
            cells = [ver, f"{r['ms']:.3f}"]
            if tier == "full":
                cells.append(lgpu.fmt_throughput(r))
            cells.append(roof)
            if tier in ("full", "mid"):
                cells.append(self.speed_cell(r["vs_prev"]))
            if tier == "full":
                cells += [self.speed_cell(r["vs_ref"]), str(r["runs"])]
            if tier in ("full", "mid"):
                cells.append(clip(r["note"], max(note_w, 5)) if r["note"] else Text("-", style="dim"))
            table.add_row(*cells, key=str(r["n"]))
        target = keep if keep is not None else rows[-1]["n"]
        table.move_cursor(row=next((i for i, r in enumerate(rows) if r["n"] == target), len(rows) - 1), animate=False)
        c = rows[-1]["ceilings"]
        box.border_subtitle = f"● your file  ★ fastest  ·  ceilings {c.get('copy_gbps', 0):.0f} GB/s, {c.get('fp32_gflops', 0) / 1000:.1f} TFLOPS"

    @on(DataTable.RowSelected, "#versions")
    def edit_note(self, event: DataTable.RowSelected):
        ch = self.current()
        if not ch or event.row_key.value is None:
            return
        n = int(event.row_key.value)
        row = next((r for r in self.ver_rows if r["n"] == n), None)
        if not row:
            return

        def saved(text):
            if text is not None:
                lgpu.set_note(ch["slug"], self.lang, n, text)
                self.refresh_versions(keep=n)

        self.push_screen(NoteScreen(f"what changed in v{n}?", row["note"]), saved)

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

    def launch(self, args):
        self.ensure_detail()
        self.run_cli(args)

    def action_run(self):
        args = self.run_args()
        if args:
            self.launch(args)

    def action_run_example(self):
        args = self.run_args("--example")
        if args:
            self.launch(args)

    def action_run_perf(self):
        args = self.run_args("--perf")
        if args:
            self.launch(args)

    def action_new(self):
        ch = self.current()
        if ch and not self.running:
            self.launch(["new", str(ch["id"]), "--lang", self.lang])

    def action_cancel(self):
        if self.running and self.proc and self.proc.returncode is None:
            self.cancelled = True
            self.proc.terminate()

    @work
    async def run_cli(self, args):
        if self.running:
            self.notify("a job is already running (x to cancel)", severity="warning")
            return
        log = self.query_one("#log", JobLog)
        self.running, self.cancelled = True, False
        self.refresh_bindings()
        log.reset()
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
                log.add(Text(line, style=line_style(line)))
            rc = await self.proc.wait()
        finally:
            timer.stop()
            log.border_subtitle = ""
            self.running = False
            self.refresh_bindings()
        elapsed = time.perf_counter() - t0
        if self.cancelled:
            log.add(Text(f"\n── cancelled after {elapsed:.1f}s ──", style="bold yellow"))
        elif rc == 0:
            log.add(Text(f"\n── done in {elapsed:.1f}s ──", style="bold green"))
        else:
            log.add(Text(f"\n── exit {rc} after {elapsed:.1f}s ──", style="bold red"))
        self.reload()


def main():
    LgpuApp().run()
