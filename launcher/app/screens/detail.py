from __future__ import annotations
import re
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen, ModalScreen
from textual.widgets import Header, Footer, Static, Button

from ..config import WinConfig


_MARKUP_RE = re.compile(r"\[/?[^\[\]]*\]")


def _strip_markup(text: str) -> str:
    """Remove Textual / Rich markup tags so the result pastes cleanly into
    a GitHub issue or Reddit comment."""
    return _MARKUP_RE.sub("", text)


class ConfirmModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmModal { align: center middle; }
    #box {
        background: #161b22;
        border: solid #f85149;
        padding: 1 3;
        width: 60;
        height: auto;
    }
    #box Button { margin: 1 1 0 0; }
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(self.message)
            with Horizontal():
                yield Button("Confirm", id="ok", variant="error")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id == "ok")


class ResultModal(ModalScreen[None]):
    DEFAULT_CSS = """
    ResultModal { align: center middle; }
    #box {
        background: #161b22;
        border: solid #58a6ff;
        padding: 1 2;
        width: 95%;
        height: 90%;
        layout: vertical;
    }
    #title { height: 1; margin-bottom: 1; }
    #body-scroll {
        height: 1fr;
        background: #0d1117;
        border: solid #30363d;
        padding: 0 1;
    }
    #button-row {
        height: 3;
        margin-top: 1;
        align: left middle;
    }
    #button-row Button {
        margin-right: 1;
        min-width: 16;
        height: 3;
    }
    """

    BINDINGS = [
        ("escape", "dismiss_modal", "Close"),
        ("q", "dismiss_modal", "Close"),
        ("c", "copy_body", "Copy"),
    ]

    def __init__(self, title: str, body: str):
        super().__init__()
        self.title_str = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(f"[b #58a6ff]{self.title_str}[/]", id="title")
            with VerticalScroll(id="body-scroll"):
                yield Static(self.body)
            with Horizontal(id="button-row"):
                yield Button("Copy (C)", id="copy", variant="success")
                yield Button("Close (Esc)", id="close", variant="primary")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "copy":
            self.action_copy_body()
        else:
            self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def action_copy_body(self) -> None:
        plain = _strip_markup(f"{self.title_str}\n\n{self.body}")
        try:
            self.app.copy_to_clipboard(plain)
            self.app.notify("Copied to clipboard", timeout=3)
        except Exception as e:
            self.app.notify(f"Copy failed: {e}", severity="error", timeout=5)


def _kv(rows: list[tuple[str, str]]) -> str:
    out: list[str] = []
    for k, v in rows:
        out.append(f" [#8b949e]{k:<13}[/] [#e6edf3]{v}[/]")
    return "\n".join(out)


class DetailScreen(Screen):
    DEFAULT_CSS = """
    DetailScreen { background: #0d1117; color: #e6edf3; layout: vertical; }
    #cols { height: 1fr; }
    #left, #right {
        background: #161b22;
        border: solid #30363d;
        padding: 0 1;
        width: 1fr;
        margin: 1 1 0 1;
    }
    #right { color: #8b949e; }
    #actions {
        height: 4;
        padding: 0 2;
        margin: 0;
        background: #11161d;
        border-top: solid #30363d;
    }
    #actions Button {
        margin: 0 1 0 0;
        min-width: 14;
        height: 3;
        content-align: center middle;
    }
    #actions Button#back-btn {
        background: #21262d;
    }
    #status-banner {
        padding: 0 2;
        text-style: bold;
        height: 1;
        content-align: left middle;
    }
    #status-banner.idle    { background: #161b22; color: #8b949e; border-left: thick #30363d; }
    #status-banner.running { background: #11202f; color: #3fb950; border-left: thick #3fb950; }
    /* Loading is the state users complain about missing, make it impossible
       to overlook: tall block, bold amber text, thick border on both sides. */
    #status-banner.loading {
        background: #2d220a;
        color: #ffbb47;
        border: thick #d29922;
        height: 7;
        padding: 1 3;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("escape", "back", "Back"),
        ("l", "load", "Load"),
        ("u", "unload", "Unload"),
        ("t", "test", "Test"),
        ("e", "edit", "Edit"),
    ]

    def __init__(self, cfg: WinConfig, bundle, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.bundle = bundle

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        cfg = self.cfg

        # Status banner is always mounted; refresh_state() flips its text +
        # CSS class between idle / loading / running. Mounting it once (vs
        # conditionally yielding) means refresh_state() can update it in
        # place without reflow or remount churn.
        yield Static("", id="status-banner", classes="idle")

        with Horizontal(id="cols"):
            with VerticalScroll(id="left"):
                yield Static(self._meta_text())
            with VerticalScroll(id="right"):
                yield Static(self._right_text())

        with Horizontal(id="actions"):
            yield Button("Load (L)", id="load-btn", variant="success")
            yield Button("Unload (U)", id="unload-btn", variant="error")
            yield Button("Test (T)", id="test-btn", variant="primary")
            yield Button("Edit (E)", id="edit-btn")
            yield Button("Back (Esc)", id="back-btn")
        yield Footer()

    def on_mount(self) -> None:
        # Kick a one-shot readiness probe so an already-loaded snapshot
        # renders as RUNNING immediately rather than briefly flashing
        # LOADING for up to 5s waiting for the throttled gate to fire.
        self.app.probe_ready_now(self.cfg.id)
        self.refresh_state()

    def on_screen_resume(self) -> None:
        self.app.probe_ready_now(self.cfg.id)
        self.refresh_state()

    def _state(self) -> str:
        """Current snapshot state from the app's perspective.

        idle   , no manifest, no optimistic-load flag → Load enabled
        loading, manifest exists but /v1/models hasn't answered yet, OR the
                  user just clicked Load and the manifest hasn't appeared in
                  the next poll → all action buttons disabled
        running, manifest exists AND /v1/models returned 200 → Unload+Test
        """
        cid = self.cfg.id
        app = self.app
        if cid in app.ready_ids:
            return "running"
        if cid in app.running_ids or cid in app.loading_ids:
            return "loading"
        return "idle"

    def refresh_state(self) -> None:
        """Sync the status banner and button-disabled flags to the live state.

        Called from on_mount, on_screen_resume, after Load/Unload clicks,
        and from the app's poll worker every 2s. Idempotent.
        """
        try:
            banner = self.query_one("#status-banner", Static)
            load_btn = self.query_one("#load-btn", Button)
            unload_btn = self.query_one("#unload-btn", Button)
            test_btn = self.query_one("#test-btn", Button)
        except Exception:
            return  # widgets not mounted yet, on_mount will rerun

        cfg = self.cfg
        state = self._state()
        blocked = (cfg.status == "blocked")

        if state == "running":
            banner.update(f"  ● RUNNING, port {cfg.port} (API ready)")
            banner.set_classes("running")
            # Already loaded → Load disabled, Unload+Test enabled.
            load_btn.disabled = True
            unload_btn.disabled = False
            test_btn.disabled = False
        elif state == "loading":
            banner.update(
                f"[b #ffbb47]⏳  LOADING vLLM on port {cfg.port}...[/]\n"
                f"[#d29922]This takes 60-120 seconds, compiling kernels + loading weights.[/]\n"
                f"[#8b949e]The new console window shows live progress. "
                f"Buttons re-enable once /v1/models responds.[/]"
            )
            banner.set_classes("loading")
            # In-flight → no actions allowed (Back still works via Esc).
            load_btn.disabled = True
            unload_btn.disabled = True
            test_btn.disabled = True
        else:
            banner.update("  ○ Not running")
            banner.set_classes("idle")
            load_btn.disabled = blocked
            unload_btn.disabled = True
            test_btn.disabled = True

    def _meta_text(self) -> str:
        cfg = self.cfg
        sd = self.bundle.shared_defaults
        meta_rows = [
            ("ID", cfg.id),
            ("Tagline", cfg.tagline),
            ("Tier / Status", f"{cfg.tier} / {cfg.status}"),
            ("Bat", cfg.bat),
            ("Py", cfg.py),
            ("GPU", cfg.gpu),
            ("TP × PP", f"{cfg.tp} × {cfg.pp}"),
            ("mem-util", f"{cfg.mem_util}"),
            ("ctx (max-model-len)", f"{cfg.ctx:,}"),
            ("port", str(cfg.port)),
            ("MTP n", "," if cfg.mtp_n is None else str(cfg.mtp_n)),
            ("draft-model n", "," if cfg.draft_model_n is None else str(cfg.draft_model_n)),
            ("Power cap", f"{cfg.power_cap_w or sd.get('power_cap_w', '?')} W"),
        ]
        shared_rows = [
            (k, str(sd[k])) for k in [
                "model_path", "served_model_name", "quantization", "kv_cache_dtype",
                "attention_backend", "vllm_version", "max_num_seqs",
                "max_num_batched_tokens", "block_size",
            ] if k in sd
        ]
        return (
            f"[b #58a6ff]Config metadata[/]\n"
            f"{_kv(meta_rows)}\n\n"
            f"[b #58a6ff]Shared defaults[/]\n"
            f"{_kv(shared_rows)}"
        )

    def _right_text(self) -> str:
        cfg = self.cfg
        bench_rows: list[tuple[str, str]] = []
        for k, v in [
            ("decode_tps (24k prompt)", cfg.decode_tps),
            ("decode_tps (short)", cfg.decode_tps_short),
            ("decode_tps (long)", cfg.decode_tps_long),
            ("prefill cold tok/s", cfg.prefill_tps_cold),
        ]:
            if v is not None:
                bench_rows.append((k, str(v)))
        bench = _kv(bench_rows) if bench_rows else "  [#8b949e],[/]"

        sweep_lines: list[str] = []
        for r in self.bundle.mtp_sweep.get("rows", []):
            mark = " [b #ffbb7f]★ peak[/]" if r.get("peak") else ""
            sweep_lines.append(
                f"  [#8b949e]n={r['mtp_n']}[/]   [#e6edf3]ctx={r['ctx']//1000:>4}k[/]   "
                f"[#3fb950]{r['decode_tps']:>5} t/s[/]{mark}"
            )

        notes = (cfg.notes or ",").rstrip()

        return (
            f"[b #58a6ff]Benchmarks[/]\n"
            f"{bench}\n\n"
            f"[b #58a6ff]Notes[/]\n"
            f" [#8b949e]{notes}[/]\n\n"
            f"[b #58a6ff]MTP sweep (speed config baseline)[/]\n"
            + "\n".join(sweep_lines)
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_load(self) -> None:
        # Honor the same gating the buttons enforce, the keyboard shortcut
        # must not bypass the disabled state (e.g. pressing 'l' while the
        # snapshot is RUNNING should be a no-op, not a duplicate launch).
        if self._state() != "idle" or self.cfg.status == "blocked":
            return
        self._do_load()

    def action_unload(self) -> None:
        if self._state() != "running":
            return
        self._do_unload()

    def action_test(self) -> None:
        if self._state() != "running":
            return
        self._do_test()

    def action_edit(self) -> None:
        # Pop back to the dashboard first so the manager's on-dismiss
        # dashboard rebuild lands on a clean stack, otherwise we'd return
        # from the manager into a stale DetailScreen referencing a possibly
        # renamed/deleted cfg.
        self.app.pop_screen()
        self.app.open_snapshot_manager(select_id=self.cfg.id)

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "back-btn":
            self.action_back()
        elif ev.button.id == "load-btn":
            self._do_load()
        elif ev.button.id == "unload-btn":
            self._do_unload()
        elif ev.button.id == "test-btn":
            self._do_test()
        elif ev.button.id == "edit-btn":
            self.action_edit()

    def _do_load(self) -> None:
        cfg = self.cfg
        # Port-collision pre-check: if another running snapshot already owns
        # this port, the new wrapper would silently exit on bind, leaving
        # the dashboard correctly lit on the *other* config. The user reads
        # that as "wrong card lit" (issue #2 follow-up). Refuse the launch
        # with a clear message instead.
        conflict = next(
            (rp for cid, rp in self.app.running.items()
             if cid != cfg.id and rp.port == cfg.port),
            None,
        )
        if conflict is not None:
            self.app.notify(
                f"Port {cfg.port} is in use by '{conflict.matched_id or conflict.pid}'. "
                f"Unload it first, or change {cfg.id}'s port via Edit Snapshots.",
                severity="error", timeout=8,
            )
            return
        def _yes(ok: bool):
            if ok:
                from .. import runtime
                try:
                    runtime.start_config(cfg.bat, cfg.id)
                    self.app.notify(f"Launched {cfg.id} in Windows Terminal", timeout=4)
                    # Optimistically flip the screen into LOADING immediately
                    # so the user sees the state change without waiting for
                    # the next 2s poll. The flag is dropped automatically the
                    # moment the snapshot's manifest is detected.
                    self.app.loading_ids.add(cfg.id)
                    self.refresh_state()
                except Exception as e:
                    self.app.notify(f"Launch failed: {e}", severity="error", timeout=6)
        self.app.push_screen(
            ConfirmModal(f"Launch {cfg.id} ({cfg.bat})?\nA new console window will open."),
            _yes,
        )

    def _do_unload(self) -> None:
        cfg = self.cfg
        proc = self.app.running.get(cfg.id)
        if not proc:
            self.app.notify("Not running", severity="warning"); return
        def _yes(ok: bool):
            if not ok:
                return
            self.app.notify("Unloading vLLM (releasing VRAM, closing terminal)...", timeout=3)
            self.app.run_worker(
                lambda: self._unload_blocking(proc.pid, proc.port),
                thread=True, exclusive=False,
            )
        self.app.push_screen(
            ConfirmModal(f"Kill PID {proc.pid} on port {proc.port}?\nThis stops vLLM."),
            _yes,
        )

    def _unload_blocking(self, pid: int, port: int) -> None:
        try:
            from .. import runtime
            success, msg = runtime.kill_pid(pid, port=port)
        except Exception as e:
            success, msg = False, f"unload exception: {e}"
        self.app.call_from_thread(
            self.app.notify,
            ("Killed " if success else "Kill failed: ") + msg,
            severity="information" if success else "error",
            timeout=6,
        )
        self.app.call_from_thread(self.app.refresh_running)

    def _do_test(self) -> None:
        cfg = self.cfg
        port = cfg.port
        served = self.bundle.shared_defaults.get("served_model_name", "qwen3.6-27b-autoround")
        self.app.notify(
            "Benchmarking decode tok/s (300-token transformer-attention prompt)...",
            timeout=8,
        )
        self.app.run_worker(
            lambda: self._test_blocking(port, served),
            thread=True, exclusive=True,
        )

    def _test_blocking(self, port: int, served: str) -> None:
        try:
            from .. import inference

            def _on_progress(n: int) -> None:
                self.app.call_from_thread(
                    self.app.notify,
                    f"Benchmarking... {n}/300 tokens decoded",
                    timeout=4,
                )

            result = inference.test_chat(
                port, model=served, on_progress=_on_progress,
            )
            if not result.get("ok"):
                self.app.call_from_thread(
                    self.app.push_screen,
                    ResultModal("Benchmark failed", f"[#f85149]{result.get('error','?')}[/]"),
                )
                return

            def _fmt(v, spec: str, dash: str = ",") -> str:
                if v is None:
                    return dash
                try:
                    return format(v, spec)
                except Exception:
                    return str(v)

            ttft = _fmt(result.get("ttft_s"), ".2f")
            total = _fmt(result.get("total_s"), ".2f")
            decode_window = _fmt(result.get("decode_window_s"), ".2f")
            decode = _fmt(result.get("decode_tps"), ".1f")
            wall = _fmt(result.get("wall_tps"), ".1f")
            text = (result.get("text") or "").strip() or "[i #8b949e](empty response)[/]"
            preview = text if len(text) <= 600 else text[:600].rstrip() + " […]"
            body = (
                f"[b #3fb950]decode tok/s: {decode}[/]   "
                f"[#8b949e]wall tok/s:[/] {wall}\n\n"
                f"[#8b949e]prompt_tokens:[/]     {result.get('prompt_tokens', 0)}\n"
                f"[#8b949e]completion_tokens:[/] {result.get('completion_tokens', 0)}\n"
                f"[#8b949e]TTFT:[/]              {ttft}s   [#6e7681](prefill cost)[/]\n"
                f"[#8b949e]decode window:[/]     {decode_window}s\n"
                f"[#8b949e]total wall:[/]        {total}s\n\n"
                f"[b]Response preview:[/]\n{preview}\n\n"
                f"[#6e7681]Same prompt and metric as windows_tools\\bench.py, "
                f"compare directly against published numbers.[/]"
            )
            self.app.call_from_thread(
                self.app.notify,
                f"Benchmark complete: {decode} decode tok/s",
                timeout=6,
            )
            self.app.call_from_thread(
                self.app.push_screen, ResultModal(f"Benchmark → {self.cfg.id}", body)
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.app.call_from_thread(
                self.app.push_screen,
                ResultModal("Benchmark crashed", f"[#f85149]{e}[/]\n\n[#8b949e]{tb}[/]"),
            )
