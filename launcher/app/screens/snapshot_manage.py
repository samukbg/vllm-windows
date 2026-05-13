"""Snapshot manager screen, full CRUD editor for windows.configs[].

Master-detail layout: ListView of snapshot ids on the left, form on the right.
Save persists to configs.yaml AND rewrites the matching constants in the
snapshot's start_*.py file (PORT, TP, PP, USE_MTP, NUM_SPEC_TOKENS, CTX,
GPU_MEM_UTIL, MAX_NUM_BATCHED_TOKENS).

Mirrors the profile_manage.py pattern from the Claude Code Launcher.
"""
from __future__ import annotations

import copy
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, ListView, Select,
    Static, TextArea,
)

from ..windows_input import WindowsInput
from .. import snapshot_io as sio
from ..config import WinConfig, ConfigsBundle


# Editable fields shown in the form. (input_id, label, placeholder, default_str_for_None)
_NUMERIC_FIELDS = [
    ("port", "Port", "5001", ""),
    ("tp", "TP", "tensor-parallel", "1"),
    ("pp", "PP", "pipeline-parallel", "1"),
    ("mem_util", "GPU mem-util","0.0 - 1.0", "0.92"),
    ("mtp_n", "MTP n", "blank = no spec-decode", ""),
    ("ctx", "Context", "tokens", "32000"),
    ("decode_tps", "Decode tok/s","measured", ""),
    ("prefill_tps_cold", "Prefill tok/s cold", "measured", ""),
    ("power_cap_w", "Power cap", "watts", ""),
]

_TIERS = ("active", "legacy", "blocked")
_STATUSES = ("recommended", "experimental", "conditional", "superseded", "blocked")
_GPUS = ("GPU0", "GPU1", "GPU0+1")


class SnapshotManageScreen(Screen[bool]):
    """CRUD editor for vllm snapshots. Dismisses with True if anything changed."""

    DEFAULT_CSS = """
    SnapshotManageScreen { background: #0d1117; color: #e6edf3; }

    #sm-body { height: 1fr; }
    #sm-left { width: 40; min-width: 32; max-width: 48;
               background: #11161d; border-right: solid #30363d; }
    #sm-right { width: 1fr; padding: 0 2 0 2; layout: vertical; }
    #sm-form-scroll { height: 1fr; padding: 1 1 0 0; }

    .sm-section-title {
        color: #58a6ff;
        text-style: bold;
        padding: 1 1 1 1;
        height: auto;
    }

    #sm-list {
        height: 1fr;
        background: #11161d;
        scrollbar-size: 0 0;
    }
    #sm-list > ListItem {
        padding: 1 1 0 1;
        background: #11161d;
    }
    #sm-list > ListItem.-highlight {
        background: #161b22;
    }
    #sm-list:focus > ListItem.-highlight {
        background: #1a2733;
    }

    #sm-list-buttons {
        height: 3;
        padding: 0 1;
        background: #11161d;
    }
    #sm-list-buttons Button {
        margin: 0 1 0 0;
        min-width: 10;
        height: 3;
        content-align: center middle;
    }

    .sm-row {
        height: 3;
        margin-bottom: 1;
    }
    .sm-row Label {
        width: 22;
        padding: 1 1 0 0;
        color: #8b949e;
    }
    .sm-row Input {
        width: 1fr;
        height: 3;
    }
    .sm-row Select {
        width: 1fr;
        height: 3;
    }

    #sm-notes-row { height: 8; margin-top: 0; margin-bottom: 1; }
    #sm-notes-row Label { width: 22; padding: 1 1 0 0; color: #8b949e; }
    #sm-notes { height: 8; background: #161b22; border: solid #30363d; }

    #sm-form-buttons {
        height: 3;
        margin-top: 1;
        padding: 0;
        background: #0d1117;
    }
    #sm-form-buttons Button {
        margin: 0 1 0 0;
        min-width: 16;
        height: 3;
        content-align: center middle;
    }
    Button.sm-btn-primary  { background: #11202f; color: #58a6ff; text-style: bold; }
    Button.sm-btn-danger   { background: #2d0a0f; color: #f85149; }

    #sm-status {
        height: 1;
        padding: 0 2;
        color: #8b949e;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+n", "new", "New"),
        Binding("ctrl+d", "duplicate", "Duplicate"),
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, bundle: ConfigsBundle,
                 select_id: str | None = None) -> None:
        super().__init__()
        # Working copy, edits land in this list, written to disk only on Save.
        self.bundle = bundle
        self._snapshot_bundle = copy.deepcopy(bundle.windows)
        self._current_id: str | None = None  # None == editing an unsaved new entry
        self._initial_select = select_id
        self._form_tier: str = "active"
        self._form_status: str = "recommended"
        self._form_gpu: str = "GPU1"

    # ── Layout ───────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="sm-body"):
            with Vertical(id="sm-left"):
                yield Static("Snapshots", classes="sm-section-title")
                yield ListView(id="sm-list")
                with Horizontal(id="sm-list-buttons"):
                    yield Button("New", id="sm-btn-new", classes="sm-btn-primary")
                    yield Button("Duplicate", id="sm-btn-dup")
                    yield Button("Delete", id="sm-btn-del", classes="sm-btn-danger")
            with Vertical(id="sm-right"):
                yield Static("Edit Snapshot", classes="sm-section-title")
                with VerticalScroll(id="sm-form-scroll"):
                    with Horizontal(classes="sm-row"):
                        yield Label("ID")
                        yield WindowsInput(placeholder="snapshot key, e.g. my_64k",
                                    id="sm-in-id")
                    with Horizontal(classes="sm-row"):
                        yield Label("Tagline")
                        yield WindowsInput(placeholder="one-line description",
                                    id="sm-in-tagline")

                    with Horizontal(classes="sm-row"):
                        yield Label("Tier")
                        yield Select(
                            [(t, t) for t in _TIERS],
                            id="sm-sel-tier", value="active", allow_blank=False,
                        )
                    with Horizontal(classes="sm-row"):
                        yield Label("Status")
                        yield Select(
                            [(s, s) for s in _STATUSES],
                            id="sm-sel-status", value="recommended",
                            allow_blank=False,
                        )
                    with Horizontal(classes="sm-row"):
                        yield Label("GPU")
                        yield Select(
                            [(g, g) for g in _GPUS],
                            id="sm-sel-gpu", value="GPU1", allow_blank=False,
                        )

                    for fid, lab, placeholder, _ in _NUMERIC_FIELDS:
                        with Horizontal(classes="sm-row"):
                            yield Label(lab)
                            yield WindowsInput(placeholder=placeholder,
                                               id=f"sm-in-{fid}", type="text")

                    with Horizontal(id="sm-notes-row"):
                        yield Label("Notes")
                        yield TextArea("", id="sm-notes")

                with Horizontal(id="sm-form-buttons"):
                    yield Button("Save [Ctrl+S]", id="sm-btn-save",
                                 classes="sm-btn-primary")
                    yield Button("Revert", id="sm-btn-revert")
                    yield Button("Back [Esc]", id="sm-btn-back")

        yield Static("", id="sm-status")
        yield Footer()

    def on_mount(self) -> None:
        # Honor the select_id passed in (from DetailScreen → Edit) when it
        # actually exists in the bundle; otherwise fall back to the first row.
        existing_ids = {c.id for c in self._snapshot_bundle}
        target = (self._initial_select if self._initial_select in existing_ids
                  else (self._snapshot_bundle[0].id
                        if self._snapshot_bundle else None))
        self._refresh_list(select_id=target)
        self.query_one("#sm-list", ListView).focus()

    # ── List helpers ─────────────────────────────────────────────────────

    def _refresh_list(self, select_id: str | None = None) -> None:
        lv = self.query_one("#sm-list", ListView)
        lv.clear()
        target_index = 0
        for i, c in enumerate(self._snapshot_bundle):
            tier_color = {"active": "#3fb950", "legacy": "#d29922",
                          "blocked": "#f85149"}.get(c.tier, "#8b949e")
            label = (
                f"[{tier_color}]●[/]  [b]{c.id}[/]\n"
                f"  [#8b949e]{c.tagline}[/]"
            )
            item = ListItem(Label(label))
            item.data_id = c.id  # type: ignore[attr-defined]
            lv.append(item)
            if c.id == select_id:
                target_index = i
        if self._snapshot_bundle:
            lv.index = target_index
            self._load_into_form(self._snapshot_bundle[target_index].id)
        else:
            self._load_into_form(None)

    def _find(self, cid: str | None) -> WinConfig | None:
        if not cid:
            return None
        return next((c for c in self._snapshot_bundle if c.id == cid), None)

    def _load_into_form(self, cid: str | None) -> None:
        self._current_id = cid
        c = self._find(cid)
        if c is None:
            self._set_form_strings({"id": "", "tagline": ""})
            for fid, _, _, _ in _NUMERIC_FIELDS:
                self.query_one(f"#sm-in-{fid}", Input).value = ""
            self.query_one("#sm-notes", TextArea).text = ""
            self._form_tier = "active"
            self._form_status = "experimental"
            self._form_gpu = "GPU1"
            self._sync_selects_from_state()
            return

        self._set_form_strings({
            "id": c.id,
            "tagline": c.tagline,
        })
        numeric = {
            "port": c.port,
            "tp": c.tp,
            "pp": c.pp,
            "mem_util": c.mem_util,
            "mtp_n": c.mtp_n,
            "ctx": c.ctx,
            "decode_tps": c.decode_tps,
            "prefill_tps_cold": c.prefill_tps_cold,
            "power_cap_w": c.power_cap_w,
        }
        for fid, _, _, _ in _NUMERIC_FIELDS:
            v = numeric.get(fid)
            self.query_one(f"#sm-in-{fid}", Input).value = "" if v is None else str(v)

        self.query_one("#sm-notes", TextArea).text = c.notes or ""
        self._form_tier = c.tier or "active"
        self._form_status = c.status or "experimental"
        self._form_gpu = c.gpu or "GPU1"
        self._sync_selects_from_state()

    def _set_form_strings(self, values: dict[str, str]) -> None:
        for k, v in values.items():
            try:
                inp = self.query_one(f"#sm-in-{k}", Input)
            except Exception:
                continue
            inp.value = v
            inp.cursor_position = 0

    def _sync_selects_from_state(self) -> None:
        """Push self._form_tier/status/gpu into the Select widgets."""
        try:
            self.query_one("#sm-sel-tier", Select).value = self._form_tier
            self.query_one("#sm-sel-status", Select).value = self._form_status
            self.query_one("#sm-sel-gpu", Select).value = self._form_gpu
        except Exception:
            pass

    def _set_status(self, msg: str, kind: str = "info") -> None:
        bar = self.query_one("#sm-status", Static)
        color = {"info": "#58a6ff", "ok": "#3fb950",
                 "err": "#f85149", "warn": "#d29922"}.get(kind, "#8b949e")
        bar.update(f"[{color}]{msg}[/]")

    # ── Form harvest ─────────────────────────────────────────────────────

    def _read_form(self) -> dict:
        def _val(fid: str) -> str:
            return self.query_one(f"#sm-in-{fid}", Input).value.strip()
        out: dict = {
            "id": _val("id"),
            "tagline": _val("tagline"),
            "tier": self._form_tier,
            "status": self._form_status,
            "gpu": self._form_gpu,
            "notes": self.query_one("#sm-notes", TextArea).text,
        }
        for fid, _, _, _ in _NUMERIC_FIELDS:
            out[fid] = _val(fid)
        return out

    # ── Events ───────────────────────────────────────────────────────────

    def on_list_view_selected(self, ev: ListView.Selected) -> None:
        self._sync_from_list_item(ev.item)

    def on_list_view_highlighted(self, ev: ListView.Highlighted) -> None:
        # Arrow-key navigation fires Highlighted but not Selected. Without
        # this, the form keeps showing the previously-clicked entry while
        # the user thinks they've navigated, so Delete operates on the
        # wrong row.
        self._sync_from_list_item(ev.item)

    def _sync_from_list_item(self, item) -> None:
        if item is None:
            return
        cid = getattr(item, "data_id", None)
        if cid and cid != self._current_id:
            self._load_into_form(cid)

    def on_select_changed(self, ev: Select.Changed) -> None:
        sid = ev.select.id or ""
        val = ev.value
        if val is Select.BLANK:
            return
        if sid == "sm-sel-tier":
            self._form_tier = str(val)
        elif sid == "sm-sel-status":
            self._form_status = str(val)
        elif sid == "sm-sel-gpu":
            self._form_gpu = str(val)

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        bid = ev.button.id or ""
        if bid == "sm-btn-new":
            self.action_new()
        elif bid == "sm-btn-dup":
            self.action_duplicate()
        elif bid == "sm-btn-del":
            self.action_delete()
        elif bid == "sm-btn-save":
            self.action_save()
        elif bid == "sm-btn-revert":
            self._load_into_form(self._current_id)
            self._set_status("Reverted form", "info")
        elif bid == "sm-btn-back":
            self.action_go_back()

    # ── Actions ──────────────────────────────────────────────────────────

    def action_new(self) -> None:
        self._current_id = None
        self._set_form_strings({"id": "", "tagline": ""})
        for fid, _, _, default in _NUMERIC_FIELDS:
            self.query_one(f"#sm-in-{fid}", Input).value = default
        self.query_one("#sm-notes", TextArea).text = ""
        self._form_tier = "active"
        self._form_status = "experimental"
        self._form_gpu = "GPU1"
        self._sync_selects_from_state()
        self.query_one("#sm-in-id", Input).focus()
        self._set_status(
            "New snapshot, fill in ID and fields, Ctrl+S to save. "
            "A new start_<id>.py and .bat will be generated from start_speed.py.",
            "info",
        )

    def action_duplicate(self) -> None:
        c = self._find(self._current_id)
        if c is None:
            self._set_status("Nothing to duplicate", "warn"); return
        # Pick a unique suffix.
        suggest = f"{c.id}_copy"
        i = 2
        existing = {x.id for x in self._snapshot_bundle}
        while suggest in existing:
            suggest = f"{c.id}_copy{i}"
            i += 1
        self._current_id = None  # force-insert as new on save
        self.query_one("#sm-in-id", Input).value = suggest
        # Tagline
        self.query_one("#sm-in-tagline", Input).value = c.tagline + " (copy)"
        # numerics from the source
        for fid, _, _, _ in _NUMERIC_FIELDS:
            v = getattr(c, fid, None)
            self.query_one(f"#sm-in-{fid}", Input).value = "" if v is None else str(v)
        self.query_one("#sm-notes", TextArea).text = c.notes or ""
        self._form_tier = c.tier
        self._form_status = c.status
        self._form_gpu = c.gpu
        self._sync_selects_from_state()
        self.query_one("#sm-in-id", Input).focus()
        self._set_status(
            f"Duplicated '{c.id}' -> '{suggest}'. Edit fields, Ctrl+S to save "
            f"(generates start_{suggest}.py + .bat).",
            "info",
        )

    def action_delete(self) -> None:
        from .detail import ConfirmModal
        c = self._find(self._current_id)
        if c is None:
            self._set_status("Nothing to delete", "warn"); return
        msg = (
            f"[b #f85149]Delete snapshot '{c.id}'?[/]\n\n"
            f"This removes the entry from configs.yaml and deletes the\n"
            f"associated files:\n"
            f"  • {Path(c.py).name}\n"
            f"  • {Path(c.bat).name}\n\n"
            f"This cannot be undone from the launcher."
        )
        def _yes(ok: bool):
            if not ok: return
            removed = sio.delete_snapshot_files(Path(c.py), Path(c.bat))
            self._snapshot_bundle = [x for x in self._snapshot_bundle if x.id != c.id]
            # Persist to YAML immediately, avoids stale file references on
            # next launcher boot if the user closes without explicit save.
            self.bundle.windows = list(self._snapshot_bundle)
            sio.save_configs_yaml(self.bundle)
            self._refresh_list(
                select_id=self._snapshot_bundle[0].id if self._snapshot_bundle else None
            )
            self._set_status(
                f"Deleted '{c.id}' (removed {len(removed)} file(s) + YAML entry)",
                "ok",
            )
        self.app.push_screen(ConfirmModal(msg), _yes)

    def action_save(self) -> None:
        form = self._read_form()
        new_id = form["id"].strip()
        if not new_id:
            self._set_status("ID is required", "err"); return
        if any(c in new_id for c in " \t\n\r:\"'\\/"):
            self._set_status("ID must not contain whitespace, quotes, slashes, or colons",
                             "err"); return

        is_new = self._current_id is None
        is_rename = (not is_new) and self._current_id != new_id

        existing_ids = {c.id for c in self._snapshot_bundle if c.id != self._current_id}
        if new_id in existing_ids:
            self._set_status(f"ID '{new_id}' already exists, pick another", "err")
            return

        # Validate numerics. Accept empty for optional fields.
        try:
            port = int(form["port"]) if form["port"] else 0
            tp   = int(form["tp"] or 1)
            pp   = int(form["pp"] or 1)
            mem  = float(form["mem_util"] or 0.92)
            ctx  = int(form["ctx"] or 32000)
            mtp  = int(form["mtp_n"]) if form["mtp_n"] else None
            d_t  = float(form["decode_tps"]) if form["decode_tps"] else None
            p_t  = float(form["prefill_tps_cold"]) if form["prefill_tps_cold"] else None
            pcap = int(form["power_cap_w"]) if form["power_cap_w"] else None
        except ValueError as e:
            self._set_status(f"Numeric field invalid: {e}", "err"); return
        if not port:
            self._set_status("Port is required", "err"); return
        if not (0.0 < mem <= 1.0):
            self._set_status("mem_util must be in (0, 1]", "err"); return

        # Snapshot dir for path resolution.
        from ..config import _resolve_env
        env = _resolve_env()
        snaps_dir = Path(env["SNAPSHOTS_DIR"])
        new_py = snaps_dir / f"start_{new_id}.py"
        new_bat = snaps_dir / f"start_{new_id}.bat"

        # Source for new/dup file generation: start_speed.py (the blessed template).
        # If editing an existing config, keep the existing file paths.
        if is_new:
            template_py = snaps_dir / "start_speed.py"
            if not template_py.exists():
                self._set_status(
                    f"Template missing: {template_py.name} (need it for New/Dup)",
                    "err"); return
            try:
                sio.copy_py_template(template_py, new_py, {
                    "PORT": port, "TP": tp, "PP": pp, "USE_MTP": mtp is not None,
                    "NUM_SPEC_TOKENS": mtp if mtp is not None else 1,
                    "CTX": ctx, "GPU_MEM_UTIL": mem,
                })
            except FileExistsError:
                self._set_status(f"File already exists: {new_py.name}", "err"); return
            sio.write_bat(new_bat)
            new_cfg = WinConfig(
                id=new_id, tagline=form["tagline"], tier=form["tier"],
                status=form["status"],
                bat=str(new_bat), py=str(new_py),
                gpu=form["gpu"], tp=tp, pp=pp, mem_util=mem,
                ctx=ctx, port=port, mtp_n=mtp,
                decode_tps=d_t, prefill_tps_cold=p_t, power_cap_w=pcap,
                notes=form["notes"], raw={},
            )
            self._snapshot_bundle.append(new_cfg)
        else:
            cfg = self._find(self._current_id)
            assert cfg is not None
            # Update in place. On rename we also rename the .py/.bat files.
            if is_rename:
                old_py = Path(cfg.py); old_bat = Path(cfg.bat)
                if new_py.exists() or new_bat.exists():
                    self._set_status(
                        f"Target file exists: {new_py.name}/{new_bat.name}",
                        "err"); return
                if old_py.exists(): old_py.rename(new_py)
                if old_bat.exists(): old_bat.rename(new_bat)
                # Refresh bat to reflect new basename so it still runs the renamed .py.
                if new_bat.exists():
                    sio.write_bat(new_bat)
                cfg.id = new_id
                cfg.py = str(new_py); cfg.bat = str(new_bat)
            # Push numeric edits to the .py file.
            try:
                changes = sio.update_py_constants(Path(cfg.py), {
                    "PORT": port, "TP": tp, "PP": pp,
                    "USE_MTP": mtp is not None,
                    "NUM_SPEC_TOKENS": mtp if mtp is not None else 1,
                    "CTX": ctx, "GPU_MEM_UTIL": mem,
                })
            except Exception as e:
                self._set_status(f".py update failed: {e}", "err"); return
            # Apply form values to the dataclass + raw dict.
            cfg.tagline = form["tagline"]
            cfg.tier = form["tier"]; cfg.status = form["status"]
            cfg.gpu = form["gpu"]; cfg.tp = tp; cfg.pp = pp
            cfg.mem_util = mem; cfg.ctx = ctx; cfg.port = port
            cfg.mtp_n = mtp
            cfg.decode_tps = d_t; cfg.prefill_tps_cold = p_t
            cfg.power_cap_w = pcap
            cfg.notes = form["notes"]
            _ = changes  # status banner could include this if we wanted detail

        # Persist YAML, point bundle.windows at the working list, dump.
        self.bundle.windows = list(self._snapshot_bundle)
        try:
            sio.save_configs_yaml(self.bundle)
        except Exception as e:
            self._set_status(f"YAML save failed: {e}", "err"); return

        verb = "Created" if is_new else ("Renamed + saved" if is_rename else "Saved")
        self._refresh_list(select_id=new_id)
        self._set_status(f"{verb} '{new_id}'  (configs.yaml + snapshot files)", "ok")

    def action_go_back(self) -> None:
        self.dismiss(True)
