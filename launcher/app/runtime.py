from __future__ import annotations
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

CREATE_NEW_CONSOLE = 0x00000010
CREATE_NO_WINDOW = 0x08000000


@dataclass
class RunningProc:
    pid: int
    port: int
    cmdline: str
    max_model_len: int | None
    mtp_n: int | None
    matched_id: str | None = None


def _netstat_pids(ports: list[int]) -> dict[int, int]:
    """Return {port: pid} for LISTENING tcp ports."""
    out: dict[int, int] = {}
    try:
        r = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            timeout=5, creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return out
    for line in (r.stdout or "").splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local = parts[1]
        try:
            port = int(local.rsplit(":", 1)[-1])
        except ValueError:
            continue
        if port in ports:
            try:
                out[port] = int(parts[-1])
            except ValueError:
                pass
    return out


def _cmdline_for(pid: int) -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            capture_output=True, text=True, timeout=8,
            creationflags=CREATE_NO_WINDOW,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


_RE_LEN = re.compile(r"--max-model-len[= ](\d+)")
_RE_MTP = re.compile(r'num_speculative_tokens"\s*:\s*(\d+)')


def _logs_dir() -> Path:
    """Mirror snapshots/_common._resolve_logs_dir(), write-tolerant."""
    env = os.environ.get("VLLM_WINDOWS_LOGS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    repo = here.parents[2] if len(here.parents) >= 3 else here.parent
    candidate = repo / "logs"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return Path(base) / "qwen36-windows-server" / "logs"


def _manifest_dir() -> Path:
    return _logs_dir() / "runtime"


def _read_manifests() -> list[dict]:
    """Read all <logs>/runtime/<port>.json. Returns list of dicts."""
    import json
    d = _manifest_dir()
    if not d.exists():
        return []
    out: list[dict] = []
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["__path__"] = str(p)
            out.append(data)
        except (OSError, ValueError):
            try: p.unlink()
            except OSError: pass
    return out


def _port_listening(port: int, host: str = "127.0.0.1",
                    timeout: float = 1.5) -> bool:
    """Locale-free port check, connect attempt, no netstat parsing.

    Default timeout is generous (1.5s) because vLLM holds the bound port
    while compiling kernels at boot, listen() has been called but the
    accept() loop isn't running yet, so a stingy timeout reports the
    port as 'not listening' even though it's clearly owned. We don't
    need a fast probe here, `detect_running` runs at 2s intervals.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _pid_alive(pid: int) -> bool:
    """Locale-free PID liveness via tasklist /fi (exit code, not text)."""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=5,
            creationflags=CREATE_NO_WINDOW,
        )
        # When PID exists, stdout has a CSV row starting with a quoted image name.
        # When absent, stdout is "INFO: No tasks..." (locale text varies), but
        # the row form is quote-prefixed, locale-stable.
        return bool((r.stdout or "").lstrip().startswith('"'))
    except Exception:
        return True  # fail-open: don't GC a manifest just because tasklist hiccuped


def _bat_basename(c) -> str:
    """Resolve the snapshot bat basename from a config (configs.yaml `bat:` field)."""
    raw = (c.bat or "").replace("/", "\\")
    return raw.rsplit("\\", 1)[-1].lower()


def detect_running(ports: list[int], configs) -> dict[str, RunningProc]:
    """Return {config_id: RunningProc} for matched configs.

    Primary path: read manifests written by snapshots on boot. Each manifest
    names the snapshot_id (== bat/py basename) authoritatively, so port
    collisions (start_speed/start_127k/start_mtp4 all on 5001) resolve
    correctly. We then verify the port is actually listening and the
    wrapper pid is still alive, both via locale-free probes (no netstat
    "LISTENING" parse, no Win32_Process WMI).

    Fallback: legacy port-only match for processes started before the
    manifest code shipped (no manifest exists). This preserves the prior
    behavior for pre-upgrade running servers.
    """
    result: dict[str, RunningProc] = {}
    matched_ports: set[int] = set()

    # ------- Primary: manifest-driven --------
    # Liveness rule: a manifest is considered LIVE if
    #   port-listening  OR  wrapper-pid-alive
    # We GC only when BOTH are false. The "port listening" check alone is
    # not authoritative during the ~60-90s vLLM compile window, where the
    # port has been listen()'d but accept() is starved by kernel-compile
    # work and a quick connect() probe times out. Killing the manifest in
    # that window flips the dashboard back to the legacy port-only
    # fallback and lights the wrong card. Conversely, the wrapper python
    # is reliably alive throughout the snapshot's lifetime, it's the
    # process holding the vllm subprocess open and tee'ing its log.
    by_bat = {_bat_basename(c): c for c in configs}
    for mf in _read_manifests():
        port = int(mf.get("port") or 0)
        if not port:
            continue
        wrapper_pid = int(mf.get("wrapper_pid") or 0)
        port_up = _port_listening(port)
        wrapper_up = bool(wrapper_pid) and _pid_alive(wrapper_pid)
        if not port_up and not wrapper_up:
            # Both signals say the snapshot is gone, safe to GC.
            try: Path(mf["__path__"]).unlink()
            except (OSError, KeyError): pass
            continue

        snap_bat = (mf.get("snapshot_bat") or "").lower()
        cfg = by_bat.get(snap_bat)
        if cfg is None:
            continue  # manifest references an unknown snapshot

        # Resolve the netstat pid for kill_pid, fall back to wrapper_pid.
        pid = wrapper_pid
        np = _netstat_pids([port]).get(port)
        if np:
            pid = np

        proc = RunningProc(
            pid=pid, port=port, cmdline="<from-manifest>",
            max_model_len=mf.get("max_model_len"),
            mtp_n=mf.get("mtp_n"),
            matched_id=cfg.id,
        )
        result[cfg.id] = proc
        matched_ports.add(port)

    # ------- Fallback: legacy netstat (process predates manifest support) -------
    legacy_ports = [p for p in ports if p not in matched_ports]
    if legacy_ports:
        pid_by_port = _netstat_pids(legacy_ports)
        for port, pid in pid_by_port.items():
            cmd = _cmdline_for(pid)
            m1 = _RE_LEN.search(cmd)
            m2 = _RE_MTP.search(cmd)
            ml = int(m1.group(1)) if m1 else None
            mn = int(m2.group(1)) if m2 else None
            proc = RunningProc(pid=pid, port=port, cmdline=cmd,
                               max_model_len=ml, mtp_n=mn)
            for c in configs:
                if c.port != port:
                    continue
                if ml is not None and c.ctx == ml:
                    if mn is None and c.mtp_n is None:
                        proc.matched_id = c.id; break
                    if mn is not None and c.mtp_n == mn:
                        proc.matched_id = c.id; break
            if proc.matched_id is None:
                for c in configs:
                    if c.port == port:
                        proc.matched_id = c.id; break
            if proc.matched_id and proc.matched_id not in result:
                result[proc.matched_id] = proc
    return result


def probe_ready(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> bool:
    """True iff vLLM has reached 'Application startup complete' on this port.

    Reads the snapshot's log file directly (truncated on each boot, so any
    marker found is from the current run) instead of issuing an HTTP GET
    against 127.0.0.1.

    Why log-grep instead of HTTP: TUN-mode VPN clients (sing-box / Nekobox
    TUN whitelist / Clash TUN / v2rayN TUN, etc.) intercept TCP at the
    network-adapter layer, *below* urllib. Even with
    urllib.request.ProxyHandler({}), which v1.3.5 added to bypass IE/
    registry HTTP proxies, the kernel still routes 127.0.0.1 traffic
    through the TUN adapter when the calling python.exe is in the TUN
    whitelist (or when sing-box's `strict_route` covers 127.0.0.0/8). The
    HTTP probe then hangs or returns the wrong content, the launcher
    never moves the snapshot from LOADING into READY, and the user reports
    "interface won't show that model is running" even though vLLM logged
    Application startup complete (issue #12, second report).

    Reading a local file is immune to any network-layer interception.
    The host / timeout params are accepted for API compatibility with
    callers and unused.
    """
    del host, timeout  # unused, kept for API stability
    log = _logs_dir() / f"vllm_server.{port}.log"
    if not log.is_file():
        return False
    try:
        size = log.stat().st_size
        with log.open("rb") as f:
            # Tail the last ~16 KiB; the readiness banner lands within the
            # last few hundred bytes of a successful boot. A 16 KiB slice
            # also keeps the read O(1) regardless of total log size.
            if size > 16384:
                f.seek(size - 16384)
            data = f.read()
    except OSError:
        return False
    return b"Application startup complete" in data


def clear_manifest_for_port(port: int) -> None:
    """Remove <logs>/runtime/<port>.json, called after a successful kill."""
    try:
        (_manifest_dir() / f"{port}.json").unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _wt_exe() -> str | None:
    """Locate Windows Terminal, prefer bundled portable, then system, then PATH."""
    import os, shutil
    root = os.environ.get("CC_PORTABLE_ROOT")
    candidates = []
    if root:
        candidates.append(os.path.join(root, "terminal", "WindowsTerminal.exe"))
    candidates += [
        r"C:\Program Files\WindowsTerminal\wt.exe",
        r"C:\Program Files\WindowsTerminal\WindowsTerminal.exe",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return shutil.which("wt") or shutil.which("WindowsTerminal")


def start_config(bat_path: str, config_id: str = "") -> None:
    """Launch the .bat in a new Windows Terminal tab (falls back to cmd console)."""
    import os
    cwd = os.path.dirname(bat_path)
    title = f"vLLM {config_id}" if config_id else "vLLM"
    wt = _wt_exe()
    if wt:
        # Reuse a named window so repeated launches stack as tabs.
        subprocess.Popen(
            [wt, "-w", "vllm-launcher", "new-tab", "-d", cwd,
             "--title", title, "cmd", "/k", bat_path],
            creationflags=CREATE_NO_WINDOW,
            cwd=cwd,
        )
        return
    subprocess.Popen(
        ["cmd", "/c", "start", "", "/D", cwd, bat_path],
        creationflags=CREATE_NEW_CONSOLE,
        cwd=cwd,
    )


def _ancestor_chain(pid: int, max_depth: int = 8) -> list[tuple[int, str]]:
    """Walk parent pids up to max_depth. Returns [(pid, name), ...] starting from pid."""
    chain: list[tuple[int, str]] = []
    cur = pid
    seen: set[int] = set()
    for _ in range(max_depth):
        if cur in seen or cur <= 0:
            break
        seen.add(cur)
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={cur}';"
                 " if ($p) { \"$($p.ProcessId)|$($p.Name)|$($p.ParentProcessId)\" }"],
                capture_output=True, text=True, timeout=8,
                creationflags=CREATE_NO_WINDOW,
            )
            line = (r.stdout or "").strip()
            if not line or "|" not in line:
                break
            spid, name, ppid = line.split("|", 2)
            chain.append((int(spid), name.strip().lower()))
            cur = int(ppid)
        except Exception:
            break
    return chain


def _all_descendants(root_pid: int) -> list[int]:
    """Return every descendant pid of root_pid (transitive), not including root."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "ForEach-Object { \"$($_.ProcessId)|$($_.ParentProcessId)\" }"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return []
    children: dict[int, list[int]] = {}
    for line in (r.stdout or "").splitlines():
        if "|" not in line:
            continue
        try:
            pid_s, ppid_s = line.split("|", 1)
            pid_i, ppid_i = int(pid_s), int(ppid_s)
        except ValueError:
            continue
        children.setdefault(ppid_i, []).append(pid_i)
    out: list[int] = []
    stack = [root_pid]
    seen: set[int] = {root_pid}
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            if ch in seen:
                continue
            seen.add(ch)
            out.append(ch)
            stack.append(ch)
    return out


def _kill_one(pid: int, tree: bool = True) -> tuple[bool, str]:
    try:
        args = ["taskkill", "/F", "/PID", str(pid)]
        if tree:
            args.insert(2, "/T")
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return False, str(e)


def _port_free(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return False
        except OSError:
            return True


_RE_LOG_PIDS = re.compile(r"(?:EngineCore|APIServer)\s+pid=(\d+)")


def _sweep_log_pids(log_dirs: list[Path]) -> set[int]:
    """Parse vLLM log files for EngineCore / APIServer pid markers."""
    pids: set[int] = set()
    for d in log_dirs:
        if not d.exists():
            continue
        for log in list(d.glob("vllm_server*.log")):
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _RE_LOG_PIDS.findall(text):
                try:
                    pids.add(int(m))
                except ValueError:
                    pass
    return pids


def _repo_log_dirs() -> list[Path]:
    """Best-effort log dirs to scan for EngineCore/APIServer pid markers.

    Mirrors snapshots/stop_vllm.py: VLLM_WINDOWS_LOGS env, then <repo>/logs/,
    then snapshot dir itself (legacy)."""
    dirs: list[Path] = []
    env_dir = os.environ.get("VLLM_WINDOWS_LOGS")
    if env_dir:
        dirs.append(Path(env_dir))
    here = Path(__file__).resolve()
    # launcher/app/runtime.py -> repo root is parents[2]
    repo = here.parents[2] if len(here.parents) >= 3 else here.parent
    dirs.append(repo / "logs")
    dirs.append(repo / "snapshots")
    return dirs


def kill_pid(pid: int, port: int | None = None) -> tuple[bool, str]:
    """Hard-stop a vLLM config: free CUDA/VRAM, close the WT tab, sweep orphans.

    Process tree on Windows when launched via `wt -> cmd /k start_*.bat`:
      WindowsTerminal -> OpenConsole -> cmd.exe (cmd /k)
        -> python.exe (start_*.py wrapper that tees logs)
           -> vllm.exe / python.exe (API server, what netstat returns)
              -> python.exe workers (EngineCore, etc.)

    Killing the netstat pid + /T only reaps the API server and workers; the
    wrapper python and cmd.exe survive. vLLM also occasionally reparents
    workers, leaving CUDA-holding orphans. Strategy:

      1. Snapshot every descendant of the parent cmd.exe BEFORE killing.
      2. Sweep `logs/vllm_server*.log` for EngineCore/APIServer pid markers.
      3. Kill the netstat pid tree first (fast VRAM release).
      4. Kill the parent cmd.exe tree (catches wrapper python + closes WT tab).
      5. Kill any snapshot/log pids still alive (catches reparented orphans).
      6. If a port was given, wait up to 10s for it to free.
    """
    msgs: list[str] = []
    ok_any = False

    # 1) locate parent cmd.exe (don't cross into the terminal itself)
    cmd_pid: int | None = None
    for ancestor_pid, name in _ancestor_chain(pid)[1:]:
        if name == "cmd.exe":
            cmd_pid = ancestor_pid
            break
        if name in ("windowsterminal.exe", "openconsole.exe", "conhost.exe"):
            break

    # 2) snapshot descendants of cmd (or pid if no cmd) BEFORE killing
    snap_root = cmd_pid if cmd_pid is not None else pid
    snapshot = set(_all_descendants(snap_root))
    snapshot.add(pid)
    if cmd_pid is not None:
        snapshot.add(cmd_pid)

    # 3) sweep log files for EngineCore / APIServer pids
    log_pids = _sweep_log_pids(_repo_log_dirs())
    if log_pids:
        msgs.append(f"log-pids:{sorted(log_pids)}")

    # 4) kill netstat pid first (fast VRAM release for the API server tree)
    ok, m = _kill_one(pid, tree=True)
    ok_any = ok_any or ok
    msgs.append(f"py({pid}):{m}")

    # 5) kill the cmd parent tree (wrapper python + closes WT tab)
    if cmd_pid is not None:
        ok, m = _kill_one(cmd_pid, tree=True)
        ok_any = ok_any or ok
        msgs.append(f"cmd({cmd_pid}):{m}")
    else:
        msgs.append("cmd:not-found")

    # 6) sweep: kill any snapshot pid and log-pid still alive
    survivors = snapshot | log_pids
    survivors.discard(pid)
    if cmd_pid is not None:
        survivors.discard(cmd_pid)
    if survivors:
        time.sleep(0.3)  # give the previous taskkills a moment
        killed_extra: list[int] = []
        for p in sorted(survivors):
            ok, _ = _kill_one(p, tree=True)
            if ok:
                killed_extra.append(p)
        if killed_extra:
            msgs.append(f"sweep:{killed_extra}")

    # 7) wait for the port to free (best effort)
    if port is not None:
        deadline = time.time() + 10
        while time.time() < deadline:
            if _port_free(port):
                msgs.append(f"port({port}):free")
                break
            time.sleep(0.4)
        else:
            msgs.append(f"port({port}):still-busy")

    # 8) GC the runtime manifest so the dashboard stops showing this card
    #    as RUNNING. Snapshots also clear it on clean exit, but external
    #    kills (this code path) need to unlink themselves.
    if port is not None:
        clear_manifest_for_port(port)

    return ok_any, " | ".join(msgs)
