"""
Bring the Windows Terminal window to the foreground from an external process.

Called by start-cc.bat after spawning a new WT tab. Since this process runs
OUTSIDE Windows Terminal, it cannot use AttachThreadInput (no message queue).
Instead it uses the SendInput(VK_MENU) ALT-key trick to convince Windows that
this process just received user input, which unlocks SetForegroundWindow.

Fallback: temporarily sets HWND_TOPMOST then removes it, forces Z-order
even if SetForegroundWindow still returns False.

Usage:  python activate_wt.py [window_title_substring]
        Defaults to searching for "Cursor" in the window title.
"""

import ctypes
import ctypes.wintypes
import sys
import time


user32 = ctypes.windll.user32

# ── SendInput structures ────────────────────────────────────────────────────

INPUT_KEYBOARD = 1
VK_MENU = 0x12  # Alt key
KEYEVENTF_KEYUP = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT_UNION)]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _send_alt_key():
    """Fake an ALT press+release to unlock SetForegroundWindow."""
    inputs = (INPUT * 2)()
    for i, flags in enumerate([0, KEYEVENTF_KEYUP]):
        inputs[i].type = INPUT_KEYBOARD
        inputs[i]._input.ki.wVk = VK_MENU
        inputs[i]._input.ki.dwFlags = flags
    user32.SendInput(2, inputs, ctypes.sizeof(INPUT))


def _find_wt_hwnd(title_hint: str) -> int | None:
    """Find a visible window whose title contains title_hint (case-insensitive)."""
    result = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        if title_hint.lower() in buf.value.lower():
            result.append(hwnd)
            return False  # stop on first match
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return result[0] if result else None


def _force_foreground(hwnd: int) -> bool:
    """Bring hwnd to foreground from a background process."""
    SW_RESTORE = 9
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_FLAGS = SWP_NOMOVE | SWP_NOSIZE
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2

    # Step 1: restore if minimised, wait for animation
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.05)

    # Already foreground?
    if user32.GetForegroundWindow() == hwnd:
        return True

    # Step 2: fake ALT key, tricks Windows into thinking we got input
    _send_alt_key()
    time.sleep(0.02)

    # Step 3: primary activation
    user32.SetForegroundWindow(hwnd)

    if user32.GetForegroundWindow() == hwnd:
        return True

    # Step 4: TOPMOST toggle fallback, forces Z-order change
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_FLAGS)
    user32.SetForegroundWindow(hwnd)
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_FLAGS)

    return bool(user32.GetForegroundWindow() == hwnd)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    title = sys.argv[1] if len(sys.argv) > 1 else "Cursor"

    # Poll briefly, wt.exe may not have finished creating the tab yet
    hwnd = None
    for _ in range(15):  # up to ~1.5 s
        hwnd = _find_wt_hwnd(title)
        if hwnd:
            break
        time.sleep(0.1)

    if not hwnd:
        # Broader fallback: any WT window
        hwnd = _find_wt_hwnd("WindowsTerminal")
        if not hwnd:
            sys.exit(2)

    ok = _force_foreground(hwnd)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
