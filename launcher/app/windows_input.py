"""Drop-in ``Input`` subclass with Windows-style text-editing shortcuts.

Textual's default ``Input`` widget follows emacs/readline conventions:

* ``Ctrl+A``  = go to line start (users expect: Select All)
* ``Ctrl+E``  = go to line end
* ``Ctrl+Shift+A`` = Select All (users expect: ``Ctrl+A``)
* No undo / redo at all.
* Double-click and triple-click do nothing.

This subclass makes every Windows text-editing convention work:

Keyboard
    ``Ctrl+A``                 Select all text
    ``Ctrl+Z``                 Undo
    ``Ctrl+Y``                 Redo
    ``Ctrl+Shift+Z``           Redo (Photoshop / VS / browser convention)
    ``Alt+Backspace``          Undo (legacy Win32 alias)
    ``Shift+Home/End``         Extend selection to line start/end
    ``Ctrl+Home/End``          Go to start/end (same as ``Home``/``End`` in a
                               single-line Input)
    ``Ctrl+Shift+Home/End``    Select to start/end
    ``Ctrl+Left/Right``        Jump one word
    ``Ctrl+Shift+Left/Right``  Select one word
    ``Ctrl+Backspace``         Delete word to the left
    ``Ctrl+Delete``            Delete word to the right
    ``Ctrl+C/X/V``             Copy / cut / paste
    ``Shift+Delete``           Cut (CUA legacy alias)
    ``Shift+Insert``           Paste (CUA legacy alias)
    ``Ctrl+Insert``            Copy (CUA legacy alias)

Mouse
    Single-click               Position cursor
    Double-click               Select word under cursor
    Triple-click               Select all
    Click-and-drag             Select range  (built into Textual 8.2+)
    Shift+click                Extend current selection to click position

Textual 8.2+ already implements: click-to-position, drag-to-select, Shift+arrow
selection, Ctrl+Shift+arrow word-selection, Shift+Home/End, Ctrl+C/X/V,
Backspace/Delete. This subclass adds the rest and rebinds ``Ctrl+A`` to
``select_all`` (the parent's ``home,ctrl+a`` entry is overridden because
subclass BINDINGS take precedence per-key, see ``DOMNode._merge_bindings``).

Undo/redo design
----------------
Win32 ``EDIT`` controls offer single-level undo only. WPF ``TextBox`` and
Notepad offer multi-level. We follow the Notepad/WPF behavior: a bounded
history of ``(value, cursor_position)`` snapshots, with consecutive character
typing coalesced into one undo group (typical behavior of every modern
desktop editor). Cursor navigation, cut, paste, delete, and word-delete each
break the typing group so they can be undone separately.

The undo stack is per-widget instance, lives only in memory, and resets when
the screen / app is rebuilt.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.widgets import Input
from textual.widgets._input import Selection


_MAX_UNDO_DEPTH = 200


def _classify(ch: str) -> int:
    """Return 0 for word chars, 1 for whitespace, 2 for punctuation."""
    if ch.isalnum() or ch == "_":
        return 0
    if ch.isspace():
        return 1
    return 2


class WindowsInput(Input):
    """Input widget with Windows-style text editing shortcuts and mouse."""

    # Opt out of Textual's screen-level "drag to highlight text across widgets"
    # mechanism. The Input has its own internal selection model (``self.selection``
    # backing the caret + selected range), which is what Ctrl+A / double-click /
    # triple-click need to talk to. Leaving ``ALLOW_SELECT`` at the inherited
    # default of True causes ``Widget._on_click`` (widget.py:4693) to fire
    # ``select_container.text_select_all()`` on a chain==3 click, which
    # highlights every sibling in the nearest scrollable ancestor, labels,
    # other Input fields, the whole row. Disabling it confines triple-click to
    # the input's own contents, which is what users expect.
    ALLOW_SELECT: ClassVar[bool] = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("home", "home", "Go to start", show=False),
        Binding("end", "end", "Go to end", show=False),
        Binding("ctrl+home", "home", "Go to start", show=False),
        Binding("ctrl+end", "end", "Go to end", show=False),
        Binding("ctrl+shift+home", "home(True)", "Select to start", show=False),
        Binding("ctrl+shift+end", "end(True)", "Select to end", show=False),
        Binding("ctrl+backspace", "delete_left_word", "Delete word left", show=False),
        Binding("ctrl+delete", "delete_right_word", "Delete word right", show=False),
        Binding("shift+insert", "paste", "Paste", show=False),
        Binding("ctrl+insert", "copy", "Copy", show=False),
        Binding("shift+delete", "cut", "Cut", show=False),
        Binding("ctrl+z", "undo", "Undo", show=False),
        Binding("ctrl+y", "redo", "Redo", show=False),
        Binding("ctrl+shift+z", "redo", "Redo", show=False),
        Binding("alt+backspace", "undo", "Undo", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Snapshots are (value, cursor_position) tuples. Each represents a
        # state we can return to. ``_last_op`` lets us coalesce consecutive
        # character-typing into one undo group (Notepad behavior).
        self._undo_stack: list[tuple[str, int]] = []
        self._redo_stack: list[tuple[str, int]] = []
        self._last_op: str = "init"
        self._suspend_history: bool = False

    # ------------------------------------------------------------------ undo

    def _record(self, op: str) -> None:
        """Snapshot current state before a mutating action.

        ``op`` labels the action class. Consecutive ``"type"`` ops coalesce
        into a single undo entry so a run of typed characters undoes as a
        unit, matching Notepad / VS / WPF behavior. Any other op forces a
        new entry, which is also why cursor navigation flips ``_last_op``
        to ``"nav"``, it breaks the typing group without itself recording.
        """
        if self._suspend_history:
            return
        snap = (self.value, self.cursor_position)
        if op == "type" and self._last_op == "type":
            # Coalesce: keep the snapshot from before the typing run started.
            return
        if self._undo_stack and self._undo_stack[-1] == snap:
            self._last_op = op
            return
        self._undo_stack.append(snap)
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._last_op = op

    def _apply_snapshot(self, snap: tuple[str, int]) -> None:
        value, cursor = snap
        self._suspend_history = True
        try:
            self.value = value
            self.cursor_position = max(0, min(cursor, len(value)))
        finally:
            self._suspend_history = False

    def action_undo(self) -> None:
        if not self._undo_stack:
            self.app.bell()
            return
        self._redo_stack.append((self.value, self.cursor_position))
        self._apply_snapshot(self._undo_stack.pop())
        self._last_op = "history"

    def action_redo(self) -> None:
        if not self._redo_stack:
            self.app.bell()
            return
        self._undo_stack.append((self.value, self.cursor_position))
        self._apply_snapshot(self._redo_stack.pop())
        self._last_op = "history"

    # ----------------------------------------------------- key / nav hooks

    async def _on_key(self, event: events.Key) -> None:
        if event.is_printable:
            self._record("type")
            await super()._on_key(event)
            return
        # Cursor navigation breaks the typing group so the next character
        # typed starts its own undo entry. We don't record here, the move
        # itself isn't a mutating op, but flipping _last_op is enough.
        if event.key in {
            "left", "right", "up", "down",
            "home", "end",
            "ctrl+left", "ctrl+right",
            "ctrl+home", "ctrl+end",
            "shift+left", "shift+right", "shift+up", "shift+down",
            "shift+home", "shift+end",
            "ctrl+shift+left", "ctrl+shift+right",
            "ctrl+shift+home", "ctrl+shift+end",
            "pageup", "pagedown",
        }:
            self._last_op = "nav"
        await super()._on_key(event)

    def _on_paste(self, event: events.Paste) -> None:
        self._record("paste")
        super()._on_paste(event)

    # --- mutating actions: snapshot before delegating to the parent -------

    def action_cut(self) -> None:
        self._record("cut")
        super().action_cut()

    def action_paste(self) -> None:
        self._record("paste")
        super().action_paste()

    def action_delete_left(self) -> None:
        self._record("delete")
        super().action_delete_left()

    def action_delete_right(self) -> None:
        self._record("delete")
        super().action_delete_right()

    def action_delete_left_word(self) -> None:
        self._record("delete_word")
        super().action_delete_left_word()

    def action_delete_right_word(self) -> None:
        self._record("delete_word")
        super().action_delete_right_word()

    def action_delete_left_all(self) -> None:
        self._record("delete_all")
        super().action_delete_left_all()

    def action_delete_right_all(self) -> None:
        self._record("delete_all")
        super().action_delete_right_all()

    # ------------------------------------------------------------ mouse

    async def _on_click(self, event: events.Click) -> None:
        if event.chain == 2:
            offset_x, _ = event.get_content_offset_capture(self)
            index = self._cell_offset_to_index(offset_x)
            start, end = self._word_span_at(index)
            if start != end:
                self.selection = Selection(start, end)
                event.stop()
            self._last_op = "nav"
        elif event.chain >= 3:
            self.select_all()
            event.stop()
            self._last_op = "nav"

    def _word_span_at(self, index: int) -> tuple[int, int]:
        """Return (start, end) of the word/whitespace/punctuation run at index."""
        text = self.value
        if not text:
            return (0, 0)
        index = max(0, min(index, len(text) - 1))
        target_class = _classify(text[index])
        start = index
        while start > 0 and _classify(text[start - 1]) == target_class:
            start -= 1
        end = index + 1
        while end < len(text) and _classify(text[end]) == target_class:
            end += 1
        return (start, end)
