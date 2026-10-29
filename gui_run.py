"""
CWAssistant -- DeepCW Edition: point-and-click CW decoder.

One window, one job: copy CW with the DeepCW neural decoder and show the text.
Pick where the audio comes from:

  * FlexRadio  -- DAX Audio from a slice (enter the radio's IP address and the
                  slice).  The slice is only listened to, never retuned.
  * Sound card -- any Windows recording device, so it works with ICOM, Yaesu,
                  Kenwood, Elecraft ... (the radio's USB audio, a sound-card
                  interface, or a virtual audio cable).

The radio connection / audio receive / decode logic is in live_run.py (imported
and run directly, not duplicated); this file is only the window.  The work runs
in a background thread so the window stays responsive; Stop ends it cleanly
within about 2 seconds.

Settings are remembered between runs in cwassistant_settings.json next to this
program.  Nothing here ever transmits or retunes a radio.

Usage: just run it -- python gui_run.py (or double-click the packaged .exe).
"""

import json
import math
import os
import re
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import audio_input
from live_run import (build_arg_parser, run_flex_audio, run_soundcard_audio,
                       DetectorError, APP_VERSION)



# Sentinel pushed onto the log queue when the worker thread finishes --
# never call Tkinter methods (including root.after()) directly from a
# background thread; routing this through the same thread-safe queue as
# ordinary log lines lets the main thread's own poll loop notice it.
_WORKER_DONE = object()

# Settings file lives next to this script (or next to the packaged .exe,
# via sys.executable, when frozen by PyInstaller) so it survives between
# runs without needing anywhere else to write.
if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(_APP_DIR, "cwassistant_settings.json")

# Defaults mirror live_run.py's own argparse defaults exactly (see
# build_arg_parser() in live_run.py) -- kept as one literal dict here
# only for the GUI's field pre-fill, not as a second source of truth for
# what the *program* defaults to when run from the command line.
DEFAULTS = {
    "source": "flex",          # "flex" or "soundcard"
    "host": "",
    "api_port": 4992,
    "slice": 0,
    "dax_channel": 1,
    "udp_port": 41991,
    "stream_id": "",
    "device": "",              # the sound-card input's name as listed
}


def load_settings():
    settings = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            saved = json.load(f)
        settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass  # no settings file yet, or it's corrupt -- just use defaults
    return settings


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass  # not critical -- worst case, next run re-enters the fields



def _nice_tick_step(span, target_ticks=5):
    """Round Hz spacing for the tone meter's frequency tick marks -- one
    of 1/2/5/10 times a power of ten, picked so `span` divided by it
    comes out close to target_ticks. Avoids an arithmetically-correct but
    useless step like 116.67 Hz; an operator wants to see 400/450/500/
    550/600, not that.
    """
    if span <= 0:
        return 1.0
    raw_step = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual < 1.5:
        nice = 1
    elif residual < 3:
        nice = 2
    elif residual < 7:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


class DetectorApp:
    # Button/status colors -- shared constants so start/stop/idle states
    # stay visually consistent everywhere they're set.
    _NEUTRAL_BG = "#e6e6e6"
    _START_ACTIVE_BG = "#2e7d32"   # green -- running
    _STOP_READY_BG = "#c62828"     # red -- click to stop
    _STOP_ACTIVE_BG = "#8e0000"    # darker red -- stop in progress
    _IDLE_TEXT = "#555555"         # dark gray text on a disabled light-gray button (readable)
    _IDLE_STATUS = "#e65100"       # orange -- "idle, waiting for you to press Start"

    # Radio IP / Slice / Center frequency boxes -- blue with white print,
    # per Ken's request (2026-09-15). Plain tk widgets (tk.Entry /
    # tk.OptionMenu), not ttk.Entry/ttk.Combobox, for the same reason the
    # Start/Stop buttons above are plain tk.Button: ttk widgets draw with
    # the native Windows theme on that platform, which ignores a custom
    # background color entirely -- already learned the hard way once on
    # this project, see start_button's own comment.
    _INPUT_BG = "#1565c0"
    _INPUT_FG = "white"

    def __init__(self, root):
        self.root = root
        self.root.title(f"CWAssistant - DeepCW Edition  v{APP_VERSION}")
        # Height: tall enough that both the raw log and the decode
        # pane below it get a real, usable height without the operator
        # needing to manually resize the window first (2026-09-15 -- see
        # the row-weight comments on those two panes for how they split
        # this). Width is sized dynamically in _size_window_to_content(),
        # called right after _build_widgets() below, to just past the
        # Center frequency box's right edge (Ken, 2026-09-16) -- computed
        # from the real built layout rather than a hardcoded guess that
        # could drift out of sync with it.
        self._DEFAULT_WINDOW_HEIGHT = 720

        self.settings = load_settings()
        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.running = False
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        self.style.configure("Bold.TLabel", font=("TkDefaultFont", 9, "bold"))
        # A greyed-out (locked while running) sound-card box must stay readable.
        self.style.map("TCombobox",
                       foreground=[("disabled", "black")],
                       fieldbackground=[("disabled", "#d9e8fb")])

        self._build_widgets()
        self._size_window_to_content()
        self._on_source_change()  # show the FlexRadio or the sound-card row to match the saved source
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _size_window_to_content(self):
        """Set the default window width to just past the right edge of
        the connection row (Ken, 2026-09-16), measured from the
        actual built layout rather than a hardcoded guess -- so it can't
        drift out of sync if a font or padding change ever shifts that
        row's width later.

        Real bug, 2026-09-16 (Ken, on real Windows): the first version of
        this used winfo_rootx() (ABSOLUTE screen position) on both the
        entry box and the root window and subtracted them. That happened
        to work under this project's own Linux/Xvfb test environment,
        but on real Windows the window manager hasn't necessarily placed
        the toplevel on screen yet at this point in startup -- Tk doesn't
        guarantee winfo_rootx() is meaningful before the window is
        actually mapped, just that update_idletasks() has settled
        internal widget sizes/layout. That produced a nonsense negative
        width (a bad geometry string like "-91x760") and crashed the
        packaged .exe outright before the window ever appeared. Fixed by
        switching to conn_frame's own winfo_reqwidth() -- a RELATIVE
        measurement (how wide that row's own content naturally wants to
        be), which Tk computes purely from the widget tree's own
        geometry negotiation and has nothing to do with where the window
        manager has (or hasn't) placed the window on screen.
        """
        self.root.update_idletasks()
        # main's own left+right padding (padding=10 on each side, see
        # _build_widgets()) plus conn_frame's own row padding (padx=6 on
        # each side, from the `pad` dict its grid() call uses) -- conn_frame
        # itself has no trailing pack() padding after the last (Center
        # frequency) box, so this is what actually carries "just past the
        # box" past its bare right edge rather than sitting flush against it.
        margin = 10 + 10 + 6 + 6
        # Measure both rows (only one is shown at a time) and take the wider.
        width = max(self.flex_row.winfo_reqwidth(), self.sound_row.winfo_reqwidth()) + margin
        # Defensive floor: never hand Tk a geometry string that could be
        # zero or negative (see the real crash above) even if some future
        # change to this layout throws the above math off.
        width = max(width, 760)
        self._window_width = width
        self.root.geometry(f"{width}x{self._DEFAULT_WINDOW_HEIGHT}")
        self.root.minsize(width, 500)

    # -- UI layout ------------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 6, "pady": 4}

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(main, text="CWAssistant  --  DeepCW Edition", style="Title.TLabel").grid(
            row=row, column=0, sticky="w", padx=6, pady=(0, 8))
        title_row = row
        row += 1

        # -- Source selector + advanced-settings toggle -----------------------
        source_frame = ttk.Frame(main)
        source_frame.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1
        ttk.Label(source_frame, text="Audio source", style="Bold.TLabel").pack(side="left", padx=(0, 8))
        self.source_var = tk.StringVar(
            value=self.settings["source"] if self.settings["source"] in ("flex", "soundcard") else "flex")
        ttk.Radiobutton(source_frame, text="FlexRadio (DAX Audio)", variable=self.source_var,
                        value="flex", command=self._on_source_change).pack(side="left")
        ttk.Radiobutton(source_frame, text="Sound card (ICOM, Yaesu, Kenwood, other radios)",
                        variable=self.source_var, value="soundcard",
                        command=self._on_source_change).pack(side="left", padx=(16, 0))
        self.advanced_visible = tk.BooleanVar(value=False)
        # Top right, level with the title, so it never gets pushed off the edge
        # of the window by the (long) sound-card radio button text.
        # A button whose wording flips: "Show advanced settings" <-> "Hide advanced settings".
        self.advanced_toggle_var = tk.StringVar(value="Show advanced settings")
        self.advanced_toggle = ttk.Button(
            main, textvariable=self.advanced_toggle_var, command=self._on_advanced_button)
        self.advanced_toggle.grid(row=title_row, column=1, sticky="e", padx=6, pady=(0, 8))

        # -- Source details: ONE of the two rows below is shown at a time -------
        # Kept inside self.conn_frame so _size_window_to_content() can read the
        # natural width of whichever row is showing (see that method).
        # Plain tk.Entry / tk.OptionMenu (not ttk) so the blue background shows
        # on Windows -- ttk widgets draw with the native theme there and ignore
        # a custom background colour.
        conn_frame = ttk.Frame(main)
        self.conn_frame = conn_frame
        conn_frame.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        self.flex_row = ttk.Frame(conn_frame)
        self.flex_row.grid(row=0, column=0, sticky="w")
        ttk.Label(self.flex_row, text="Radio IP address", style="Bold.TLabel").pack(
            side="left", padx=(0, 4))
        self.host_var = tk.StringVar(value=self.settings["host"])
        tk.Entry(self.flex_row, textvariable=self.host_var, width=16,
                 bg=self._INPUT_BG, fg=self._INPUT_FG, insertbackground=self._INPUT_FG,
                 relief="sunken", highlightthickness=0).pack(side="left", padx=(0, 16))
        ttk.Label(self.flex_row, text="Slice", style="Bold.TLabel").pack(side="left", padx=(0, 4))
        self.slice_var = tk.StringVar(value=self._slice_num_to_label(self.settings["slice"]))
        # Just A/B -- every Flex model has at least these two, so this works
        # unchanged across the whole model line.
        slice_menu = tk.OptionMenu(self.flex_row, self.slice_var, "A (0)", "B (1)")
        self.slice_menu = slice_menu
        slice_menu.config(disabledforeground=self._INPUT_FG,   # locked-while-running stays readable
                           bg=self._INPUT_BG, fg=self._INPUT_FG,
                           activebackground=self._INPUT_BG, activeforeground=self._INPUT_FG,
                           highlightthickness=0, relief="sunken", width=6)
        slice_menu["menu"].config(bg=self._INPUT_BG, fg=self._INPUT_FG)
        slice_menu.pack(side="left")

        self.sound_row = ttk.Frame(conn_frame)
        self.sound_row.grid(row=0, column=0, sticky="w")
        ttk.Label(self.sound_row, text="Sound card input", style="Bold.TLabel").pack(
            side="left", padx=(0, 4))
        self.device_var = tk.StringVar(value=self.settings["device"])
        self.device_combo = ttk.Combobox(self.sound_row, textvariable=self.device_var,
                                         state="readonly", width=48, values=[])
        self.device_combo.pack(side="left", padx=(0, 8))
        self.device_refresh_button = ttk.Button(self.sound_row, text="Refresh list",
                                                command=self._refresh_devices)
        self.device_refresh_button.pack(side="left")
        self._device_map = {}           # label -> device number

        self.advanced_frame = ttk.LabelFrame(main, text="Advanced (FlexRadio only)", padding=8)
        self.advanced_row = row
        row += 1
        self._build_advanced_widgets()

        # -- Start/Stop + status ---------------------------------------------
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        row += 1
        # Plain tk.Button (not ttk) so the Start/Stop colors below actually
        # show up on Windows -- ttk buttons draw with the native OS theme,
        # which ignores custom background colors there.
        # disabledforeground matters here: a plain tk.Button left to its
        # default greys out the text while state="disabled", which is
        # exactly when Start is showing its "running" green and Stop is
        # showing its "click me" red -- without this the white text those
        # states set gets silently overridden back to gray.
        self.start_button = tk.Button(
            btn_frame, text="Start", command=self._on_start, width=9,
            font=("TkDefaultFont", 9, "bold"), bg=self._NEUTRAL_BG,
            disabledforeground="white")
        self.start_button.pack(side="left", padx=(0, 6))
        self.stop_button = tk.Button(
            btn_frame, text="Stop", command=self._on_stop, state="disabled", width=9,
            font=("TkDefaultFont", 9, "bold"), bg=self._NEUTRAL_BG,
            disabledforeground=self._IDLE_TEXT)
        self.stop_button.pack(side="left")
        # "Clear" -- manually blanks both text panes for a clean start,
        # without needing to Stop/Start the whole run (Ken, 2026-09-16).
        # Yellow to match the log window's own "hit" highlight color
        # (self.log_text's "hit" tag, #ffeb3b) rather than a new one-off
        # color. Plain tk.Button, same reasoning as Start/Stop above.
        self.clear_button = tk.Button(
            btn_frame, text="Clear", command=self._on_clear, width=7,
            font=("TkDefaultFont", 9, "bold"), bg="#ffeb3b", fg="black",
            activebackground="#ffeb3b", activeforeground="black")
        self.clear_button.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="IDLE - press Start")
        self.status_label = tk.Label(btn_frame, textvariable=self.status_var,
                                      font=("TkDefaultFont", 10, "bold"), fg=self._IDLE_STATUS)
        self.status_label.pack(side="left", padx=12)
        # DeepCW decode mode only -- the audio note (pitch) DeepCW found, e.g.
        # "Note: 650 Hz" (see the "TONE: " log lines in live_run.py).  Blank
        # whenever no tone is found.
        self.tone_var = tk.StringVar(value="")
        self.tone_label = tk.Label(btn_frame, textvariable=self.tone_var,
                                    font=("TkDefaultFont", 10, "bold"), fg="black")
        self.tone_label.pack(side="left", padx=(0, 8))
        # Sending speed estimated from DeepCW's own time-stamped characters
        # (see estimate_wpm() in deepcw_backend.py, "WPM: " lines from live_run.py).
        self.wpm_var = tk.StringVar(value="")
        self.wpm_label = tk.Label(btn_frame, textvariable=self.wpm_var, width=7, anchor="w",
                                   font=("TkDefaultFont", 10, "bold"), fg="black")
        self.wpm_label.pack(side="left", padx=(0, 8))

        # -- Input level bar ---------------------------------------------------
        # Peak of the incoming audio, about 5 times a second ("LEVEL: " lines).
        # Answers "is audio arriving, and is it far too quiet or clipping" at a
        # glance -- the first thing to check when setting up a new radio.
        level_frame = ttk.Frame(main)
        level_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 2))
        row += 1
        ttk.Label(level_frame, text="Input level:").pack(side="left", padx=(0, 6))
        self._LEVEL_W, self._LEVEL_H = 260, 16
        self._overload_until = 0.0
        self.level_canvas = tk.Canvas(level_frame, width=self._LEVEL_W, height=self._LEVEL_H,
                                       bg="black", highlightthickness=1, highlightbackground="#555555")
        self.level_canvas.pack(side="left")
        self.level_hint_var = tk.StringVar(value="")
        self.level_hint_label = ttk.Label(level_frame, textvariable=self.level_hint_var,
                                          foreground="#555555", font=("TkDefaultFont", 8, "bold"))
        self.level_hint_label.pack(side="left", padx=8)

        # -- Audio note meter --------------------------------------------------
        # DeepCW mode -- a live "where's my audio note relative to
        # the expected CW pitch range" display (see CWCopyStreamer.
        # last_tone_hz/last_tone_spectrum and the "SPEC: " log lines in
        # live_run.py). Exists because of a real operator report: copy
        # quality changed as the audio pitch was retuned, which raised
        # the question of whether the pitch had drifted outside the
        # range this program expects a CW note to live in. Drawn full-
        # width on its own row (like CWget's own frequency graph) rather
        # than squeezed into the button row, and horizontal by request --
        # frequency runs low-to-high left-to-right, with the actual
        # measured spectrum shape traced so the operator can see the
        # real audio peak, not just trust one computed number: a red
        # vertical line marks the software's own pick (same convention
        # as CWget's own red line), blank when nothing plausible was
        # found in range that pass.
        main.rowconfigure(row, weight=0)
        tone_frame = ttk.Frame(main)
        tone_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
        row += 1
        tone_frame.columnconfigure(1, weight=1)
        ttk.Label(tone_frame, text="Audio note spectrum:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.tone_low_var = tk.StringVar(value="")
        ttk.Label(tone_frame, textvariable=self.tone_low_var, font=("TkDefaultFont", 7)).grid(
            row=0, column=1, sticky="w")
        self.tone_high_var = tk.StringVar(value="")
        ttk.Label(tone_frame, textvariable=self.tone_high_var, font=("TkDefaultFont", 7)).grid(
            row=0, column=1, sticky="e")
        self._TONE_CANVAS_W = 400
        # Split into the spectrum trace itself (unchanged size/shape from
        # the smoothed-curve version Ken already approved) plus a thin
        # strip below it just for frequency tick labels -- Ken asked
        # whether the midpoint between low/high really means the
        # midpoint Hz (yes) and whether values could be marked along
        # that line, so this adds the marks rather than making him do
        # that arithmetic by eye every time.
        self._TONE_SPECTRUM_H = 60
        self._TONE_TICK_ZONE_H = 18
        self._TONE_CANVAS_H = self._TONE_SPECTRUM_H + self._TONE_TICK_ZONE_H
        self.tone_canvas = tk.Canvas(tone_frame, width=self._TONE_CANVAS_W, height=self._TONE_CANVAS_H,
                                      bg="black", highlightthickness=1, highlightbackground="#555555")
        self.tone_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(1, 0))
        self.tone_value_var = tk.StringVar(value="")
        ttk.Label(tone_frame, textvariable=self.tone_value_var,
                  font=("TkDefaultFont", 8, "bold"), foreground="#4fc3f7").grid(
            row=2, column=0, columnspan=2, sticky="w")

        # -- Status log -----------------------------------------------------------
        # Small black box for connection messages and errors.
        # Decoded DeepCW text does NOT go here -- it has its own big pane below.
        main.rowconfigure(row, weight=2)
        log_frame = ttk.Frame(main)
        log_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", **pad)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=5, state="disabled", wrap="word",
                                 bg="black", fg="white", insertbackground="white",
                                 font=("Consolas", 12))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = scrollbar.set
        row += 1

        # -- DeepCW decode pane --------------------------------------------------
        # The one decode window of this edition.  The DeepCW neural network
        # reads a rolling 16-second window and settles text about 4-5 seconds
        # behind the signal; the text is raw network output (no dictionary).
        # Append-only, like a teleprinter.  The black box above is just the
        # status log.
        main.rowconfigure(row, weight=8)
        deepcw_frame = ttk.Frame(main)
        deepcw_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", **pad)
        deepcw_frame.columnconfigure(0, weight=1)
        deepcw_frame.rowconfigure(1, weight=1)
        ttk.Label(deepcw_frame,
                  text="DeepCW decoded copy (dim text at the end is the newest words, still being firmed up)",
                  font=("TkDefaultFont", 8, "bold"), foreground="#555555").grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.deepcw_text = tk.Text(deepcw_frame, height=10, state="disabled", wrap="word",
                                    bg="#1f1608", fg="#ffcc80", insertbackground="#ffcc80",
                                    font=("Consolas", 13))
        self.deepcw_text.tag_configure("tentative", foreground="#8d6e63")
        self._deepcw_tentative = ""
        self._deepcw_tail = ""
        self.deepcw_text.grid(row=1, column=0, sticky="nsew")
        deepcw_scroll = ttk.Scrollbar(deepcw_frame, command=self.deepcw_text.yview)
        deepcw_scroll.grid(row=1, column=1, sticky="ns")
        self.deepcw_text["yscrollcommand"] = deepcw_scroll.set
        row += 1

        # -- Credit line ----------------------------------------------------
        # Version shown here too, not just the title bar -- a screenshot
        # of the window (how problems actually got reported/debugged
        # this project) crops or covers the title bar more often than
        # not, but this line is always in frame.
        ttk.Label(main, text=f"by W9IE with Claude  --  v{APP_VERSION}", foreground="#555555",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="e", padx=6, pady=(4, 0))

    def _build_advanced_widgets(self):
        f = self.advanced_frame
        pad = {"padx": 6, "pady": 3}
        self.adv_vars = {}

        # Fourth element is a short plain-English meaning shown to the
        # right of the field, with its suggested (default) value appended
        # -- Ken asked for this directly so he doesn't have to remember or
        # ask what each one does; the suggested value shown is always
        # DEFAULTS[key] itself (see below) so this can't drift out of
        # sync with what a blank/reset field would actually use.
        fields = [
            ("api_port", "API port", "int",
             "TCP port the radio's API listens on"),
            ("dax_channel", "DAX channel", "int",
             "Which DAX Audio channel number to use"),
            ("udp_port", "UDP port", "int",
             "UDP port the DAX Audio arrives on (not 4991, so other FlexRadio software can run too)"),
            ("stream_id", "Stream ID (hex, optional)", "str",
             "Blank = auto-detect; only set this if auto-detect fails"),
        ]
        for i, (key, label, _kind, meaning) in enumerate(fields):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", **pad)
            var = tk.StringVar(value=str(self.settings[key]))
            ttk.Entry(f, textvariable=var, width=16).grid(row=i, column=1, sticky="w", **pad)
            self.adv_vars[key] = var
            suggested = DEFAULTS[key]
            suggested_display = "blank" if suggested == "" else suggested
            ttk.Label(f, text=f"{meaning}  (suggested: {suggested_display})",
                      foreground="#555555", font=("TkDefaultFont", 8),
                      wraplength=360, justify="left").grid(
                row=i, column=2, sticky="w", **pad)

    def _on_source_change(self):
        """Show the row that belongs to the chosen audio source."""
        if self.source_var.get() == "soundcard":
            self.flex_row.grid_remove()
            self.sound_row.grid()
            if not self.device_combo["values"]:
                self._refresh_devices()
        else:
            self.sound_row.grid_remove()
            self.flex_row.grid()

    def _refresh_devices(self):
        """(Re)build the sound-card list.  Keeps the previous choice when it is
        still there, otherwise picks the Windows default input."""
        previous = self.device_var.get()
        try:
            devices = audio_input.list_input_devices()
        except audio_input.AudioInputError as e:
            self._device_map = {}
            self.device_combo["values"] = []
            self.device_var.set("")
            self._log(f"ERROR: {e}")
            return
        self._device_map = {label: idx for label, idx in devices}
        labels = [label for label, _ in devices]
        self.device_combo["values"] = labels
        if previous in self._device_map:
            self.device_var.set(previous)
        else:
            self.device_var.set(labels[0] if labels else "")
        if not labels:
            self._log("No sound-card recording devices were found.")

    # The DeepCW network has no "=" in its alphabet: the BT (pause / new-paragraph)
    # prosign, dah-dit-dit-dit-dah, comes out as the two letters "BT".  Operators
    # read it as "=", so a standalone BT word is shown as "=".  A BT inside a word
    # (BTU, ...) is left alone.  It needs the space after it, so a BT at the very
    # end of the text so far stays "BT" until the next character arrives.
    _BT_RE = re.compile(r"(?<![A-Z0-9])BT(?= )")

    def _append_deepcw_text(self, text):
        # Settled text goes in BEFORE the dim preview, which is re-added after.
        w = self.deepcw_text
        w["state"] = "normal"
        if w.tag_ranges("tentative"):
            w.delete("tentative.first", "tentative.last")
        tail = self._deepcw_tail
        combined = self._BT_RE.sub("=", tail + text)
        if tail:
            w.delete(f"end-1c-{len(tail)}c", "end-1c")      # redo the last few characters
        w.insert("end-1c", combined)
        self._deepcw_tail = combined[-12:]
        if self._deepcw_tentative:
            w.insert("end-1c", self._show_tentative(self._deepcw_tentative), "tentative")
        w.see("end")
        w["state"] = "disabled"

    def _show_tentative(self, text):
        """Preview text with the same BT -> '=' rule (looking at the settled
        text just before it for the word boundary)."""
        prefix = self._deepcw_tail[-1:] or " "
        return self._BT_RE.sub("=", prefix + text)[1:]

    def _set_deepcw_tentative(self, text):
        """The newest couple of seconds of DeepCW text, shown dim at the end of
        the pane.  It can still change; the next settled text replaces it."""
        self._deepcw_tentative = text
        self.deepcw_text["state"] = "normal"
        if self.deepcw_text.tag_ranges("tentative"):
            self.deepcw_text.delete("tentative.first", "tentative.last")
        if text:
            self.deepcw_text.insert("end-1c", self._show_tentative(text), "tentative")
        self.deepcw_text.see("end")
        self.deepcw_text["state"] = "disabled"

    def _on_advanced_button(self):
        self.advanced_visible.set(not self.advanced_visible.get())
        self._toggle_advanced()

    def _toggle_advanced(self):
        self.advanced_toggle_var.set("Hide advanced settings" if self.advanced_visible.get()
                                     else "Show advanced settings")
        if self.advanced_visible.get():
            self.advanced_frame.grid(row=self.advanced_row, column=0, columnspan=2,
                                      sticky="ew", padx=6, pady=4)
        else:
            self.advanced_frame.grid_forget()

    @staticmethod
    def _slice_num_to_label(n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 1
        return f"{chr(ord('A') + n)} ({n})"

    @staticmethod
    def _slice_label_to_num(label):
        # label looks like "B (1)"
        try:
            return int(label.split("(")[1].rstrip(")"))
        except (IndexError, ValueError):
            return 1

    # -- logging ----------------------------------------------------------

    def _log(self, message):
        """Called from the worker thread -- just queues the line; the
        actual Text widget update happens on the main thread via
        _poll_log_queue(), since Tkinter widgets aren't thread-safe.
        """
        self.log_queue.put(str(message))

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is _WORKER_DONE:
                    self._on_worker_done()
                    continue
                if item.startswith("WPM: "):
                    n = item[len("WPM: "):].strip()
                    self.wpm_var.set(f"{n} WPM" if n else "")
                    continue
                if item.startswith("LEVEL: "):
                    try:
                        self._update_level(float(item[len("LEVEL: "):]))
                    except ValueError:
                        pass
                    continue
                if item.startswith("TONE: "):
                    note = item[len("TONE: "):].strip()
                    self.tone_var.set(f"Note: {note}" if note else "")
                    continue
                if item.startswith("DEEPCW: "):
                    self._append_deepcw_text(item[len("DEEPCW: "):])
                    continue
                if item.startswith("DEEPCWT: "):
                    self._set_deepcw_tentative(item[len("DEEPCWT: "):])
                    continue
                if item.startswith("SPEC: "):
                    parts = item[len("SPEC: "):].split("|", 3)
                    if len(parts) == 4:
                        try:
                            hz = None if parts[0] == "NONE" else float(parts[0])
                            bins = [float(b) for b in parts[3].split(",")] if parts[3] else []
                            self._update_tone_meter(hz, float(parts[1]), float(parts[2]), bins)
                        except ValueError:
                            pass
                    continue
                self.log_text["state"] = "normal"
                self.log_text.insert("end", item + "\n")
                self.log_text.see("end")
                self.log_text["state"] = "disabled"
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    # -- start/stop ---------------------------------------------------------

    def _collect_args(self):
        """Builds the argparse.Namespace the run functions in live_run.py expect,
        from the current form fields -- reuses live_run.py's own parser
        defaults for anything not shown in the form, so the window never
        drifts out of sync with the command-line defaults.
        """
        ns = build_arg_parser().parse_args([])
        ns.source = self.source_var.get()

        if ns.source == "soundcard":
            label = self.device_var.get()
            if label not in self._device_map:
                raise ValueError("Choose a sound-card input first (press 'Refresh list' if the "
                                 "list is empty).")
            ns.device = self._device_map[label]
            return ns

        host = self.host_var.get().strip()
        if not host:
            raise ValueError("Enter the radio's IP address.")
        ns.host = host
        ns.slice = self._slice_label_to_num(self.slice_var.get())

        def _get_int(key):
            raw = self.adv_vars[key].get().strip()
            return int(raw) if raw else DEFAULTS[key]

        ns.api_port = _get_int("api_port")
        ns.dax_channel = _get_int("dax_channel")
        ns.udp_port = _get_int("udp_port")
        ns.stream_id = self.adv_vars["stream_id"].get().strip() or None
        return ns

    def _current_settings_dict(self):
        return {
            "source": self.source_var.get(),
            "host": self.host_var.get().strip(),
            "api_port": self.adv_vars["api_port"].get(),
            "slice": self._slice_label_to_num(self.slice_var.get()),
            "dax_channel": self.adv_vars["dax_channel"].get(),
            "udp_port": self.adv_vars["udp_port"].get(),
            "stream_id": self.adv_vars["stream_id"].get(),
            "device": self.device_var.get(),
        }

    # Input-level bar scale: -50 dB (empty) .. 0 dB (full scale).  The GOOD zone is
    # where the radio's audio is comfortably above the noise floor yet nowhere near
    # clipping; DeepCW reads the tone's contrast against the noise, so a louder signal
    # decodes no better, but clipping distorts the tone and does hurt.
    _LEVEL_DB_MIN = -50.0
    _GOOD_LOW_DB = -25.0
    _GOOD_HIGH_DB = -6.0
    _OVERLOAD_DB = -1.0
    _OVERLOAD_HOLD_SEC = 1.5

    def _level_x(self, db):
        frac = (db - self._LEVEL_DB_MIN) / (0.0 - self._LEVEL_DB_MIN)
        return min(1.0, max(0.0, frac)) * self._LEVEL_W

    def _update_level(self, peak):
        """Input-level bar.  Colours: blue-grey = low, green = good (the dark green
        strip behind the bar marks the good zone, -25 to -6 dB), yellow = getting loud,
        red = overload.  An overload keeps a red block at the right end and the warning
        text for about a second and a half, so a quick spike is not missed."""
        c = self.level_canvas
        c.delete("all")
        h = self._LEVEL_H
        gx0, gx1 = self._level_x(self._GOOD_LOW_DB), self._level_x(self._GOOD_HIGH_DB)
        c.create_rectangle(gx0, 0, gx1, h, fill="#12361a", outline="")      # the good zone
        for x in (gx0, gx1):
            c.create_line(x, 0, x, h, fill="#7cb342")
        now = time.monotonic()
        if peak <= 0:
            self._overload_until = 0.0
            self.level_hint_var.set("green zone = good level")
            self.level_hint_label.config(foreground="#555555")
            return
        db = 20.0 * math.log10(max(peak, 1e-6))
        if db >= self._OVERLOAD_DB:
            self._overload_until = now + self._OVERLOAD_HOLD_SEC
        if db >= self._OVERLOAD_DB:
            color = "#c62828"
        elif db >= self._GOOD_HIGH_DB:
            color = "#f9a825"
        elif db >= self._GOOD_LOW_DB:
            color = "#2e7d32"
        else:
            color = "#607d8b"
        c.create_rectangle(0, 3, self._level_x(db), h - 3, fill=color, outline="")
        overloaded = now < self._overload_until
        self.level_hint_label.config(foreground="#c62828" if overloaded else "#555555")
        if overloaded:
            c.create_rectangle(self._LEVEL_W - 14, 0, self._LEVEL_W, h, fill="#c62828", outline="")
            self.level_hint_var.set("OVERLOAD -- turn the radio's audio output down")
        elif db >= self._GOOD_HIGH_DB:
            self.level_hint_var.set("getting loud")
        elif db >= self._GOOD_LOW_DB:
            self.level_hint_var.set("good level")
        elif db < -40:
            self.level_hint_var.set("very quiet")
        else:
            self.level_hint_var.set("a little low")

    def _update_tone_meter(self, hz, low, high, bins):
        """Horizontal spectrum strip (low Hz at the left, high Hz at the
        right, matching how CWget itself lays out its own frequency
        graph) -- `bins` is the actual measured shape of the audio in
        that range (see DeepCWStreamer._make_spectrum()), traced as
        a filled outline so the operator can see the real peak, not just
        one computed number. The red vertical line marks `hz`, this
        program's own pick of which peak is the CW tone -- same
        convention as CWget's own red line. hz is None when nothing
        plausible was found in [low, high] this pass (true silence, or
        tuned outside the range) -- the shape (if any) still draws, just
        without a red line to go with it, since there's no confident
        pick to mark.
        """
        self.tone_high_var.set(f"{high:.0f} Hz")
        self.tone_low_var.set(f"{low:.0f} Hz")
        c = self.tone_canvas
        c.delete("all")
        # Use the canvas's real (stretched) width so the trace fills the whole strip.
        w = max(int(c.winfo_width()), 100) if c.winfo_width() > 1 else self._TONE_CANVAS_W
        h = self._TONE_CANVAS_H
        spectrum_h = self._TONE_SPECTRUM_H

        # Reserve real headroom above the tallest possible peak (v=1.0) so
        # it reads as a peak inside the box, not as touching/"clipped
        # against" the top edge -- Ken reported the old, shorter canvas
        # looked clipped at the top. baseline sits near the bottom of the
        # SPECTRUM portion (not the full canvas -- the tick-label strip
        # below it is separate); the drawable span between them is scaled
        # by TOP_MARGIN_FRAC, well short of the full spectrum height.
        TOP_MARGIN_FRAC = 0.22
        baseline_y = spectrum_h - 3
        top_y = spectrum_h * TOP_MARGIN_FRAC
        span = baseline_y - top_y

        if bins:
            # Bin CENTERS (not left/right edges) run through Tkinter's own
            # spline smoothing (smooth=True below) -- gives a flowing
            # curve through the actual measured shape rather than the
            # flat-topped bar/block look the old left/right-edge version
            # produced (each bin drawn as its own little rectangle).
            # Human preference (Ken): "more of a sine wave rather than
            # the block display."
            n = len(bins)
            step = w / n
            points = [(0, baseline_y)]
            for i, v in enumerate(bins):
                x = (i + 0.5) * step
                y = baseline_y - max(0.0, min(1.0, v)) * span
                points.append((x, y))
            points.append((w, baseline_y))
            flat = [coord for point in points for coord in point]
            c.create_polygon(flat, fill="#1a4a1a", outline="limegreen", width=1, smooth=True)

        # Frequency tick marks along the bottom, in the strip reserved
        # for them -- so the meter answers "what Hz is this point?"
        # directly instead of Ken having to guess/interpolate between the
        # low/high labels above (he asked specifically: does the midpoint
        # between 400 and 600 mean 500 Hz -- yes, and this is that made
        # visible). _nice_tick_step() picks a round Hz spacing (1/2/5/10
        # x a power of ten) rather than dividing evenly into an odd
        # number like 116.67, so the labels are ones an operator would
        # actually want to read (a 400-600 Hz range gets 400/450/500/
        # 550/600, not some awkward fraction).
        if high > low:
            step = _nice_tick_step(high - low)
            tick_y = spectrum_h + 2
            label_y = spectrum_h + (self._TONE_TICK_ZONE_H - 2) / 2 + 2
            first_tick = math.ceil(low / step) * step
            tick = first_tick
            while tick <= high + 1e-6:
                frac = (tick - low) / (high - low)
                x = frac * w
                c.create_line(x, spectrum_h, x, tick_y + 3, fill="#555555")
                # Anchor edge labels to hug the border instead of
                # centering (which would run them half off the canvas at
                # the very first/last tick, right where low/high usually
                # land).
                if x < 12:
                    anchor = "w"
                elif x > w - 12:
                    anchor = "e"
                else:
                    anchor = "c"
                c.create_text(x, label_y, text=f"{tick:.0f}", fill="#ffffff",
                               font=("TkDefaultFont", 8, "bold"), anchor=anchor)
                tick += step

        if hz is None:
            self.tone_value_var.set("no CW tone found in the 200-1400 Hz range")
            return
        frac = 0.5 if high <= low else (hz - low) / (high - low)
        frac = min(1.0, max(0.0, frac))
        x = frac * w
        c.create_line(x, 0, x, h, fill="#ff5252", width=2)
        self.tone_value_var.set(f"Red line = the CW tone DeepCW is following: {hz:.0f} Hz")

    def _clear_tone_meter(self):
        self.tone_canvas.delete("all")
        self.tone_high_var.set("")
        self.tone_low_var.set("")
        self.tone_value_var.set("")

    def _clear_display(self):
        """Blank both text areas (the status log and the DeepCW pane).
        Shared by _on_start() (always starts from a blank slate) and the
        Clear button.  Only touches the DISPLAY, never the running decoder,
        so it is safe to press mid-run.
        """
        self.log_text["state"] = "normal"
        self.log_text.delete("1.0", "end")
        self.log_text["state"] = "disabled"
        self.deepcw_text["state"] = "normal"
        self.deepcw_text.delete("1.0", "end")
        self.deepcw_text["state"] = "disabled"
        self._deepcw_tentative = ""
        self._deepcw_tail = ""
        self._clear_tone_meter()
        self.wpm_var.set("")
        self._update_level(0.0)

    def _on_clear(self):
        self._clear_display()

    def _on_start(self):
        if self.running:
            return
        try:
            args = self._collect_args()
        except ValueError as e:
            messagebox.showerror("Check your settings", str(e))
            return

        save_settings(self._current_settings_dict())

        self._clear_display()
        self.tone_var.set("")
        self.wpm_var.set("")

        self.stop_event.clear()
        self.running = True
        self.start_button.config(state="disabled", bg=self._START_ACTIVE_BG, fg="white")
        self.stop_button.config(state="normal", bg=self._STOP_READY_BG, fg="white",
                                disabledforeground="white")
        if args.source == "soundcard":
            self.status_var.set("RUNNING - sound card")
        else:
            self.status_var.set(f"RUNNING - Slice {chr(ord('A') + args.slice)}")
        self.status_label.config(fg=self._START_ACTIVE_BG)
        # The slice / sound-card choice is only read when Start is pressed, so lock
        # it while running (it greys out) instead of letting a change look like it
        # should take effect.  Stop, change it, Start again.
        self.slice_menu.config(state="disabled")
        self.device_combo.config(state="disabled")

        detector_fn = run_soundcard_audio if args.source == "soundcard" else run_flex_audio

        def worker():
            try:
                detector_fn(args, log=self._log, should_stop=self.stop_event.is_set)
                self._log("(run ended)")
            except DetectorError as e:
                self._log(f"ERROR: {e}")
            except Exception as e:  # noqa: BLE001 -- surface anything unexpected to the log
                self._log(f"UNEXPECTED ERROR: {e}")
            finally:
                self.log_queue.put(_WORKER_DONE)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _on_worker_done(self):
        self.running = False
        self.start_button.config(state="normal", bg=self._NEUTRAL_BG, fg="black")
        self.stop_button.config(state="disabled", bg=self._NEUTRAL_BG, fg="black",
                                disabledforeground=self._IDLE_TEXT)
        self.status_var.set("IDLE - press Start")
        self.status_label.config(fg=self._IDLE_STATUS)
        self.slice_menu.config(state="normal")
        self.device_combo.config(state="readonly")
        self.tone_var.set("")
        self.wpm_var.set("")
        self._update_level(0.0)

    def _on_stop(self):
        if not self.running:
            return
        self.status_var.set("STOPPING...")
        self.status_label.config(fg=self._STOP_READY_BG)
        self.stop_button.config(bg=self._STOP_ACTIVE_BG, state="disabled")
        self.stop_event.set()

    def _on_close(self):
        if self.running:
            self.stop_event.set()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=3.0)
        self.root.destroy()


def main():
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
