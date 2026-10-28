"""
OPTIONAL, REMOVABLE trial backend: DeepCW neural CW decoder (ONNX).

Trial started 2026-10-07 at Ken's request ("on a trial basis ... be able to
remove it if we decide to").  To REMOVE the trial completely: delete this
file (and, if it was ever copied in, the `deepcw_model` folder).  Nothing
else in CWAssistant imports this module, so deleting it restores the exact
previous behaviour.

LICENSE NOTE: the DeepCW model/engine (github.com/e04/deepcw-engine) is
AGPL-3.0-only.  Using it for personal testing is fine; *distributing* a
CWAssistant.exe that bundles it would pull the AGPL's obligations onto
this program (source must be offered to recipients under AGPL terms).
Decide that before ever shipping it to other hams.

How it works: the model eats a log1p magnitude spectrogram of 400-1200 Hz
audio sampled at 3200 Hz and emits per-frame character probabilities that
are collapsed with greedy CTC.  The reference example only accepts 5-20 s
clips, so decode_long() cuts longer audio into overlapping windows.

onnxruntime is imported lazily so a machine without it can still import
this file (and the rest of CWAssistant never notices).
"""

import json
import math
import os
import sys
import threading

import numpy as np

MIN_SEC = 5.0
MAX_SEC = 20.0


class DeepCWError(Exception):
    pass


class DeepCWDecoder:
    def __init__(self, model_dir):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise DeepCWError("onnxruntime is not installed "
                              "(pip install onnxruntime)") from exc
        meta_path = os.path.join(model_dir, "model.onnx.json")
        model_path = os.path.join(model_dir, "model.onnx")
        if not (os.path.exists(meta_path) and os.path.exists(model_path)):
            raise DeepCWError(f"model.onnx / model.onnx.json not found in {model_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.sr = int(self.meta["sample_rate"])
        self.chars = list(self.meta["chars"])
        self.blank = int(self.meta["blank_index"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=so,
                                            providers=["CPUExecutionProvider"])
        m = self.meta
        bin_hz = self.sr / m["fft_length"]
        self._start = int(math.ceil(m["spectrogram_min_freq_hz"] / bin_hz))
        self._stop = int(math.floor(m["spectrogram_max_freq_hz"] / bin_hz)) + 1
        self._win = np.hanning(m["fft_length"] + 1)[:-1].astype(np.float32)

    # -- low level -------------------------------------------------------
    def _resample(self, audio, sr):
        from scipy.signal import resample_poly
        if int(sr) == self.sr:
            return np.asarray(audio, dtype=np.float32)
        g = math.gcd(int(round(sr)), self.sr)
        return resample_poly(audio, self.sr // g, int(round(sr)) // g).astype(np.float32)

    def _spectrogram(self, audio):
        n_fft = int(self.meta["fft_length"])
        hop = int(self.meta["hop_length"])
        pad = n_fft // 2
        a = np.pad(audio, (pad, pad), mode="reflect")
        frames = 1 + (len(a) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
        spec = np.abs(np.fft.rfft(a[idx] * self._win, n=n_fft, axis=1))
        spec = spec[:, self._start:self._stop].astype(np.float32)
        return np.log1p(spec)[None, None, :, :].astype(np.float32)

    def _greedy(self, log_probs):
        path = log_probs[0].argmax(axis=-1)
        out, prev = [], None
        for i in path:
            i = int(i)
            if i == self.blank:
                prev = None
                continue
            if i != prev:
                out.append(self.chars[i])
            prev = i
        return "".join(out)

    def decode_timed(self, audio_at_model_rate):
        """audio already at the model's own sample rate (3200 Hz), real,
        roughly +-1 scale.  Returns [(seconds_from_start, char), ...] --
        each character stamped with the time its CTC spike fired (about the
        END of that character), which is what lets a streaming caller stitch
        overlapping windows together without duplicating or losing letters.
        """
        a = np.asarray(audio_at_model_rate, dtype=np.float32)
        need = int(MIN_SEC * self.sr)
        if len(a) < need:
            a = np.concatenate([a, np.zeros(need - len(a), np.float32)])
        out = self.session.run([self.meta["onnx_output_name"]],
                               {self.meta["onnx_input_name"]: self._spectrogram(a)})[0]
        path = out[0].argmax(axis=-1)
        step = self.meta["hop_length"] / self.sr
        res, prev = [], None
        for i, p in enumerate(path):
            p = int(p)
            if p == self.blank:
                prev = None
                continue
            if p != prev:
                res.append((i * step, self.chars[p]))
            prev = p
        return res

    # -- public ----------------------------------------------------------
    def decode(self, audio, sample_rate):
        """Decode one clip (real-valued audio).  Clips shorter than MIN_SEC
        are zero-padded; longer than MAX_SEC raise (use decode_long)."""
        a = self._resample(audio, sample_rate)
        if len(a) / self.sr > MAX_SEC + 0.01:
            raise DeepCWError("clip longer than 20 s; use decode_long()")
        if len(a) / self.sr < MIN_SEC:
            a = np.concatenate([a, np.zeros(int(MIN_SEC * self.sr) - len(a), np.float32)])
        spec = self._spectrogram(a)
        out = self.session.run([self.meta["onnx_output_name"]],
                               {self.meta["onnx_input_name"]: spec})[0]
        return self._greedy(out)

    def decode_long(self, audio, sample_rate, window_sec=15.0):
        """Cut long audio into windows (no overlap handling -- a character
        straddling a cut can be damaged; fine for a trial/benchmark)."""
        a = self._resample(audio, sample_rate)
        step = int(window_sec * self.sr)
        pieces = []
        for s in range(0, len(a), step):
            seg = a[s:s + step]
            if len(seg) < self.sr * 1.0:
                break
            pieces.append(self.decode(seg, self.sr))
        return " ".join(p.strip() for p in pieces)


# ---------------------------------------------------------------------------
# Streaming wrapper used by CWAssistant (Copy mode, "DeepCW" checkbox)
# ---------------------------------------------------------------------------

def _program_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _model_candidates():
    here = _program_dir()
    cands = []
    bundled = getattr(sys, "_MEIPASS", None)      # model packed inside the .exe (--add-data)
    if bundled:
        cands.append(os.path.join(bundled, "deepcw_model"))
    return cands + [os.path.join(here, "deepcw_model"),
            here,
            os.path.join(here, "deepcw_model", "deepcw_model"),   # doubled folder from a zip extract
            os.path.join(os.getcwd(), "deepcw_model")]


def find_model_dir():
    """Folder holding model.onnx + model.onnx.json: a `deepcw_model` folder
    next to the program (or the .exe), else the program's own folder."""
    for cand in _model_candidates():
        if (os.path.exists(os.path.join(cand, "model.onnx"))
                and os.path.exists(os.path.join(cand, "model.onnx.json"))):
            return cand
    return None


def describe_model_search():
    """Plain-English account of where we looked and what was there -- used in
    the 'model not found' message so the problem can be seen at a glance."""
    lines = []
    for cand in _model_candidates():
        if not os.path.isdir(cand):
            lines.append(f"  {cand}   -> folder does not exist")
            continue
        have = [n for n in ("model.onnx", "model.onnx.json")
                if os.path.exists(os.path.join(cand, n))]
        missing = [n for n in ("model.onnx", "model.onnx.json") if n not in have]
        lines.append(f"  {cand}   -> " + ("missing " + " and ".join(missing) if missing else "OK"))
    return "\n".join(lines)


_MORSE_UNITS = {}
for _ch, _code in {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.',
        'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.',
        'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-',
        'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..',
        '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
        '6': '-....', '7': '--...', '8': '---..', '9': '----.', '/': '-..-.', '?': '..--..',
        ',': '--..--', '.': '.-.-.-'}.items():
    # length of the character in dit-units: dits 1, dahs 3, one dit-gap between elements
    _MORSE_UNITS[_ch] = sum(1 if _s == '.' else 3 for _s in _code) + (len(_code) - 1)


def estimate_wpm(items):
    """Sending speed in WPM from DeepCW's own time-stamped characters.

    DeepCW marks each character at (about) its end, so the time between two
    neighbouring characters is the second character's own keying length plus
    the gap before it (3 dit-units inside a word, 7 across a space).  Adding
    those expected units up over a run of characters and fitting time against
    units gives the dit length; WPM = 1.2 / dit (PARIS standard).  A straight
    line fit over many characters averages out the small jitter in where each
    character's mark lands.  Returns None when there is not enough text."""
    pts = []
    total = 0.0
    prev_t = None
    space = False
    for t, c in items:
        if c == ' ':
            space = True
            continue
        u = _MORSE_UNITS.get(c)
        if u is None:                      # unknown character: restart the chain
            prev_t, space = None, False
            continue
        if prev_t is None:
            total = 0.0
        else:
            total += u + (7 if space else 3)
        pts.append((t, total))
        prev_t, space = t, False
    if len(pts) < 6:
        return None
    t = np.array([p[0] for p in pts])
    u = np.array([p[1] for p in pts])
    if u.max() <= 0:
        return None
    slope = np.polyfit(u, t, 1)[0]
    if slope <= 0:
        return None
    wpm = 1.2 / slope
    return wpm if 4.0 <= wpm <= 70.0 else None


class DeepCWStreamer:
    """Feeds a continuous audio stream to DeepCW and returns settled text.

    Same push()/flush() shape as CWCopyStreamer: push() takes a chunk of
    complex samples (DAX Audio I/Q pairs, or real file audio cast to complex)
    and returns whatever NEW text has become settled.  The heavy lifting
    (about 0.4 s per window on a modest PC) runs on a background thread so
    the radio's UDP reader is never stalled.

    Method: keep the last `window_sec` seconds; every `hop_sec` of new audio
    find the tone, slide it to `target_hz` (the middle of DeepCW's 400-1200 Hz
    hearing range) so ANY dial pitch works, decimate to 3200 Hz and decode.
    Characters carry timestamps, and only characters older than `margin_sec`
    before the window's end are committed (they have plenty of right-hand
    context); the newer ones are re-decoded by the next window.  flush()
    commits the rest.  Settled text is roughly hop+margin (~2-3 s) behind the
    signal; `tentative` holds the newest, still-changing words for a live preview.
    """

    def __init__(self, sample_rate, model_dir=None, window_sec=16.0, hop_sec=1.0,
                 margin_sec=1.0, target_hz=800.0, min_sec=6.0, min_contrast=2.0):
        model_dir = model_dir or find_model_dir()
        if model_dir is None:
            raise DeepCWError("DeepCW model not found. Looked in:\n" + describe_model_search()
                              + "\nPut the 'deepcw_model' folder (with model.onnx and "
                              "model.onnx.json inside) in the same folder as gui_run.py.")
        self.decoder = DeepCWDecoder(model_dir)
        self.sr = float(sample_rate)
        self.window_n = int(window_sec * self.sr)
        self.keep_n = int((window_sec + hop_sec + 2.0) * self.sr)
        self.hop_n = int(hop_sec * self.sr)
        self.min_n = int(min_sec * self.sr)
        self.margin = float(margin_sec)
        self.target_hz = float(target_hz)
        self.min_contrast = float(min_contrast)
        g = math.gcd(int(round(self.sr)), self.decoder.sr)
        self._up = self.decoder.sr // g
        self._down = int(round(self.sr)) // g

        self._cv = threading.Condition()
        self._buf = np.zeros(0, dtype=np.complex64)
        self._buf_start = 0          # absolute index of _buf[0]
        self._total = 0
        self._last_decode_total = 0
        self._pending = []
        self._committed = []         # recent settled (time, char) pairs, for stitching
        self._last_char = None
        self._closing = False
        self._flush_req = False
        self._flush_done = threading.Event()
        self._error = None
        self.last_tone_hz = None
        self.last_wpm = None         # smoothed sending speed (None = not enough text)
        self._wpm_hist = []
        self._wpm_misses = 0
        self.tentative = ""          # newest, not-yet-settled text (may still change)
        self.spec_seq = 0            # bumps whenever last_spectrum is refreshed
        self.last_spectrum = None    # (low_hz, high_hz, [0..1 bins]) for the audio-note meter
        self._thread = threading.Thread(target=self._run, name="deepcw", daemon=True)
        self._thread.start()

    # -- public ----------------------------------------------------------
    def push(self, chunk):
        c = np.asarray(chunk).astype(np.complex64, copy=False)
        with self._cv:
            self._buf = np.concatenate([self._buf, c])
            self._total += len(c)
            if len(self._buf) > self.keep_n:
                drop = len(self._buf) - self.keep_n
                self._buf = self._buf[drop:]
                self._buf_start += drop
            if (self._total - self._last_decode_total >= self.hop_n
                    and self._total >= self.min_n):
                self._cv.notify()
            return self._take_pending()

    def flush(self, timeout=60.0):
        """Decode whatever is left and return all remaining settled text."""
        self._flush_done.clear()
        with self._cv:
            self._flush_req = True
            self._cv.notify()
        self._flush_done.wait(timeout)
        with self._cv:
            return self._take_pending()

    def close(self):
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join(timeout=5.0)

    def take_error(self):
        err, self._error = self._error, None
        return err

    # -- internals -------------------------------------------------------
    def _take_pending(self):
        text = "".join(self._pending)
        self._pending.clear()
        return text

    def _run(self):
        while True:
            with self._cv:
                while True:
                    if self._closing:
                        return
                    due = (self._total - self._last_decode_total >= self.hop_n
                           and self._total >= self.min_n)
                    if due or self._flush_req:
                        break
                    self._cv.wait(0.25)
                final = self._flush_req
                snap = self._buf[-self.window_n:].copy()
                snap_start = self._buf_start + (len(self._buf) - len(snap))
                self._last_decode_total = self._total
            items = []
            try:
                if len(snap) >= int(1.0 * self.sr):
                    items = self._decode_snapshot(snap, snap_start)
            except Exception as exc:  # noqa: BLE001 -- never kill the worker
                self._error = f"{type(exc).__name__}: {exc}"
            end_t = (snap_start + len(snap)) / self.sr
            limit = end_t if final else end_t - self.margin
            with self._cv:
                self._commit(items, limit)
                self.tentative = ("" if final else
                                  "".join(c for t, c in items if t > limit))
                if final:
                    self._flush_req = False
                    self._flush_done.set()

    SPEC_LOW_HZ = 200.0
    SPEC_HIGH_HZ = 1400.0
    SPEC_BINS = 96

    def _make_spectrum(self, sm, n):
        """Audio-note meter data: the smoothed power spectrum between
        SPEC_LOW_HZ and SPEC_HIGH_HZ folded into SPEC_BINS bins scaled 0..1
        (square-root scale, so a weak tone still shows).  Positive and
        negative frequencies are merged because DAX Audio's tone can sit on
        either side of zero."""
        try:
            df = self.sr / n
            i0 = int(self.SPEC_LOW_HZ / df)
            i1 = int(self.SPEC_HIGH_HZ / df)
            idx = np.arange(i0, i1)
            a = np.maximum(sm[idx % n], sm[(-idx) % n])
            per = len(a) // self.SPEC_BINS
            if per < 1:
                return
            b = a[:per * self.SPEC_BINS].reshape(self.SPEC_BINS, per).max(axis=1)
            top = float(b.max())
            if top <= 0:
                return
            v = np.sqrt(b / top)
            self.last_spectrum = (self.SPEC_LOW_HZ, self.SPEC_HIGH_HZ,
                                  [round(float(x), 3) for x in v])
            self.spec_seq += 1
        except Exception:  # noqa: BLE001 -- cosmetic only
            pass

    def _find_tone(self, z):
        """Returns (signed tone Hz, contrast) or (None, 0).  `contrast` is the
        tone's smoothed power over the median power just beside it: about
        1.2-1.5 for pure noise of any bandwidth, 3.5+ even for a 3 s burst of
        CW at -5 dB SNR (measured), so a threshold of 2 cleanly separates
        "there is a signal in this window" from "just receiver noise"."""
        n = len(z)
        spec = np.abs(np.fft.fft(z.astype(np.complex128))) ** 2
        freqs = np.fft.fftfreq(n, 1.0 / self.sr)
        width = max(1, int(round(10.0 / (self.sr / n))))
        sm = np.convolve(spec, np.ones(width) / width, mode="same")
        self._make_spectrum(sm, n)
        ok = (np.abs(freqs) > 60.0) & (np.abs(freqs) < min(4000.0, self.sr / 2 - 100))
        if not np.any(ok) or sm[ok].max() <= 0:
            return None, 0.0
        k = int(np.argmax(np.where(ok, sm, 0.0)))
        f0 = float(freqs[k])
        near = (np.abs(freqs - f0) > 30.0) & (np.abs(freqs - f0) < 150.0)
        base = float(np.median(sm[near])) if np.any(near) else 0.0
        return f0, float(sm[k] / max(base, 1e-30))

    def _decode_snapshot(self, snap, snap_start):
        from scipy.signal import hilbert, resample_poly
        z = snap
        if not np.any(z.imag):                    # real audio (file playback)
            z = hilbert(z.real.astype(np.float64)).astype(np.complex64)
        f0, contrast = self._find_tone(z)
        if f0 is None or contrast < self.min_contrast:
            self.last_tone_hz = None      # nothing but noise in this window
            self.last_wpm = None
            self._wpm_hist = []
            return []
        self.last_tone_hz = f0
        idx = np.arange(len(z), dtype=np.float64)
        w = z * np.exp(2j * np.pi * (self.target_hz - f0) * idx / self.sr).astype(np.complex64)
        y = resample_poly(w, self._up, self._down).real.astype(np.float32)
        peak = float(np.percentile(np.abs(y), 99.5))
        if peak < 1e-12:
            return []
        y = y * (0.83 / peak)
        t0 = snap_start / self.sr
        items = [(t0 + t, ch) for t, ch in self.decoder.decode_timed(y)]
        self._update_wpm(items, t0 + len(snap) / self.sr)
        return items

    def _update_wpm(self, items, end_t):
        """Speed from the newest ~12 s of text, median of the last 3 estimates so
        one odd window cannot make the readout jump."""
        w = estimate_wpm([(t, c) for t, c in items if t > end_t - 12.0])
        if w is None:
            self._wpm_misses += 1
            if self._wpm_misses >= 4:           # a stretch with too little text
                self._wpm_hist = []
                self.last_wpm = None
            return
        self._wpm_misses = 0
        self._wpm_hist = (self._wpm_hist + [w])[-3:]
        self.last_wpm = float(np.median(self._wpm_hist))

    def _commit(self, items, limit):
        """Stitch this window's characters onto what is already settled.

        Plain "only keep characters newer than the last one" is NOT enough:
        a CTC spike for the same letter can land 0.1-0.3 s later in the next
        window, which duplicated letters ("TU U", "IS S") in the first test.
        So the recent committed tail is aligned against this window's text
        by content (difflib), and only characters beyond the matched part
        are added; time is used only as a fallback / sanity check.
        """
        from difflib import SequenceMatcher
        cand = [(t, c) for t, c in items if t <= limit]
        if not cand:
            return
        if not self._committed:
            new = cand
        else:
            lastt = self._committed[-1][0]
            tail = [(t, c) for t, c in self._committed if t >= lastt - 4.0]
            A = "".join(c for _t, c in tail)
            win = [(t, c) for t, c in cand if t >= lastt - 4.5]
            B = "".join(c for _t, c in win)
            start = None
            if win:
                blocks = [b for b in SequenceMatcher(None, A, B, autojunk=False)
                          .get_matching_blocks() if b.size > 0]
                if blocks:
                    i, j, n = blocks[-1]
                    near_end = (i + n) >= len(A) - 3
                    same_time = abs(win[j + n - 1][0] - tail[i + n - 1][0]) < 1.0
                    if near_end and same_time:
                        start = j + n
            if start is None:
                start = 0
            # anything still timed at/before the last settled character is
            # the same material again (a substitution disagreement), skip it
            while start < len(win) and win[start][0] <= lastt + 0.05:
                start += 1
            new = win[start:]
        for t, ch in new:
            if ch == " " and self._last_char in (None, " "):
                continue
            self._pending.append(ch)
            self._committed.append((t, ch))
            self._last_char = ch
        if len(self._committed) > 120:
            self._committed = self._committed[-60:]
