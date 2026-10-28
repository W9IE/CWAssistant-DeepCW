"""
Sound-card audio input -- lets the decoder listen to ANY radio that shows up
as a Windows recording device (ICOM, Yaesu, Kenwood, Elecraft ... via the
radio's USB audio, a sound-card interface, or a virtual audio cable), instead
of FlexRadio's DAX Audio.

The audio is plain real-valued mono samples.  DeepCWStreamer already handles
real audio (it builds the analytic signal itself), so nothing else changes.

Uses the `sounddevice` package (PortAudio).  It is imported lazily so the rest
of the program -- and the FlexRadio source -- work without it installed.
Install with:  py -m pip install sounddevice
"""

import queue
import threading

import numpy as np


class AudioInputError(Exception):
    """Plain-English problem with the sound card (missing package, device
    busy, device not found ...)."""


# Tests can put a fake `sounddevice`-like object here; see test_harness_gui_deepcw.py.
_SD_OVERRIDE = None


def _sd():
    if _SD_OVERRIDE is not None:
        return _SD_OVERRIDE
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001 -- ImportError, or PortAudio missing
        raise AudioInputError(
            "The sound-card source needs the 'sounddevice' package.\n"
            "Install it with:  py -m pip install sounddevice\n"
            f"(detail: {e})")
    return sd


def list_input_devices():
    """[(label, device_index)] for every recording device in the default audio
    system, default device first.  Raises AudioInputError if sounddevice is
    not available."""
    sd = _sd()
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if sd.default.device is not None else None
    except Exception as e:  # noqa: BLE001
        raise AudioInputError(f"Could not list the recording devices: {e}")
    try:
        hostapi = sd.default.hostapi
    except Exception:  # noqa: BLE001
        hostapi = None
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if hostapi is not None and d.get("hostapi") not in (None, hostapi):
            continue          # Windows lists each device once per audio system; keep one
        out.append((str(d["name"]).strip(), i))
    out.sort(key=lambda t: (t[1] != default_in, t[0].lower()))
    # make labels unique if two devices share a name
    seen = {}
    labelled = []
    for name, idx in out:
        seen[name] = seen.get(name, 0) + 1
        labelled.append((name if seen[name] == 1 else f"{name} ({seen[name]})", idx))
    return labelled


class SoundCardSource:
    """Open one recording device and hand out mono float32 chunks.

    Same small interface the Flex receiver has: start(), stop(), chunks(timeout),
    drain_queued().  `sample_rate` is the rate actually obtained."""

    PREFERRED_RATES = (48000, 44100)

    def __init__(self, device_index, sample_rate=None):
        self.device_index = device_index
        self.sample_rate = float(sample_rate) if sample_rate else None
        self.overflows = 0
        self.chunks_seen = 0
        self._q = queue.Queue(maxsize=400)       # about 40 s of 0.1 s blocks
        self._stream = None
        self._lock = threading.Lock()

    # -- audio callback (runs on PortAudio's thread) ------------------------
    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            self.overflows += 1
        x = np.asarray(indata, dtype=np.float32)
        mono = x.mean(axis=1) if x.ndim == 2 and x.shape[1] > 1 else x.reshape(-1)
        try:
            self._q.put_nowait(np.array(mono, dtype=np.float32, copy=True))
        except queue.Full:
            self.overflows += 1

    def start(self):
        sd = _sd()
        try:
            info = sd.query_devices(self.device_index)
        except Exception as e:  # noqa: BLE001
            raise AudioInputError(f"Recording device not found: {e}")
        channels = 2 if info.get("max_input_channels", 1) >= 2 else 1
        rates = []
        if self.sample_rate:
            rates.append(int(self.sample_rate))
        rates += [r for r in self.PREFERRED_RATES if r not in rates]
        default_rate = int(info.get("default_samplerate", 0) or 0)
        if default_rate and default_rate not in rates:
            rates.append(default_rate)
        last_err = None
        for rate in rates:
            try:
                stream = sd.InputStream(
                    device=self.device_index, channels=channels, samplerate=rate,
                    dtype="float32", blocksize=int(rate * 0.1), callback=self._callback)
                stream.start()
            except Exception as e:  # noqa: BLE001 -- try the next rate
                last_err = e
                continue
            self._stream = stream
            self.sample_rate = float(rate)
            return
        raise AudioInputError(
            f"Could not open the recording device ({last_err}).\n"
            "Check that no other program is using it, that it is not disabled in "
            "Windows Sound settings, and that this program is allowed to use the "
            "microphone (Windows Settings > Privacy > Microphone).")

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 -- shutting down anyway
                pass

    def chunks(self, timeout=1.0):
        """Yield what has arrived; waits up to `timeout` for the first block,
        then returns once the queue is empty."""
        try:
            first = self._q.get(timeout=timeout)
        except queue.Empty:
            return
        self.chunks_seen += 1
        yield first
        while True:
            try:
                blk = self._q.get_nowait()
            except queue.Empty:
                return
            self.chunks_seen += 1
            yield blk

    def drain_queued(self):
        while True:
            try:
                yield self._q.get_nowait()
            except queue.Empty:
                return
