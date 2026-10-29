"""
CWAssistant -- DeepCW Edition: decode-only run logic.

Two audio sources feed the same DeepCW neural decoder:

  * FlexRadio  -- DAX Audio from one slice over the network (VITA-49 UDP).
                  The slice's frequency is never changed, only listened to.
  * Sound card -- any Windows recording device: the USB audio of an ICOM /
                  Yaesu / Kenwood / Elecraft radio, a sound-card interface, or
                  a virtual audio cable.

Nothing here ever transmits or retunes a radio.

Both runners follow the same contract so gui_run.py (the window) and the
console entry point below can drive them identically:

    run_flex_audio(args, log=print, should_stop=None)
    run_soundcard_audio(args, log=print, should_stop=None)

log is called with each status line; should_stop is polled between audio
chunks.  Lines starting "DEEPCW:", "DEEPCWT:", "TONE:", "WPM:", "SPEC:" and
"LEVEL:" are machine messages for the window (see _ToneReporter and
_feed_deepcw); everything else is plain text for the user.

Console usage:
    python live_run.py --source flex --host 192.168.1.104 --slice 0
    python live_run.py --source soundcard --list-devices
    python live_run.py --source soundcard --device 3
"""

import argparse
import sys
import threading
import time

import numpy as np

from flex_api_client import FlexAPIClient, FlexAPIError
from vita49_audio import VitaAudioReceiver, decode_audio_float32_stereo_be

# Bumped every time the actual behavior changes (not just comments/docs).
# Shown in the console banner and in the window's title/credit line
# (gui_run.py imports this) so "which build is this" is never a guess -- a
# stale .exe looks identical to a current one until it is rebuilt.
# Format: YYYY-MM-DD.N
APP_VERSION = "2026-10-29.3-deepcw"


class DetectorError(Exception):
    """A setup failure that should stop this run cleanly.  The console entry
    point prints the message; gui_run.py shows it in its own window."""
    pass


# ---------------------------------------------------------------------------
# DeepCW plumbing shared by both sources
# ---------------------------------------------------------------------------

def _start_deepcw(sample_rate, log, is_complex):
    """Start the DeepCW streamer or raise DetectorError with a plain-English
    reason (missing onnxruntime, missing model folder...)."""
    try:
        import deepcw_backend
        streamer = deepcw_backend.DeepCWStreamer(sample_rate)
    except Exception as e:  # noqa: BLE001 -- turn anything into a readable error
        raise DetectorError(
            f"DeepCW could not start: {e}\n"
            "It needs the 'onnxruntime' package (py -m pip install onnxruntime) and the "
            "'deepcw_model' folder (model.onnx + model.onnx.json) next to this program.")
    log("DeepCW decoder loaded -- its text appears in the big pane below; the newest few "
        "words show dimmer first and firm up a moment later.")
    return streamer


def _feed_deepcw(deepcw, chunk, log):
    if deepcw is None:
        return
    try:
        text = deepcw.push(chunk)
        if text:
            log(f"DEEPCW: {text}")
        tent = getattr(deepcw, "tentative", "")
        if tent != getattr(deepcw, "_sent_tentative", ""):
            deepcw._sent_tentative = tent
            log(f"DEEPCWT: {tent}")     # live preview of the newest, still-changing words
        err = deepcw.take_error()
        if err:
            log(f"DeepCW error (continuing): {err}")
    except Exception as e:  # noqa: BLE001 -- keep the run alive
        log(f"DeepCW error (continuing): {e}")


def _finish_deepcw(deepcw, log, flush=True):
    if deepcw is None:
        return
    try:
        if flush:
            tail = deepcw.flush()
            if tail:
                log(f"DEEPCW: {tail}")
            log("DEEPCWT: ")
    finally:
        deepcw.close()


class _ToneReporter:
    """Sends SPEC / WPM / TONE lines to the window whenever DeepCW's view of
    the signal changes -- cheap, and it answers "is my pitch where the decoder
    expects it" and "how fast is the sender"."""

    def __init__(self, deepcw, log):
        self.deepcw = deepcw
        self.log = log
        self.last = "unset"
        self.last_spec_seq = -1
        self.last_wpm = "unset"

    def update(self):
        seq = getattr(self.deepcw, "spec_seq", 0)
        spec = getattr(self.deepcw, "last_spectrum", None)
        if spec is not None and seq != self.last_spec_seq:
            self.last_spec_seq = seq
            low, high, bins = spec
            hz = getattr(self.deepcw, "last_tone_hz", None)
            hz_s = "NONE" if hz is None else f"{abs(hz):.1f}"
            self.log(f"SPEC: {hz_s}|{low:.0f}|{high:.0f}|" + ",".join(f"{b:.3f}" for b in bins))
        w = getattr(self.deepcw, "last_wpm", None)
        wkey = None if w is None else int(round(w))
        if wkey != self.last_wpm:
            self.last_wpm = wkey
            self.log("WPM: " + ("" if wkey is None else str(wkey)))
        hz = getattr(self.deepcw, "last_tone_hz", None)
        key = None if hz is None else int(round(abs(hz) / 10.0)) * 10
        if key != self.last:
            self.last = key
            self.log("TONE: " + ("" if key is None else f"{key} Hz"))


class _LevelReporter:
    """Sends "LEVEL: 0.123" (peak of the audio since the last report, 0..1,
    about 5 times a second) so the window can show an input-level bar -- the
    quickest way to see whether the radio's audio is arriving and is not
    clipped or far too quiet."""

    def __init__(self, log, interval=0.2):
        self.log = log
        self.interval = interval
        self.peak = 0.0
        self.last_t = time.time()

    def push(self, chunk):
        c = np.asarray(chunk)
        if c.size:
            p = float(np.max(np.abs(c)))
            if p > self.peak:
                self.peak = p
        now = time.time()
        if now - self.last_t >= self.interval:
            self.last_t = now
            self.log(f"LEVEL: {min(self.peak, 1.0):.3f}")
            self.peak = 0.0


def _run_decode_loop(source, deepcw, log, should_stop, quiet_hint):
    """Pull chunks from `source` (anything with chunks(timeout) and
    drain_queued()), feed DeepCW, report tone / speed / level.  Returns after
    the source has been drained and DeepCW flushed.  `source.stop()` must
    be safe to call twice."""
    tone = _ToneReporter(deepcw, log)
    level = _LevelReporter(log)
    last_note = time.time()
    got_ever = False
    while not should_stop():
        got_this_round = False
        for chunk in source.chunks(timeout=2.0):
            got_this_round = got_ever = True
            level.push(chunk)
            _feed_deepcw(deepcw, chunk, log)
            tone.update()
            if should_stop():
                break
        if should_stop():
            break
        if not got_this_round and time.time() - last_note > 5.0:
            last_note = time.time()
            log("  ... no audio received yet -- " + quiet_hint() if not got_ever
                else "  ... audio stopped arriving -- " + quiet_hint())
    # Stop the source BEFORE draining so what is left is bounded (draining a
    # live feed first never finishes -- this hung Stop once).
    source.stop()
    for chunk in source.drain_queued():
        _feed_deepcw(deepcw, chunk, log)
    tone.update()
    log("LEVEL: 0.000")


# ---------------------------------------------------------------------------
# Source 1: FlexRadio DAX Audio
# ---------------------------------------------------------------------------

def _assign_dax_channel(client, slice_id, wanted_channel, log, wait_sec=1.5):
    """Make sure the chosen slice -- and not some other slice -- is what the DAX
    channel carries, and return the channel number to create the stream on.

    Why this exists: a DAX channel is tied to ONE slice by the slice's own "dax"
    setting.  The old code only sent "dax audio set <ch> slice=<n>", which the radio
    can accept without actually moving the channel -- so if SmartSDR already had
    that channel on Slice A, the stream kept carrying Slice A whatever slice was
    picked.  Now:
      1. ask the radio which slices are open and which DAX channel each has;
      2. if the chosen slice already has a channel, use it; otherwise give it one
         that no other slice is using (never taking a channel away from another
         slice, so SmartSDR / other DAX programs keep working);
      3. set it with the slice's own "dax=" setting AND "dax audio set";
      4. read back what the radio now says and log it, with a warning on mismatch.
    """
    letter = chr(ord("A") + slice_id) if 0 <= slice_id <= 7 else "?"
    try:
        client.subscribe_slices()
    except FlexAPIError as e:
        log(f"  (could not read slice status from the radio: {e})")

    # give the radio a moment to send its slice status
    deadline = time.time() + wait_sec
    while time.time() < deadline and slice_id not in client.slice_dax_map():
        time.sleep(0.05)
    before = client.slice_dax_map()

    def describe(m):
        if not m:
            return "none seen"
        return ", ".join(f"Slice {chr(ord('A') + n)}=" + (f"DAX {d}" if d else "no DAX")
                         for n, d in sorted(m.items()))

    log(f"  radio reports: {describe(before)}")
    if before and slice_id not in before:
        log(f"  WARNING: Slice {letter} does not appear to be open on the radio -- open it in "
            "SmartSDR first, or pick the slice that is open.")

    others = {d for n, d in before.items() if n != slice_id and d > 0}
    own = before.get(slice_id, 0)
    if own > 0:
        channel = own
        log(f"  Slice {letter} already uses DAX channel {channel} -- using that")
    else:
        channel = wanted_channel
        if channel in others:
            free = [c for c in range(1, 9) if c not in others]
            if not free:
                raise DetectorError("All 8 DAX channels are already in use by other slices.")
            channel = free[0]
            log(f"  DAX channel {wanted_channel} is already used by another slice -- "
                f"using free channel {channel} for Slice {letter} instead")
        client.assign_slice_dax(slice_id, channel)          # "slice s <n> dax=<ch>"
    client.configure_dax_audio(channel, slice_id)           # "dax audio set <ch> slice=<n> tx=0"

    # read back
    deadline = time.time() + wait_sec
    while time.time() < deadline and client.slice_dax_map().get(slice_id) != channel:
        time.sleep(0.05)
    after = client.slice_dax_map()
    if not after:
        log(f"  Slice {letter} -> DAX channel {channel} (could not read the setting back from the radio)")
    elif after.get(slice_id) == channel:
        log(f"  confirmed by the radio: Slice {letter} is on DAX channel {channel}")
    else:
        log(f"  WARNING: the radio still reports {describe(after)} -- the audio may not be from "
            f"Slice {letter}. In SmartSDR's DAX panel, set Slice {letter}'s DAX channel to {channel}.")
    return channel

def run_flex_audio(args, log=print, should_stop=None):
    """DeepCW decode of the live DAX AUDIO stream for one slice.

    args needs: host, api_port, slice, dax_channel, udp_port, stream_id
    (hex string or None)."""
    if should_stop is None:
        should_stop = lambda: False

    sample_rate = 24000.0  # confirmed against real captured DAX Audio -- see vita49_audio.py
    log(f"CWAssistant DeepCW Edition -- version {APP_VERSION}")
    deepcw = _start_deepcw(sample_rate, log, is_complex=True)   # fail fast, before touching the radio
    slice_letter = chr(ord("A") + args.slice) if 0 <= args.slice <= 7 else "?"

    log(f"Connecting to {args.host}:{args.api_port} ...")
    try:
        client = FlexAPIClient(args.host, args.api_port)
        client.connect()
    except Exception as e:  # noqa: BLE001
        _finish_deepcw(deepcw, log, flush=False)
        raise DetectorError(f"Could not connect to the radio at {args.host}:{args.api_port}: {e}")
    log(f"  connected -- radio version {client.version}, handle {client.handle}")
    log(f"  using slice {args.slice} (Slice {slice_letter}) -- this slice's own frequency is "
        "never changed, only decoded")

    log(f"  registering UDP port {args.udp_port} with the radio for this client ...")
    try:
        client.register_udp_port(args.udp_port)
    except FlexAPIError as e:
        client.close()
        _finish_deepcw(deepcw, log, flush=False)
        raise DetectorError(
            f"FAILED to register UDP port {args.udp_port}: {e}\n"
            "This usually means another program on this PC already claimed "
            "that port (another instance of this program, or another piece "
            "of FlexRadio software). Close whatever else might be running, "
            "or pick a different port, and try again.")

    log(f"Setting up DAX Audio for slice {args.slice} (Slice {slice_letter}) "
        "(this does NOT retune anything -- copying whatever this slice is "
        "already tuned to and filtered for, the same way you'd normally "
        "listen to it) ...")
    try:
        dax_channel = _assign_dax_channel(client, args.slice, args.dax_channel, log)
        status, message = client.create_dax_audio_stream(dax_channel)
    except DetectorError:
        client.close()
        _finish_deepcw(deepcw, log, flush=False)
        raise
    except FlexAPIError as e:
        client.close()
        _finish_deepcw(deepcw, log, flush=False)
        raise DetectorError(
            f"The radio refused the DAX Audio setup for Slice {slice_letter}: {e}\n"
            f"Check that Slice {slice_letter} is open in SmartSDR.")
    log(f"  stream create response: status={status} message={message!r}")
    stream_id = int(args.stream_id, 16) if args.stream_id else None
    if stream_id is None and message and all(c in "0123456789abcdefABCDEF" for c in message.strip()):
        stream_id = int(message.strip(), 16)
        log(f"  auto-detected stream ID from response: 0x{stream_id:08X}")
    elif stream_id is None:
        log("  no stream ID given and none could be parsed from the response above -- "
            "accepting all packets on the UDP port. If more than one DAX stream could "
            "be active on this radio, provide the stream ID explicitly instead.")

    log(f"Listening for DAX Audio on UDP port {args.udp_port} "
        f"(assuming {sample_rate:.0f} Hz complex sample rate) ...")
    receiver = VitaAudioReceiver(bind_port=args.udp_port, stream_id=stream_id,
                                 decoder=decode_audio_float32_stereo_be)
    receiver.start()
    log(f"Decoding CW on Slice {slice_letter} with DeepCW -- following whatever this slice is "
        "tuned to live. Retune the dial any time -- no restart needed.")

    def hint():
        return (f"packets_seen={receiver.packets_seen}, decoded={receiver.packets_decoded}, "
                f"parse_errors={receiver.parse_errors}. Check that this slice has DAX Audio "
                "available and is tuned to a station")

    try:
        _run_decode_loop(receiver, deepcw, log, should_stop, hint)
        _finish_deepcw(deepcw, log)
        deepcw = None
    except KeyboardInterrupt:
        pass
    finally:
        _finish_deepcw(deepcw, log, flush=False)   # no-op if already finished above
        receiver.stop()                            # idempotent
        client.close()
        log("\nStopped.")


# ---------------------------------------------------------------------------
# Source 2: sound card (any transceiver)
# ---------------------------------------------------------------------------

def run_soundcard_audio(args, log=print, should_stop=None):
    """DeepCW decode of a Windows recording device.

    args needs: device (integer index from audio_input.list_input_devices())."""
    import audio_input
    if should_stop is None:
        should_stop = lambda: False

    log(f"CWAssistant DeepCW Edition -- version {APP_VERSION}")
    if args.device is None:
        raise DetectorError("No sound-card input was chosen. Pick one in the Sound card list.")

    source = audio_input.SoundCardSource(args.device)
    try:
        source.start()
    except audio_input.AudioInputError as e:
        raise DetectorError(str(e))
    rate = source.sample_rate
    log(f"Listening to the sound card at {rate:.0f} Hz. DeepCW finds the CW tone itself "
        "(400-1200 Hz) -- just keep the radio's audio level moderate (the level bar "
        "should move but not sit at the far right).")

    try:
        deepcw = _start_deepcw(rate, log, is_complex=False)
    except DetectorError:
        source.stop()
        raise

    def hint():
        return ("check the chosen recording device, and that the radio is sending "
                "audio to it (USB audio / audio interface cable)")

    try:
        _run_decode_loop(source, deepcw, log, should_stop, hint)
        _finish_deepcw(deepcw, log)
        deepcw = None
    except KeyboardInterrupt:
        pass
    finally:
        _finish_deepcw(deepcw, log, flush=False)
        source.stop()
        log("\nStopped.")


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------

def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="CW decoder (DeepCW) for a FlexRadio's DAX Audio or any sound card")
    ap.add_argument("--source", choices=["flex", "soundcard"], default="flex")
    ap.add_argument("--list-devices", action="store_true",
                    help="Print the sound-card inputs and exit.")
    # FlexRadio
    ap.add_argument("--host", default=None, help="Radio's LAN IP, e.g. 192.168.1.104 (flex source)")
    ap.add_argument("--api-port", type=int, default=4992)
    ap.add_argument("--slice", type=int, default=0,
                    help="Numeric slice ID (A=0, B=1, C=2, ...). Default 0 (Slice A).")
    ap.add_argument("--dax-channel", type=int, default=1)
    ap.add_argument("--udp-port", type=int, default=41991,
                    help="UDP port this program registers with the radio for its DAX audio. "
                         "Deliberately NOT the well-known 4991 default, so this can run "
                         "alongside other FlexRadio software. Change it only if 41991 is in use.")
    ap.add_argument("--stream-id", default=None,
                    help="Hex VITA-49 stream ID to filter on, if known (normally auto-detected).")
    # Sound card
    ap.add_argument("--device", type=int, default=None,
                    help="Sound-card input device number (see --list-devices).")
    return ap


def main():
    args = build_arg_parser().parse_args()

    if args.list_devices:
        import audio_input
        try:
            for label, idx in audio_input.list_input_devices():
                print(f"{idx:4d}  {label}")
        except audio_input.AudioInputError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
        return

    try:
        if args.source == "flex":
            if not args.host:
                print("  ERROR: --host is required for the flex source, e.g. --host 192.168.1.104")
                sys.exit(1)
            run_flex_audio(args)
        else:
            run_soundcard_audio(args)
    except DetectorError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
