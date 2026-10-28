"""
DAX AUDIO receiver -- the new foundation for CW Copy mode, replacing the
DAX IQ approach after real-hardware testing showed the fundamental
problem with it: decoding a whole 10 kHz IQ passband requires guessing
which of possibly several real CW signals in that passband is the one
the operator actually wants, and that guess can be wrong on a busy band
no matter how the scoring/lock-on logic is tuned (see channelizer.py's
find_cw_pitch_offset docstring for the full story). DAX Audio sidesteps
that problem entirely: it's the radio's OWN already-demodulated,
already-filtered single-channel output for one slice -- whatever the
operator has tuned the dial and the receiver's own CW filter to. There's
essentially only ever one real signal in it, by construction, and it
follows the slice live (retuning the dial changes what comes out of DAX
Audio immediately, no separate "recenter the passband" step needed) --
which is also how CWget (Ken's reference point for what he wants this to
do) works: it uses DAX Audio, not DAX IQ.

Command syntax -- CONFIRMED against FlexRadio's own published wiki
(TCPIP-dax and TCPIP-stream pages, see flex_api_client.py):
  - Assign: "dax audio set <channel> slice=<slice> tx=0"
  - Create:  "stream create type=dax_rx dax_channel=<num>"

Payload format -- CONFIRMED, 2026-09-02, against a real captured DAX
Audio recording (Ken's own, via capture_audio.py -- see PHASE3.md):
32-bit float, interleaved pairs, BIG-endian -- notably the *opposite*
endianness from DAX IQ's own confirmed little-endian, so this was not a
safe assumption to carry over; had to be pinned down from real bytes
again, the same "try several candidates, see which produces sane values"
methodology that settled DAX IQ's format. The one pre-capture lead found
(a FlexRadio staff forum reply describing FlexLib's higher-level
CreateAudioStream wrapper as "2 channels of 32-bit floats ... at 24kHz")
got the sample type and rate right but not the byte order, and turned
out to describe something subtly different from what the raw pairs
actually are:

They are NOT simple stereo (the same mono content sent twice, or two
independent audio channels) -- decoding real captured bytes that way
gave two channels with IDENTICAL frequency content (an FFT of each shows
the same dominant CW tone, confirming both carry the same signal) but
essentially ZERO linear correlation sample-to-sample, which is the
signature of a quadrature (I/Q) pair, not stereo: two real audio
channels carrying the same tone would be strongly correlated, but sine
and cosine components of the very same tone are not, despite describing
the same thing. Treating the pair as complex (left + j*right) and taking
its magnitude confirmed this decisively: that gives a textbook-clean CW
on/off keying envelope (near-zero during key-up, a consistent strong
plateau during key-down, sharp transitions) directly, with NO further
demodulation, filtering, or pitch-searching needed -- because the
magnitude of a complex sinusoid is just its amplitude, independent of
its frequency. That property is what makes DAX Audio genuinely simpler
than the old DAX IQ approach: there's no offset to search for at all
here, unlike find_cw_pitch_offset()'s whole reason for existing.

decode_audio_float32_stereo_be() below implements this confirmed-correct
decode (despite the name, the "stereo" pair is what it's returned to it
as, but the RETURN VALUE is the complex I/Q pair itself, not an averaged
mono signal -- see its docstring). The other candidates are kept only as
historical/diagnostic reference (see diagnose_payload()) -- capture_audio.py
still exists for re-diagnosing if a different radio/firmware ever
produces something else.
"""

import socket
import struct
import threading
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np


# VITA-49 packet header parsing (the header layout is the same for every VITA-49 stream).
@dataclass
class VitaHeader:
    pkt_type: int
    class_id_present: bool
    trailer_present: bool
    tsi: int
    tsf: int
    pkt_count: int
    pkt_size_words: int
    stream_id: Optional[int]
    class_oui: Optional[int]
    info_class_code: Optional[int]
    packet_class_code: Optional[int]
    int_timestamp: Optional[int]
    frac_timestamp: Optional[int]
    header_bytes: int  # total bytes consumed by header+stream_id+class_id+timestamps


def parse_vita_header(data):
    """Parse a VITA-49.0 header per the standard's fixed field layout.
    Returns (VitaHeader, payload_bytes) -- payload_bytes has any trailer
    already excluded if trailer_present (last 4 bytes dropped).
    """
    if len(data) < 4:
        raise ValueError("packet too short for a VITA header word")

    word0 = struct.unpack_from(">I", data, 0)[0]
    pkt_type = (word0 >> 28) & 0xF
    class_id_present = bool((word0 >> 27) & 0x1)
    trailer_present = bool((word0 >> 26) & 0x1)
    tsi = (word0 >> 22) & 0x3
    tsf = (word0 >> 20) & 0x3
    pkt_count = (word0 >> 16) & 0xF
    pkt_size_words = word0 & 0xFFFF

    offset = 4
    stream_id = None
    # Packet types 1, 3, 5, 7 carry a Stream ID word (odd type = "with stream ID")
    if pkt_type in (1, 3, 5, 7):
        stream_id = struct.unpack_from(">I", data, offset)[0]
        offset += 4

    class_oui = info_class_code = packet_class_code = None
    if class_id_present:
        class_word0, class_word1 = struct.unpack_from(">II", data, offset)
        class_oui = class_word0 & 0xFFFFFF
        info_class_code = (class_word1 >> 16) & 0xFFFF
        packet_class_code = class_word1 & 0xFFFF
        offset += 8

    int_timestamp = None
    if tsi != 0:
        int_timestamp = struct.unpack_from(">I", data, offset)[0]
        offset += 4

    frac_timestamp = None
    if tsf != 0:
        frac_timestamp = struct.unpack_from(">Q", data, offset)[0]
        offset += 8

    header = VitaHeader(
        pkt_type=pkt_type, class_id_present=class_id_present,
        trailer_present=trailer_present, tsi=tsi, tsf=tsf, pkt_count=pkt_count,
        pkt_size_words=pkt_size_words, stream_id=stream_id, class_oui=class_oui,
        info_class_code=info_class_code, packet_class_code=packet_class_code,
        int_timestamp=int_timestamp, frac_timestamp=frac_timestamp,
        header_bytes=offset,
    )

    total_bytes = pkt_size_words * 4
    payload_end = min(total_bytes, len(data))
    if trailer_present:
        payload_end -= 4
    payload = data[offset:payload_end]
    return header, payload



DEFAULT_DAX_AUDIO_UDP_PORT = 4991  # same port DAX IQ uses; streams are told apart by Stream ID


def decode_audio_float32_stereo_be(payload_bytes):
    """CONFIRMED CORRECT (2026-09-02, against a real captured DAX Audio
    recording -- see module docstring and PHASE3.md): interleaved 32-bit
    float pairs, BIG-endian. Returns them as a single complex128 array
    (left + j*right) -- NOT averaged to mono -- because the pair is a
    quadrature (I/Q) representation of one demodulated tone, not stereo;
    np.abs() on the result gives the AM envelope directly. This is the
    default decoder in live_run.py/capture_audio.py's Copy-mode path.
    """
    samples = np.frombuffer(payload_bytes, dtype=">f4")
    if len(samples) % 2:
        samples = samples[:-1]
    left = samples[0::2].astype(np.float64)
    right = samples[1::2].astype(np.float64)
    return left + 1j * right


def decode_audio_float32_stereo_le(payload_bytes):
    """Reference candidate only, kept for diagnose_payload() comparison --
    same layout as the confirmed decoder but little-endian (DAX IQ's own
    byte order, which did NOT carry over to DAX Audio). Real captured
    data decodes under this as wildly out-of-range garbage (~1e35+),
    confirming it's wrong. Do not use this for real operation.
    """
    samples = np.frombuffer(payload_bytes, dtype="<f4")
    if len(samples) % 2:
        samples = samples[:-1]
    left = samples[0::2].astype(np.float64)
    right = samples[1::2].astype(np.float64)
    return left + 1j * right


def decode_audio_int16_mono_be(payload_bytes):
    """Reference candidate only: single-channel 16-bit signed int,
    big-endian. Real captured data decodes under this as bounded-looking
    but structureless values (int16 is trivially bounded to [-1, 1)
    regardless of what the underlying bytes actually mean, so
    "boundedness" alone doesn't confirm anything here) -- confirmed wrong
    by the float32 decode's much cleaner, tone-bearing result instead.
    """
    samples = np.frombuffer(payload_bytes, dtype=">i2")
    return samples.astype(np.float64) / 32768.0


def decode_audio_int16_stereo_be(payload_bytes):
    """Reference candidate only: interleaved 16-bit signed int pairs,
    big-endian, as a complex array like the confirmed decoder. Same
    caveat as decode_audio_int16_mono_be() -- ruled out, not confirmed.
    """
    samples = np.frombuffer(payload_bytes, dtype=">i2")
    if len(samples) % 2:
        samples = samples[:-1]
    left = samples[0::2].astype(np.float64) / 32768.0
    right = samples[1::2].astype(np.float64) / 32768.0
    return left + 1j * right


CANDIDATE_DECODERS = {
    "float32_stereo_be (confirmed)": decode_audio_float32_stereo_be,
    "float32_stereo_le": decode_audio_float32_stereo_le,
    "int16_mono_be": decode_audio_int16_mono_be,
    "int16_stereo_be": decode_audio_int16_stereo_be,
}


def diagnose_payload(payload_bytes):
    """Print stats for each candidate decode side by side -- same
    approach vita49_iq.py's diagnose_payload() used to settle DAX IQ's
    format against a real captured packet. A correct decode of real CW
    audio (background noise, maybe a tone) should show bounded, sane
    magnitude with no huge outliers or NaN/Inf; a wrong candidate
    typically produces garbage (near-zero subnormals, wild spikes, or an
    odd/impossible sample count).
    """
    for name, fn in CANDIDATE_DECODERS.items():
        try:
            audio = fn(payload_bytes)
            print(f"  candidate {name:20s}: n={len(audio):5d}  "
                  f"mean|x|={np.abs(audio).mean():.6g}  max|x|={np.abs(audio).max():.6g}  "
                  f"has_nan_or_inf={bool(np.any(~np.isfinite(audio)))}")
        except Exception as e:
            print(f"  candidate {name:20s}: decode failed -- {e}")


class VitaAudioReceiver:
    """Receives DAX Audio over UDP and yields decoded complex128 chunks
    (the confirmed I/Q pair, see module docstring) -- same chunks()
    interface as vita49_iq.VitaIQReceiver, so it's a drop-in source for
    CWCopyStreamer(source="audio") in cw_copy_streamer.py.

    stream_id: filters to just this DAX Audio stream (get it from the
    "stream create type=dax_rx ..." response, same pattern as DAX IQ).
    decoder: one of CANDIDATE_DECODERS' functions -- defaults to the
    confirmed-correct float32 BE decode; only override for diagnosing a
    different radio/firmware that might not match (see capture_audio.py).
    """

    def __init__(self, bind_host="0.0.0.0", bind_port=DEFAULT_DAX_AUDIO_UDP_PORT,
                 stream_id=None, decoder=decode_audio_float32_stereo_be, max_queue=2000):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.stream_id = stream_id
        self.decoder = decoder
        self._queue = queue.Queue(maxsize=max_queue)
        self._sock = None
        self._thread = None
        self._running = False
        self.packets_seen = 0
        self.packets_decoded = 0
        self.parse_errors = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.bind((self.bind_host, self.bind_port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    def _recv_loop(self):
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            self.packets_seen += 1
            try:
                header, payload = parse_vita_header(data)
            except (ValueError, struct.error):
                self.parse_errors += 1
                continue
            if self.stream_id is not None and header.stream_id != self.stream_id:
                continue
            try:
                audio = self.decoder(payload)
            except Exception:
                self.parse_errors += 1
                continue
            self.packets_decoded += 1
            try:
                self._queue.put_nowait(audio)
            except queue.Full:
                pass  # consumer isn't keeping up; drop rather than block the receiver

    def chunks(self, timeout=1.0):
        while self._running:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                return

    def drain_queued(self):
        """Yield whatever's already sitting in the queue RIGHT NOW,
        non-blocking, and without gating on self._running the way
        chunks() does -- for pulling out already-received packets during
        shutdown. Only safe to rely on for "everything, and nothing
        more" once stop() has already been called, so nothing new can
        be added underneath it; used before that, it just yields
        whatever happens to have been queued by that instant, which is
        fine for other uses but not guaranteed complete.

        chunks() itself is the wrong tool for this: it stops yielding
        entirely once self._running is False (so already-queued packets
        would be silently abandoned right after stop()), and even
        before that, its timeout-based wait for the NEXT item never
        actually ends against a genuinely continuous live stream --
        something is always arriving within the timeout, so the "loop
        until the queue's been empty for a bit" pattern never sees that
        happen and hangs forever. Confirmed against a real radio feed:
        Copy mode's Stop button would say "Stopping..." indefinitely
        because of exactly this, only ever showing up against a truly
        continuous source rather than the finite synthetic test streams
        that first justified draining anything here at all.
        """
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return
