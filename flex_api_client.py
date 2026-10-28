"""
SmartSDR TCP/IP command-channel client (port 4992).

Handles the text command/response protocol used to control a Flex radio:
connect, read the version/handle prologue, send sequence-numbered
commands, and match up their responses.  This program only needs it to set
up a DAX AUDIO stream for one slice (see vita49_audio.py); it never tunes
the radio and never touches PTT/transmit.

Command/response protocol, per FlexRadio's published API docs
(https://github.com/flexradio/smartsdr-api-docs/wiki/SmartSDR-TCPIP-API):
  - Client sends:  "C<seq>|<command>\\r\\n"
  - Radio replies: "R<seq>|<hex_status>|<message>\\r\\n" (hex_status 0 = OK)
  - On connect, before any of that, the radio sends a "V<version>" line
    and an "H<handle>" line unprompted.
  - No login/auth needed -- commands are accepted as soon as the TCP
    connection is open.

Commands used here (TCPIP-dax / TCPIP-stream pages of the same wiki):
  - Register our UDP port:    "client udpport <port>"
  - Assign DAX audio channel: "dax audio set <channel> slice=<slice> tx=0"
  - Create the audio stream:  "stream create type=dax_rx dax_channel=<channel>"
"""

import socket
import threading
import time


class FlexAPIError(Exception):
    pass


class FlexAPIClient:
    def __init__(self, host, port=4992):
        self.host = host
        self.port = port
        self.sock = None
        self.version = None
        self.handle = None

        self._seq = 0
        self._seq_lock = threading.Lock()
        self._responses = {}          # seq -> (hex_status, message)
        self._response_event = threading.Condition()
        self._status_lines = []       # raw "S..." lines (newest 2000 kept), for diagnostics
        self._status_lock = threading.Lock()
        self._slice_dax = {}          # slice number -> DAX channel the radio says it has (0 = none)

        self._recv_thread = None
        self._running = False
        self._sock_file = None

    # -- connection lifecycle -------------------------------------------------

    def connect(self, timeout=5.0):
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self._sock_file = self.sock.makefile("r", encoding="ascii", errors="replace", newline="\n")

        # Prologue: radio sends "V<version>" then "H<handle>" before anything else.
        deadline = time.time() + timeout
        while self.version is None or self.handle is None:
            if time.time() > deadline:
                raise FlexAPIError("Timed out waiting for radio's V/H prologue")
            line = self._sock_file.readline()
            if not line:
                raise FlexAPIError("Connection closed during prologue")
            line = line.strip()
            if line.startswith("V"):
                self.version = line[1:]
            elif line.startswith("H"):
                self.handle = line[1:]

        self._running = True
        self._recv_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._recv_thread.start()

    def close(self):
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

    def _read_loop(self):
        while self._running:
            try:
                line = self._sock_file.readline()
            except (OSError, ValueError):
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("R"):
                self._handle_response(line)
            elif line.startswith("S"):
                with self._status_lock:
                    self._status_lines.append(line)
                    if len(self._status_lines) > 2000:
                        del self._status_lines[:1000]
                    self._note_slice_status(line)
            # "M..." (message) and anything else: ignored for this project's purposes

    def _note_slice_status(self, line):
        """Keep a running table of "which DAX channel does each slice have" from the
        radio's own slice status lines, e.g.
            S1A2B3C4D|slice 1 RF_frequency=7.030000 ... dax=2 ...
        Status lines after the first one only carry what CHANGED, so this updates
        the table piece by piece.  Caller holds _status_lock."""
        body = line.split("|", 1)[1] if "|" in line else ""
        parts = body.split()
        if len(parts) < 2 or parts[0] != "slice" or not parts[1].isdigit():
            return
        n = int(parts[1])
        kv = {}
        for tok in parts[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        if kv.get("in_use") == "0":
            self._slice_dax.pop(n, None)
            return
        cur = self._slice_dax.get(n, 0)
        if "dax" in kv:
            try:
                cur = int(kv["dax"])
            except ValueError:
                pass
        self._slice_dax[n] = cur

    def slice_dax_map(self):
        """{slice number: DAX channel} as last reported by the radio (0 = none).
        Only slices that exist (are open) appear.  Empty until subscribe_slices()
        has been sent and the radio has answered."""
        with self._status_lock:
            return dict(self._slice_dax)

    def _handle_response(self, line):
        # "R<seq>|<hex_status>|<message>[|debug]"
        body = line[1:]
        parts = body.split("|")
        try:
            seq = int(parts[0])
        except (ValueError, IndexError):
            return
        hex_status = parts[1] if len(parts) > 1 else ""
        message = parts[2] if len(parts) > 2 else ""
        with self._response_event:
            self._responses[seq] = (hex_status, message)
            self._response_event.notify_all()

    # -- sending commands -------------------------------------------------------

    def send_command(self, command, timeout=3.0, wait=True):
        """Send one command. Returns (hex_status, message) if wait=True
        (raises FlexAPIError on timeout or non-zero status), else None.
        """
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        line = f"C{seq}|{command}\r\n"
        self.sock.sendall(line.encode("ascii"))

        if not wait:
            return None

        deadline = time.time() + timeout
        with self._response_event:
            while seq not in self._responses:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise FlexAPIError(f"Timed out waiting for response to: {command!r}")
                self._response_event.wait(timeout=remaining)
            hex_status, message = self._responses.pop(seq)

        if hex_status not in ("0", "00000000"):
            raise FlexAPIError(f"Command {command!r} failed (status {hex_status}): {message}")
        return hex_status, message

    # -- the specific commands this project needs --------------------------------


    def register_udp_port(self, port):
        """Tell the radio which UDP port THIS client wants its data (DAX audio,
        meters, etc.) sent to. Found via a real FlexRadio community thread,
        not the wiki -- without this, the radio has no way to know where a
        raw API client (as opposed to a full GUI client like SmartSDR/
        Maestro) wants its UDP traffic delivered, so it simply never sends
        any. This is the usual reason no audio packets arrive
        even though every DAX setup command reported success.
        """
        return self.send_command(f"client udpport {port}")


    def subscribe_slices(self):
        """Ask the radio to send us slice status ("sub slice all"), so we can see
        which slices are open and which DAX channel each one has."""
        return self.send_command("sub slice all")

    def assign_slice_dax(self, slice_id, dax_channel):
        return self.send_command(f"slice s {slice_id} dax={dax_channel}")




    def configure_dax_audio(self, dax_channel, slice_id):
        """Assign a DAX AUDIO channel to a slice -- confirmed against
        FlexRadio's own published wiki (TCPIP-dax page):
        "dax audio set <channel> slice=<slice> tx=<1|0>". This is a
        DAX Audio is one of two independent DAX subsystems (see
        vita49_audio.py's module docstring). tx=0 always here: this project only ever receives,
        never transmits through DAX.
        """
        return self.send_command(f"dax audio set {dax_channel} slice={slice_id} tx=0")

    def create_dax_audio_stream(self, dax_channel):
        """Confirmed against FlexRadio's own published wiki (TCPIP-stream
        page): "stream create type=dax_rx dax_channel=<num>".
        """
        return self.send_command(f"stream create type=dax_rx dax_channel={dax_channel}")



