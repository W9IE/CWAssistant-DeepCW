# Legal notices

## Copyright and license

Copyright (c) 2026 W9IE.

CWAssistant -- DeepCW Edition is free software: you can redistribute it and/or modify it under
the terms of the **GNU Affero General Public License, version 3**, as published by the Free
Software Foundation. The full text is in the file `LICENSE`.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**;
without even the implied warranty of merchantability or fitness for a particular purpose.
The author is not liable for any damages arising from its use. Use it at your own risk.

### Why AGPL-3.0

The program includes the DeepCW neural-network model (in the `deepcw_model` folder), which is
licensed **AGPL-3.0-only** by its authors (see `deepcw_model/LICENSE`, and the DeepCW project:
https://github.com/e04/deepcw-engine). A program that includes it must be distributed under the
same license. In practice:

- You may use, copy, modify and share this program, including the .exe in Releases.
- If you pass on the .exe (modified or not), you must also make the corresponding source
  available to the people you give it to, under the same license, and keep the copyright and
  license notices intact.
- If you modify it, say so prominently.

The `deepcw_model` folder, with its LICENSE file, must stay with the program.

## Third-party software

The program uses the following. Their licenses are their own; summaries here are for
convenience, and the license text shipped with each package is what counts.

| Component | Used for | License (as I understand it) |
|---|---|---|
| DeepCW model (e04/deepcw-engine) | CW decoding | AGPL-3.0-only |
| ONNX Runtime | runs the model | MIT |
| NumPy | numerical code | BSD-3-Clause |
| SciPy | signal processing, WAV files | BSD-3-Clause |
| sounddevice / PortAudio | sound-card input | MIT / MIT-style |
| Python and Tcl/Tk (tkinter) | the language and the window | PSF License / Tcl-Tk license |
| PyInstaller | builds the .exe | GPL-2.0 with an exception that allows the programs it builds to use any license |

## Trademarks and no affiliation

FlexRadio, SmartSDR, DAX, ICOM, Yaesu, Kenwood and Elecraft are trademarks of their respective
owners. This project is independent and is not affiliated with, sponsored by, or endorsed by
any of them. The names appear only to describe which radios the program can work with.
It talks to FlexRadio equipment through the publicly documented SmartSDR TCP/IP API.

## Amateur radio use

This program only receives. Operating a station is the operator's responsibility; follow the
rules of your licensing authority.

## The unsigned .exe

The .exe is not code-signed, so Windows SmartScreen warns about it. You can instead build it
yourself from this source (see README.md).
