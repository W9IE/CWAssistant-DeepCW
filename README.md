# CWAssistant -- DeepCW Edition

A Windows program that copies Morse code (CW) from your radio and shows the text on screen,
using the open-source **DeepCW** neural-network decoder. It works with:

- **FlexRadio** (6000, 8000 and Aurora series) -- it listens to one slice's DAX audio over your network.
- **Any other transceiver** (ICOM, Yaesu, Kenwood, Elecraft ...) -- it listens to a Windows
  recording device: the radio's USB audio, a sound-card interface, or a virtual audio cable.

It only listens. It never transmits, never touches PTT, and never changes the radio's frequency.

Written by W9IE (with help from Claude).

## What you see

- A big window of decoded text. The dim text at the end is the newest few words, still being
  firmed up. A stand-alone "BT" (the pause sign) is shown as "=".
- The sending speed (WPM) and the audio pitch ("Note") DeepCW is following.
- An input-level bar with a dark green "good level" zone: the bar is green inside it, yellow
  when getting loud, and red (with a warning that stays up for a moment) on overload.
- Start / Stop / Clear buttons, and a blue **Help** button that opens a Quick start window.

DeepCW finds the CW tone itself (roughly 400-1200 Hz), so you do not enter a pitch.

## Easiest way: the .exe

1. Open the **Releases** page of this repository and download `CWAssistantDeepCW.exe`.
2. Double-click it. Windows may show a blue "Windows protected your PC" (SmartScreen) box
   because the program is not code-signed. Click **More info**, then **Run anyway**.
3. Pick your audio source (below) and press **Start**.

The decoder model is packed inside the .exe. Nothing else needs installing.

## Using a FlexRadio

1. Start SmartSDR and open the slice you want to copy (Slice A or B).
2. In the program choose **FlexRadio (DAX Audio)**, type the radio's IP address, pick the slice.
3. Press **Start**. The status box shows what the radio reports, for example
   "confirmed by the radio: Slice B is on DAX channel 2".
4. If Windows Firewall asks, allow the program on your private network.

The slice is chosen when you press Start. To change slices, press Stop, change it, press Start.
If the slice has no DAX channel yet, the program gives it one that no other slice is using.
The program's default network port is 41991 (not the usual 4991), so it can run alongside
other FlexRadio software. The "Show advanced settings" button shows these options.

## Using any other radio (sound card)

1. Make the radio's audio reach a Windows recording device (for an ICOM or similar this is
   usually its USB audio device, often listed as something like "USB Audio CODEC").
2. In the program choose **Sound card**, pick the device in the list (press **Refresh list** if
   it is missing), and press **Start**.
3. Watch the **Input level** bar. Aim for the bar to sit inside the dark green zone on signals.
   Red means too loud and the audio may be distorted -- turn the radio's audio output down.

If nothing appears, check Windows **Settings > Privacy > Microphone** and make sure desktop
apps are allowed to use the microphone.

Sound-card mode has so far been tested with simulated audio only. **Reports from real ICOM / Yaesu / Kenwood setups are very welcome** (open an Issue).

## Running from source (instead of the .exe)

You need Python 3 for Windows (developed on Windows 11 with Python 3.14).

    py -m pip install numpy scipy onnxruntime sounddevice
    py gui_run.py

(`sounddevice` is only needed for the sound-card source.) Keep the `deepcw_model` folder next
to `gui_run.py`.

To build your own .exe:

    py -m pip install pyinstaller
    py -m PyInstaller --onefile --windowed --name CWAssistantDeepCW --clean --collect-all onnxruntime --collect-all sounddevice --add-data "deepcw_model;deepcw_model" gui_run.py

Close the program before rebuilding; Windows cannot replace a running .exe.

## Limits

- Very weak, fading or badly sent signals can still produce errors. It is a decoder, not magic.
- The text is about 2-3 seconds behind the signal, because the network reads a rolling 16-second
  window and firms up the newest words as more audio arrives.
- Decoded text is raw network output, with no dictionary correction.

## License

This program is free software under the **GNU Affero General Public License v3.0** (see
`LICENSE`), because it includes the DeepCW model, which is AGPL-3.0. If you give someone a
copy of the .exe, the AGPL requires the matching source to be available to them -- this
repository is that source. See `LEGAL.md` for the details, third-party notices and
disclaimers.

This project is not affiliated with or endorsed by FlexRadio Systems, ICOM, Yaesu or Kenwood;
those names are used only to say which radios it works with.
