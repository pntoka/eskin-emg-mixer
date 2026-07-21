"""Serial port auto-detection: figures out which Arduino-like port is the
e-skin board and which is the dual-load-cell force-sensor board, so COM
ports never need to be hardcoded."""

import time

import serial
import serial.tools.list_ports

from .eskin import ESKIN_BAUD, ESKIN_DATA_LEN, ESKIN_GRID, ESKIN_MODE
from .forces import FORCE_FRAME_SIZE, FORCE_HEADER0, FORCE_HEADER1, FORCES_BAUD

# USB VIDs seen on Arduino boards and the common clone USB-serial chips.
_ARDUINO_VIDS = {
    0x2341,  # Arduino LLC (genuine)
    0x2A03,  # Arduino SRL
    0x1B4F,  # SparkFun
    0x1A86,  # QinHeng CH340/CH341 (very common in cheap clones)
    0x0403,  # FTDI FT232 (older/cloned Unos, Nano)
    0x10C4,  # Silicon Labs CP210x
    0x16C0,  # Van Ooijen Technische Informatica (Teensy / native-USB boards)
}
_DESC_KEYWORDS = ("arduino", "ch340", "ch341", "cp210",
                  "ftdi", "usb serial", "usb-serial", "wchusbserial")


def list_ports() -> None:
    """Print all serial ports currently visible to the OS."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")


def _arduino_like_ports():
    """Return device names of all ports that look like an Arduino/clone."""
    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if (p.vid is not None and p.vid in _ARDUINO_VIDS) or \
           any(k in desc for k in _DESC_KEYWORDS):
            candidates.append(p.device)
    return candidates


def _probe_eskin(port_name: str, timeout: float = 2.5) -> bool:
    """Return True if the board on `port_name` behaves like the e-skin
    request/response protocol: silent until asked, then replies with a
    full 520-byte frame.

    The silence check matters because a continuously streaming board (like
    the force-sensor board) would otherwise also "succeed" a naive
    fixed-length read of ESKIN_DATA_LEN bytes -- it just accumulates
    whatever happens to be flowing by, regardless of what command (if any)
    was sent.
    """
    try:
        with serial.Serial(port_name, ESKIN_BAUD, timeout=timeout) as ser:
            time.sleep(2.0)  # let the board finish its auto-reset
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # Unsolicited silence check: rules out continuous streamers.
            idle = ser.read(16)
            if len(idle) > 0:
                return False

            cmd = ((ESKIN_MODE << 6) | ESKIN_GRID).to_bytes(1, "big")
            ser.write(cmd)
            d = ser.read(ESKIN_DATA_LEN)
            return len(d) == ESKIN_DATA_LEN
    except Exception:
        return False


def _probe_forces(port_name: str, timeout: float = 2.5) -> bool:
    """Return True if the board on `port_name` streams the two-load-cell
    frame protocol (0xAA 0x55 header recurring every 10 bytes)."""
    try:
        with serial.Serial(port_name, FORCES_BAUD, timeout=timeout) as ser:
            time.sleep(2.0)
            ser.reset_input_buffer()
            chunk = ser.read(300)
            for i in range(len(chunk) - FORCE_FRAME_SIZE - 1):
                if (chunk[i] == FORCE_HEADER0 and chunk[i + 1] == FORCE_HEADER1
                        and chunk[i + FORCE_FRAME_SIZE] == FORCE_HEADER0
                        and chunk[i + FORCE_FRAME_SIZE + 1] == FORCE_HEADER1):
                    return True
            return False
    except Exception:
        return False


def find_two_ports():
    """
    Auto-detect which serial port is the e-skin board and which is the
    dual-load-cell board. Probes every Arduino-like port with both
    protocols and matches on the response.
    """
    candidates = _arduino_like_ports()
    if len(candidates) < 2:
        all_ports = list(serial.tools.list_ports.comports())
        listing = "\n".join(f"  {p.device:20s}  {p.description}"
                             for p in all_ports) or "  (none)"
        raise RuntimeError(
            "Need two Arduino-like serial ports (e-skin + force sensors), "
            f"found {len(candidates)}. Available ports:\n{listing}\n"
            "Close the Arduino IDE Serial Monitor, check both boards are "
            "plugged in, or pass the ports explicitly."
        )

    eskin_port = None
    forces_port = None
    print("Probing ports to identify the e-skin and force-sensor boards...")
    for port in candidates:
        if eskin_port is None and _probe_eskin(port):
            print(f"  {port}: e-skin board (16x16 FSR matrix)")
            eskin_port = port
            continue
        if forces_port is None and _probe_forces(port):
            print(f"  {port}: force-sensor board (2 load cells)")
            forces_port = port
            continue
        print(f"  {port}: no match")

    if eskin_port is None or forces_port is None:
        raise RuntimeError(
            "Could not identify both boards automatically "
            f"(eskin={eskin_port}, forces={forces_port}). "
            "Check connections/firmware, or pass the ports explicitly."
        )
    return eskin_port, forces_port
