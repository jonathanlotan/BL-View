"""Adapter for Vaisala CL31 / CL51 raw ASCII data messages ("message 2").

Format notes
------------
The CL-series data message is proprietary and only semi-documented in the
public manuals.  The structure implemented here is the one that is consistently
described in the CL31/CL51 user guides and reproduced by the community
toolchains, and it is exactly what :mod:`blview.synth.generate` writes:

    -2026-08-25 00:00:00                      <- logger timestamp line ("-" prefix)
    <SOH>CL010112<CR><LF>                     <- message header
    30 01230 12340 23450 FEDCBA98<CR><LF>     <- detection status / cloud bases
    00100 10 0770 099 +34 099 0000 L0112HN15 139 026<CR><LF>   <- parameters
    <3850 hex characters><CR><LF>             <- 770 gates x 5 hex chars
    <ETX>1a2b<CR><LF>                         <- end of text + CRC-16

Every field this parser *interprets* is listed in ASSUMPTIONS.md (section V).
The parser is deliberately tolerant: it recovers the gate count and vertical
resolution from the parameter line (falling back to the profile-line length),
never trusts the header digits for anything numeric, and treats a CRC mismatch
as a quality flag rather than a fatal error -- real deployments differ in their
CRC seeding and in whether the logger rewrites line endings.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..model import ProfileSet, QualityFlag
from .base import AdapterError, CeilometerAdapter, register_adapter

SOH = "\x01"
ETX = "\x03"

#: Physical value of one profile count.  The CL-series profile is a 20-bit
#: two's-complement integer; the manuals give its unit as 1e-9 m-1 sr-1 but
#: this is one of the least well documented parts of the format, so it is a
#: single named constant that can be overridden per deployment via the
#: ``profile_unit`` adapter option (ASSUMPTIONS.md #V4).
PROFILE_UNIT_M1_SR1 = 1.0e-9

#: 20-bit two's-complement bounds.  Real CL31/CL51 profiles clip here inside
#: dense cloud, which is why cloud tops are so often unavailable.
INT20_MAX = 0x7FFFF          # 524287
INT20_MOD = 0x100000         # 1048576

#: ``-YYYY-MM-DD hh:mm:ss`` logger timestamp line.
_TS_RE = re.compile(
    r"^-\s*(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})"
)
_HEADER_RE = re.compile(r"^\x01(?P<kind>CL|CT)(?P<digits>[0-9A-Za-z]{2,8})\s*$")
_HEX_LINE_RE = re.compile(r"^[0-9A-Fa-f]{100,}$")

#: Detection-status digit -> meaning (CL31/CL51 message 2, character 1).
DETECTION_STATUS = {
    "0": "no significant backscatter",
    "1": "one cloud base detected",
    "2": "two cloud bases detected",
    "3": "three cloud bases detected",
    "4": "full obscuration, no cloud base",
    "5": "some obscuration, transparent",
    "/": "raw data missing or suspect",
}


def crc16_vaisala(payload: bytes) -> int:
    """CRC-16 as used by Vaisala ASCII messages.

    Reflected CCITT polynomial 0x8408, initial value 0xFFFF, final complement.
    The exact seeding is not published; see ASSUMPTIONS.md #V5.  Only used to
    *flag* suspect messages, never to reject them.
    """
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return (~crc) & 0xFFFF


def decode_profile_hex(hexstr: str, n_gates: int) -> np.ndarray:
    """Decode ``n_gates`` 5-character 20-bit two's-complement fields.

    Returns raw integer counts; ``NaN`` where a field is not valid hex.
    """
    expected = n_gates * 5
    if len(hexstr) < expected:
        raise AdapterError(
            f"profile line too short: {len(hexstr)} chars, expected {expected}"
        )
    raw = np.frombuffer(hexstr[:expected].encode("ascii"), dtype="S1").reshape(
        n_gates, 5
    )
    out = np.empty(n_gates, dtype="float64")
    # Vectorised hex decode: map ASCII codes to nibble values.
    codes = raw.view(np.uint8)
    nib = np.full(codes.shape, -1, dtype="int32")
    digit = (codes >= 48) & (codes <= 57)
    upper = (codes >= 65) & (codes <= 70)
    lower = (codes >= 97) & (codes <= 102)
    nib[digit] = codes[digit] - 48
    nib[upper] = codes[upper] - 55
    nib[lower] = codes[lower] - 87
    bad = (nib < 0).any(axis=1)
    value = np.zeros(n_gates, dtype="int64")
    for k in range(5):
        value = (value << 4) | np.maximum(nib[:, k], 0).astype("int64")
    # 20-bit two's complement -> signed
    signed = np.where(value > INT20_MAX, value - INT20_MOD, value)
    out[:] = signed
    out[bad] = np.nan
    return out


class _Message:
    """One parsed raw message."""

    __slots__ = (
        "time", "status_digit", "warning_flag", "cloud_bases", "vertical_visibility",
        "highest_signal", "alarm_hex", "scale", "resolution_m", "n_gates",
        "laser_energy_pct", "laser_temp_c", "receiver_sensitivity_pct",
        "window_contamination_mv", "instrument_params", "background_light_mv",
        "backscatter_sum", "counts", "crc_ok",
    )

    def __init__(self) -> None:
        self.time: float | None = None
        self.status_digit = "0"
        self.warning_flag = "0"
        self.cloud_bases: list[float] = []
        self.vertical_visibility: float | None = None
        self.highest_signal: float | None = None
        self.alarm_hex = ""
        self.scale = 100.0
        self.resolution_m = 10.0
        self.n_gates = 0
        self.laser_energy_pct = np.nan
        self.laser_temp_c = np.nan
        self.receiver_sensitivity_pct = np.nan
        self.window_contamination_mv = np.nan
        self.instrument_params = ""
        self.background_light_mv = np.nan
        self.backscatter_sum = np.nan
        self.counts: np.ndarray | None = None
        self.crc_ok: bool | None = None


@register_adapter
class VaisalaCLAdapter(CeilometerAdapter):
    """Reader for logged Vaisala CL31/CL51 raw message files."""

    name = "vaisala_cl"
    description = "Vaisala CL31/CL51 raw ASCII data message (message 2), as logged"
    patterns = ("*.dat", "*.DAT", "*.txt", "*.raw")

    def __init__(
        self,
        profile_unit: float = PROFILE_UNIT_M1_SR1,
        default_interval_s: float = 30.0,
        start_time: float | None = None,
        strict_crc: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.profile_unit = float(profile_unit)
        self.default_interval_s = float(default_interval_s)
        self.start_time = start_time
        self.strict_crc = bool(strict_crc)

    # ---------------------------------------------------------------- sniff
    @classmethod
    def sniff(cls, path: str | Path) -> bool:
        try:
            with open(path, "r", encoding="ascii", errors="replace") as fh:
                head = fh.read(8192)
        except OSError:
            return False
        if SOH + "CL" in head or SOH + "CT" in head:
            return True
        # Some loggers strip SOH.  Fall back to "timestamp line followed by a
        # long hex line" which is still highly specific to this format.
        has_ts = bool(_TS_RE.search(head))
        has_hex = any(_HEX_LINE_RE.match(ln.strip()) for ln in head.splitlines())
        return has_ts and has_hex

    # ----------------------------------------------------------------- read
    def read(self, path: str | Path) -> ProfileSet:
        path = Path(path)
        text = path.read_text(encoding="ascii", errors="replace")
        messages = list(self._iter_messages(text))
        if not messages:
            raise AdapterError(f"no CL31/CL51 data messages found in {path}")

        self._assign_missing_times(messages, path)

        resolutions = {m.resolution_m for m in messages}
        gate_counts = {m.n_gates for m in messages}
        if len(resolutions) > 1 or len(gate_counts) > 1:
            raise AdapterError(
                f"{path}: mixed range grids in one file "
                f"(resolutions={sorted(resolutions)}, gates={sorted(gate_counts)}); "
                "split the file per instrument configuration"
            )
        resolution = resolutions.pop()
        n_gates = gate_counts.pop()

        # Gate centres.  Gate i integrates [i*res, (i+1)*res) -> centre at
        # (i + 0.5) * res  (ASSUMPTIONS.md #V6).
        range_ = (np.arange(n_gates, dtype="float64") + 0.5) * resolution

        n_time = len(messages)
        beta = np.full((n_time, n_gates), np.nan)
        time = np.empty(n_time, dtype="float64")
        quality = np.zeros(n_time, dtype="int64")
        cloud_reported = np.full((n_time, 3), np.nan)

        for i, m in enumerate(messages):
            time[i] = m.time
            counts = m.counts
            scale = m.scale if m.scale > 0 else 100.0
            beta[i] = counts * self.profile_unit * (100.0 / scale)

            flags = QualityFlag.OK
            if m.warning_flag.upper() == "W":
                flags |= QualityFlag.INSTRUMENT_WARNING
            elif m.warning_flag.upper() == "A":
                flags |= QualityFlag.INSTRUMENT_ALARM
            if m.status_digit == "4":
                flags |= QualityFlag.FOG
            if m.status_digit == "/":
                flags |= QualityFlag.LOW_SNR
            if np.isfinite(m.window_contamination_mv) and m.window_contamination_mv > 2000:
                flags |= QualityFlag.WINDOW_CONTAMINATED
            if np.nanmax(np.abs(counts)) >= INT20_MAX:
                flags |= QualityFlag.SATURATED
            if m.crc_ok is False:
                flags |= QualityFlag.LOW_SNR if self.strict_crc else QualityFlag.OK
            quality[i] = int(flags)

            for k, cb in enumerate(m.cloud_bases[:3]):
                cloud_reported[i, k] = cb

        order = np.argsort(time, kind="stable")
        last = messages[order[-1]]
        attrs: dict[str, Any] = {
            "source_file": str(path),
            "adapter": self.name,
            "instrument_resolution_m": resolution,
            "instrument_n_gates": n_gates,
            "profile_unit_m1_sr1": self.profile_unit,
            "laser_energy_pct": last.laser_energy_pct,
            "laser_temp_c": last.laser_temp_c,
            "receiver_sensitivity_pct": last.receiver_sensitivity_pct,
            "window_contamination_mv": last.window_contamination_mv,
            "instrument_params": last.instrument_params,
            "n_crc_failures": sum(1 for m in messages if m.crc_ok is False),
        }

        return ProfileSet(
            time=time[order],
            range_=range_,
            beta=beta[order],
            # CL-series profiles are delivered already range- and
            # background-corrected by the instrument firmware.
            range_corrected=True,
            background_subtracted=True,
            quality=quality[order],
            cloud_base_reported=cloud_reported[order],
            attrs=attrs,
        )

    # --------------------------------------------------------------- internal
    def _iter_messages(self, text: str):
        """Split the file into messages and parse each one."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        pending_time: float | None = None
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            ts = _TS_RE.match(line.strip())
            if ts:
                y, mo, d, h, mi, s = (int(g) for g in ts.groups())
                pending_time = dt.datetime(
                    y, mo, d, h, mi, s, tzinfo=dt.timezone.utc
                ).timestamp()
                i += 1
                continue
            if not line.startswith(SOH) and not _HEADER_RE.match(line):
                i += 1
                continue
            block, i = self._collect_block(lines, i)
            try:
                msg = self._parse_block(block)
            except AdapterError:
                continue  # skip corrupt message, keep going
            if msg is None:
                continue
            msg.time = pending_time
            pending_time = None
            yield msg

    @staticmethod
    def _collect_block(lines: list[str], start: int) -> tuple[list[str], int]:
        """Collect lines from a header up to (and including) the ETX line."""
        block = [lines[start]]
        i = start + 1
        while i < len(lines):
            ln = lines[i]
            if ln.startswith(SOH):          # next message began without an ETX
                break
            if _TS_RE.match(ln.strip()) and len(block) > 3:
                break
            block.append(ln)
            i += 1
            if ln.startswith(ETX):
                break
        return block, i

    def _parse_block(self, block: list[str]) -> _Message | None:
        msg = _Message()
        # Locate the profile line first -- it is unambiguous and anchors the rest.
        hex_idx = None
        for idx, ln in enumerate(block):
            if _HEX_LINE_RE.match(ln.strip()):
                hex_idx = idx
                break
        if hex_idx is None or hex_idx < 2:
            return None
        hexstr = block[hex_idx].strip()

        self._parse_status_line(block[1], msg)
        self._parse_parameter_line(block[hex_idx - 1], msg)

        if msg.n_gates <= 0 or msg.n_gates * 5 > len(hexstr):
            msg.n_gates = len(hexstr) // 5
        if msg.n_gates <= 0:
            return None
        msg.counts = decode_profile_hex(hexstr, msg.n_gates)

        # CRC over everything after <SOH> up to and including <ETX>.
        for ln in block[hex_idx + 1:]:
            if ln.startswith(ETX):
                given = ln[1:5]
                payload = ("\r\n".join([block[0][1:]] + block[1:hex_idx + 1]) + "\r\n" + ETX)
                try:
                    msg.crc_ok = int(given, 16) == crc16_vaisala(payload.encode("ascii"))
                except ValueError:
                    msg.crc_ok = False
                break
        return msg

    @staticmethod
    def _parse_status_line(line: str, msg: _Message) -> None:
        """``30 01230 12340 23450 FEDCBA98`` -> detection status + cloud bases."""
        s = line.strip()
        if not s:
            return
        msg.status_digit = s[0]
        msg.warning_flag = s[1] if len(s) > 1 else "0"
        tokens = s[2:].split()
        if tokens and re.fullmatch(r"[0-9A-Fa-f]{8}", tokens[-1]):
            msg.alarm_hex = tokens[-1]
            tokens = tokens[:-1]
        values: list[float] = []
        for tok in tokens:
            try:
                values.append(float(int(tok)))
            except ValueError:
                values.append(np.nan)
        if msg.status_digit in "123":
            msg.cloud_bases = values[: int(msg.status_digit)]
        elif msg.status_digit in "45":
            # Field 1 is vertical visibility, field 2 the highest signal seen.
            if values:
                msg.vertical_visibility = values[0]
            if len(values) > 1:
                msg.highest_signal = values[1]

    @staticmethod
    def _parse_parameter_line(line: str, msg: _Message) -> None:
        """``00100 10 0770 099 +34 099 0000 L0112HN15 139 026``."""
        tokens = line.strip().split()

        def num(idx: int, default: float = np.nan) -> float:
            if idx >= len(tokens):
                return default
            try:
                return float(int(tokens[idx]))
            except ValueError:
                return default

        msg.scale = num(0, 100.0)
        res = num(1, 10.0)
        msg.resolution_m = res if res and np.isfinite(res) and res > 0 else 10.0
        gates = num(2, 0.0)
        msg.n_gates = int(gates) if np.isfinite(gates) else 0
        msg.laser_energy_pct = num(3)
        msg.laser_temp_c = num(4)
        msg.receiver_sensitivity_pct = num(5)
        msg.window_contamination_mv = num(6)
        # Token 7 ("L0112HN15") is the instrument's internal measurement
        # parameter string.  Its layout is not documented; kept verbatim.
        msg.instrument_params = tokens[7] if len(tokens) > 7 else ""
        msg.background_light_mv = num(8)
        msg.backscatter_sum = num(9)

    def _assign_missing_times(self, messages: list[_Message], path: Path) -> None:
        """Fill in timestamps for files whose logger did not write them."""
        known = [(i, m.time) for i, m in enumerate(messages) if m.time is not None]
        if len(known) == len(messages):
            return
        if not known:
            # Nothing at all: synthesise a regular series ending at file mtime,
            # or starting at an explicitly supplied start_time.
            if self.start_time is not None:
                t0 = float(self.start_time)
            else:
                t0 = path.stat().st_mtime - self.default_interval_s * (len(messages) - 1)
            for i, m in enumerate(messages):
                m.time = t0 + i * self.default_interval_s
            return
        # Some known: interpolate/extrapolate at the observed cadence.
        idx = np.array([i for i, _ in known], dtype="float64")
        tt = np.array([t for _, t in known], dtype="float64")
        cadence = (
            float(np.median(np.diff(tt) / np.diff(idx)))
            if len(known) > 1
            else self.default_interval_s
        )
        for i, m in enumerate(messages):
            if m.time is None:
                j = int(np.argmin(np.abs(idx - i)))
                m.time = tt[j] + (i - idx[j]) * cadence
