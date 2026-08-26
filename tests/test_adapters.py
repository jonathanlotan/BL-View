"""Adapters must round-trip, be tolerant of damage, and be swappable."""

import numpy as np
import pytest

from blview.adapters import available_adapters, detect_adapter, get_adapter
from blview.adapters.base import AdapterError
from blview.adapters.generic_csv import GenericCSVAdapter
from blview.adapters.vaisala_cl import INT20_MAX, crc16_vaisala, decode_profile_hex
from blview.model import QualityFlag


def test_registry_has_both_bundled_adapters():
    assert {"vaisala_cl", "generic_csv"} <= set(available_adapters())


@pytest.mark.parametrize(
    "value", [0, 1, 1000, 12345, INT20_MAX, -1, -INT20_MAX - 1]
)
def test_20bit_twos_complement_round_trip(value):
    decoded = decode_profile_hex("%05X" % (value & 0xFFFFF), 1)
    assert int(decoded[0]) == value


def test_hex_decode_accepts_lowercase_and_flags_garbage():
    assert decode_profile_hex("000ab000FF", 2).astype(int).tolist() == [171, 255]
    out = decode_profile_hex("00001ZZZZZ", 2)
    assert out[0] == 1 and np.isnan(out[1])


def test_hex_decode_rejects_a_short_profile_line():
    with pytest.raises(AdapterError):
        decode_profile_hex("0000", 2)


def test_crc_is_deterministic_and_sensitive():
    assert crc16_vaisala(b"hello") == crc16_vaisala(b"hello")
    assert crc16_vaisala(b"hello") != crc16_vaisala(b"hellp")


def test_vaisala_round_trip_is_exact_to_quantisation(short_dataset):
    """Written -> parsed must agree to within half a profile count."""
    profiles = get_adapter("vaisala_cl").read(short_dataset["raw"])
    original = short_dataset["data"]
    assert profiles.n_time == original["beta"].shape[0]
    assert np.allclose(profiles.time, original["time"])
    assert np.allclose(profiles.range_, original["range"])
    # One profile count is 1e-9 m-1 sr-1, so rounding costs at most half of it.
    assert np.nanmax(np.abs(profiles.beta - original["beta"])) <= 0.5e-9 + 1e-15
    assert profiles.attrs["n_crc_failures"] == 0


def test_vaisala_reports_instrument_state(short_dataset):
    profiles = get_adapter("vaisala_cl").read(short_dataset["raw"])
    assert profiles.range_corrected is True       # CL firmware already did it
    assert profiles.background_subtracted is True
    assert profiles.range_resolution == pytest.approx(10.0)
    assert profiles.attrs["instrument_n_gates"] == 770


def test_sniffing_picks_the_right_adapter(short_dataset, tmp_path):
    assert detect_adapter(short_dataset["raw"]).name == "vaisala_cl"
    csv = tmp_path / "x.csv"
    csv.write_text("# blview-csv v1\ntime,10,20\n2026-01-01T00:00:00Z,1e-6,2e-6\n")
    assert detect_adapter(csv).name == "generic_csv"


def test_corrupt_message_is_skipped_not_fatal(short_dataset, tmp_path):
    """A damaged message must cost one profile, not the whole file."""
    text = short_dataset["raw"].read_text()
    good = get_adapter("vaisala_cl").read(short_dataset["raw"]).n_time
    blocks = text.split("\x01")
    blocks[3] = blocks[3][: len(blocks[3]) // 3]        # truncate one message
    damaged = tmp_path / "damaged.dat"
    damaged.write_text("\x01".join(blocks))
    assert get_adapter("vaisala_cl").read(damaged).n_time == good - 1


def test_crc_mismatch_flags_rather_than_fails(short_dataset, tmp_path):
    text = short_dataset["raw"].read_text().replace("\x03", "\x03ffff"[:1] + "0000", 1)
    path = tmp_path / "badcrc.dat"
    path.write_text(text)
    profiles = get_adapter("vaisala_cl").read(path)     # must not raise
    assert profiles.n_time > 0


def test_fog_status_digit_sets_the_fog_flag(tmp_path):
    """Detection status 4 means full obscuration."""
    from blview.synth.generator import SyntheticScenario, generate, write_vaisala_file
    scenario = SyntheticScenario(duration_h=0.2)
    data = generate(scenario, start_time=1756090800.0)   # 03:00 UTC -> fog
    raw = write_vaisala_file(tmp_path / "fog.dat", data, scenario)
    profiles = get_adapter("vaisala_cl").read(raw)
    assert ((profiles.quality & int(QualityFlag.FOG)) != 0).all()


def test_generic_csv_round_trip_and_range_corrected_flag(short_dataset, tmp_path):
    """The second adapter exists to prove a new format needs no downstream change."""
    profiles = get_adapter("vaisala_cl").read(short_dataset["raw"])
    path = GenericCSVAdapter.write(tmp_path / "out.csv", profiles)
    back = get_adapter("generic_csv").read(path)
    assert back.n_time == profiles.n_time
    assert np.allclose(back.range_, profiles.range_)
    assert np.nanmax(np.abs(back.beta - profiles.beta)) < 1e-11
    # The CSV writer preserved the flag; a CSV without it defaults to False,
    # which is what keeps the R^2 code path live.
    assert back.range_corrected is True
    bare = tmp_path / "bare.csv"
    bare.write_text("# blview-csv v1\ntime,10,20\n2026-01-01T00:00:00Z,1e-6,2e-6\n")
    assert get_adapter("generic_csv").read(bare).range_corrected is False


def test_unknown_adapter_name_is_a_clear_error():
    with pytest.raises(AdapterError, match="unknown adapter"):
        get_adapter("no_such_format")
