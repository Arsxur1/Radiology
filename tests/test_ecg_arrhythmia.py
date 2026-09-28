"""ecg-arrhythmia (Chapman/Ningbo) adapter: SNOMED mapping + header parsing.

The dataset files themselves are not present (PhysioNet is not reachable from
this environment's egress policy), so these tests cover the pure label-mapping
and header-parsing logic that runs once a local checkout is provided.
"""

from cardiolens.validation import parse_dx_from_header, snomed_dx_to_codes


def test_snomed_maps_known_codes():
    codes = snomed_dx_to_codes(["426783006", "39732003", "164889003"])
    assert codes == {"sinus_rhythm", "left_axis_deviation", "atrial_fibrillation"}


def test_snomed_ignores_unmapped_and_accepts_ints():
    assert snomed_dx_to_codes([426177001, "999999999"]) == {"sinus_bradycardia"}


def test_parse_dx_from_header_comma_and_semicolon():
    header = (
        "JS00001 12 500 5000\n"
        "JS00001.mat 16 1000/mV 16 0 0 0 0 I\n"
        "#Age: 61\n"
        "#Sex: Male\n"
        "#Dx: 426783006,39732003;164931005\n"
    )
    dx = parse_dx_from_header(header)
    assert dx == ["426783006", "39732003", "164931005"]
    assert snomed_dx_to_codes(dx) == {"sinus_rhythm", "left_axis_deviation", "st_elevation"}


def test_parse_dx_challenge2021_spaced_comment():
    """Challenge 2021 headers write '# Dx:' — must not silently yield no gold labels."""
    header = ("E03527 12 500 5000\n# Age: 51\n# Sex: Male\n"
              "# Dx: 426783006,164889003\n# Rx: Unknown\n")
    assert parse_dx_from_header(header) == ["426783006", "164889003"]


def test_parse_dx_missing_returns_empty():
    assert parse_dx_from_header("JS 12 500\n#Age: 40\n") == []
