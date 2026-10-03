"""MIMIC-IV-ECG cart statements -> gold finding codes; the fetch script's selection helpers (D62)."""

from cardiolens.validation import mimic_statements_to_codes
from cardiolens.validation.fetch_mimic import CLASSES, record_path


def test_stemi_statements_and_their_wording_variants():
    assert "st_elevation" in mimic_statements_to_codes(["*** CONSIDER ACUTE ST ELEVATION MI ***", "SINUS RHYTHM"])
    assert "st_elevation" in mimic_statements_to_codes(["INFERIOR ST ELEVATION, CONSIDER ACUTE INFARCT"])
    assert "st_elevation" in mimic_statements_to_codes(["ANTERIOR INFARCT, ACUTE"])
    # an old infarct is not an acute one
    assert "st_elevation" not in mimic_statements_to_codes(["INFERIOR INFARCT - AGE UNDETERMINED"])


def test_exclusions_keep_borderline_possible_and_incomplete_out():
    assert mimic_statements_to_codes(["INCOMPLETE LEFT BUNDLE BRANCH BLOCK"]) == set()
    assert mimic_statements_to_codes(["POSSIBLE LEFT VENTRICULAR HYPERTROPHY"]) == set()
    assert mimic_statements_to_codes(["SINUS RHYTHM WITH BORDERLINE 1ST DEGREE A-V BLOCK"]) == {"sinus_rhythm"}
    assert mimic_statements_to_codes(["ST JUNCTIONAL DEPRESSION IS NONSPECIFIC"]) == set()


def test_rhythm_and_hyperkalaemia_statements():
    codes = mimic_statements_to_codes(["SINUS BRADYCARDIA", "TALL T WAVES - CONSIDER ACUTE ISCHEMIA OR HYPERKALEMIA",
                                       "PROLONGED QT INTERVAL", "LEFT BUNDLE BRANCH BLOCK"])
    assert codes == {"sinus_bradycardia", "peaked_t_waves", "long_qt", "lbbb"}
    assert mimic_statements_to_codes(["ATRIAL FIBRILLATION WITH RAPID VENTRICULAR RESPONSE"]) == {"atrial_fibrillation"}


def test_record_path_follows_the_dataset_layout_and_classes_are_disjoint_regexes():
    assert record_path(10000032, 40689238) == "files/p1000/p10000032/s40689238/40689238"
    assert set(CLASSES) == {"stemi", "normal", "tall_t", "long_qt", "lbbb"}
