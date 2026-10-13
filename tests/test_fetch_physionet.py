"""Balanced-subset selection and index parsing for the PhysioNet fetch script.

Pure logic only — no network. The download itself needs egress to physionet.org.
"""

from collections import Counter

from cardiolens.validation.fetch_physionet import (
    UNLABELED,
    choose_folders,
    parse_index,
    parse_records,
    select_balanced,
)


def test_parse_records_and_index():
    assert parse_records("WFDBRecords/01/010/\nWFDBRecords/01/011\n") == [
        "WFDBRecords/01/010/", "WFDBRecords/01/011/"]
    html = '<a href="JS00002.hea">x</a> <a href="JS00002.mat">y</a> <a href="JS00001.hea">z</a>'
    assert parse_index(html) == ["JS00001", "JS00002"]


def test_choose_folders_spreads_across_top_level_dirs():
    folders = [f"WFDBRecords/{t:02d}/{t:02d}{s}/" for t in (1, 2, 3) for s in range(5)]
    chosen = choose_folders(folders, 6, seed=1)
    assert len(chosen) == 6
    assert Counter(f.split("/")[1] for f in chosen) == {"01": 2, "02": 2, "03": 2}
    assert choose_folders(folders, 6, seed=1) == chosen  # deterministic


def test_select_balanced_caps_each_class_and_prefers_rare_ones():
    records = {f"n{i}": {"sinus_bradycardia"} for i in range(50)}
    records.update({f"a{i}": {"atrial_fibrillation", "sinus_bradycardia"} for i in range(10)})
    records.update({f"f{i}": {"atrial_flutter"} for i in range(3)})
    records.update({f"u{i}": set() for i in range(7)})
    chosen = select_balanced(records, per_class=5, seed=0)
    labels = Counter(c for r in chosen for c in (records[r] or {UNLABELED}))
    assert labels["atrial_flutter"] == 3            # every rare record kept
    assert labels["atrial_fibrillation"] == 5       # capped
    assert labels[UNLABELED] == 5                   # negatives kept, capped
    # Common class filled by side effect + its own cap, never unbounded.
    assert 5 <= labels["sinus_bradycardia"] <= 10
    assert select_balanced(records, per_class=5, seed=0) == chosen
