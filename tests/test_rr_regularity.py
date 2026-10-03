"""An isolated RR of twice the median is a missed detection, not irregularity (D52).

On a 10-s strip one undetected R leaves a single doubled RR, and that alone
pushed the RR coefficient of variation from 0.07 to 0.28 — past the AF gate — on
digitised pages where a small complex was lost. 2:1 block (every RR long) and AF
(no clustering at 2x) are left untouched.
"""

import numpy as np

from cardiolens.layer3_measure.measure import _rr_for_regularity


def _cv(x):
    return float(np.std(x) / np.mean(x))


def test_single_doubled_rr_is_healed():
    rr = np.array([0.8, 0.82, 0.8, 1.6, 0.81, 0.8, 0.79, 0.8])
    healed, n = _rr_for_regularity(rr)
    assert n == 1 and len(healed) == len(rr) + 1
    assert _cv(rr) > 0.25 and _cv(healed) < 0.05


def test_two_to_one_block_is_not_touched():
    rr = np.array([1.6, 1.62, 1.6, 1.58, 1.6, 1.61])        # every RR is the long one
    healed, n = _rr_for_regularity(rr)
    assert n == 0 and np.array_equal(healed, rr)


def test_atrial_fibrillation_is_not_touched():
    rng = np.random.default_rng(1)
    rr = rng.uniform(0.45, 1.3, size=14)                     # irregularly irregular
    healed, n = _rr_for_regularity(rr)
    assert _cv(healed) > 0.2
    assert n <= 0.25 * len(rr)


def test_many_long_intervals_are_a_rhythm_not_detector_loss():
    rr = np.array([0.8, 1.6, 0.8, 1.6, 0.8, 1.6, 0.8, 1.6])  # bigeminy-like alternation
    healed, n = _rr_for_regularity(rr)
    assert n == 0 and np.array_equal(healed, rr)
