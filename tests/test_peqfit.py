import math

import pytest

from ezbeq import iir, peqfit
from ezbeq.peqfit import HIGH_SHELF, LOW_SHELF, PEAKING, Filter, Layout

HYPERION = Layout(has_low_shelf=True, peaking_bands=14)


@pytest.mark.parametrize('kind,reference', [
    (PEAKING, iir.PeakingEQ),
    (LOW_SHELF, iir.LowShelf),
    (HIGH_SHELF, iir.HighShelf),
])
def test_coefficients_match_ezbeq_iir(kind, reference):
    b, a = peqfit.coefficients(kind, 25.0, 6.0, 0.7, 48000)
    expected = reference(48000, 25.0, 0.7, 6.0)

    assert [v / a[0] for v in b] == pytest.approx(expected.b, rel=1e-12)
    assert [v / a[0] for v in a] == pytest.approx(expected.a, rel=1e-12)


def test_response_of_a_peak_is_its_gain_at_its_centre():
    grid = peqfit.Grid(5.0, 300.0, 48000)
    response = grid.response([Filter(PEAKING, 40.0, 6.0, 2.0)])
    nearest = min(range(len(grid.freqs)), key=lambda i: abs(grid.freqs[i] - 40.0))

    assert response[nearest] == pytest.approx(6.0, abs=0.1)


def test_entry_which_fits_the_layout_is_used_as_it_is():
    entry = [
        {'type': LOW_SHELF, 'freq': 20.0, 'gain': 8.0, 'q': 0.7},
        {'type': PEAKING, 'freq': 30.0, 'gain': -2.0, 'q': 3.0, 'count': 2},
    ]

    result = peqfit.fit(entry, HYPERION)

    assert result.exact
    assert result.max_error_db == 0.0
    assert result.low_shelf == Filter(LOW_SHELF, 20.0, 8.0, 0.7)
    assert result.peaks == [Filter(PEAKING, 30.0, -2.0, 3.0)] * 2


def test_stacked_shelves_at_different_frequencies_are_fitted():
    entry = [
        {'type': LOW_SHELF, 'freq': 10.0, 'gain': 4.6, 'q': 1.0, 'count': 2},
        {'type': PEAKING, 'freq': 10.9, 'gain': -2.0, 'q': 4.0},
        {'type': LOW_SHELF, 'freq': 22.0, 'gain': 8.2, 'q': 0.7, 'count': 4},
        {'type': PEAKING, 'freq': 45.0, 'gain': -1.0, 'q': 1.6},
    ]

    result = peqfit.fit(entry, HYPERION)

    assert not result.exact
    assert result.max_error_db < 1.0
    assert len(result.peaks) <= 14
    assert all(abs(f.gain) <= peqfit.MAX_GAIN for f in result.filters)


def test_reported_error_matches_the_returned_filters():
    entry = [{'type': LOW_SHELF, 'freq': 15.0, 'gain': 5.0, 'q': 0.9, 'count': 3},
             {'type': LOW_SHELF, 'freq': 30.0, 'gain': 3.0, 'q': 0.7}]
    layout = Layout(has_low_shelf=True, peaking_bands=6)

    result = peqfit.fit(entry, layout)

    grid = peqfit.Grid(layout.min_freq, peqfit.DEFAULT_MAX_FREQ, layout.fs)
    target = grid.response(peqfit.expand_filters(entry))
    actual = max(abs(a - b) for a, b in zip(grid.response(result.filters), target, strict=True))
    assert result.max_error_db == pytest.approx(actual)


def test_filters_stay_above_the_minimum_frequency():
    entry = [{'type': LOW_SHELF, 'freq': 12.0, 'gain': 5.0, 'q': 0.9, 'count': 4},
             {'type': PEAKING, 'freq': 8.0, 'gain': 3.0, 'q': 2.0}]

    result = peqfit.fit(entry, Layout(has_low_shelf=True, peaking_bands=14, min_freq=18.0))

    assert all(f.freq >= 18.0 for f in result.filters)
    assert result.max_error_db < 1.0


def test_max_boost_limits_the_response():
    entry = [{'type': LOW_SHELF, 'freq': 20.0, 'gain': 6.0, 'q': 0.7, 'count': 3}]
    layout = Layout(has_low_shelf=True, peaking_bands=14, max_boost=10.0)

    result = peqfit.fit(entry, layout)

    grid = peqfit.Grid(layout.min_freq, peqfit.DEFAULT_MAX_FREQ, layout.fs)
    assert max(grid.response(result.filters)) <= 10.0 + result.max_error_db


def test_peaks_only_layout_has_no_shelf():
    entry = [{'type': LOW_SHELF, 'freq': 25.0, 'gain': 6.0, 'q': 0.7}]

    result = peqfit.fit(entry, Layout(has_low_shelf=False, peaking_bands=8))

    assert result.low_shelf is None
    assert all(f.filter_type == PEAKING for f in result.peaks)
    assert result.max_error_db < 1.0


def test_high_shelf_is_fitted_rather_than_copied():
    entry = [{'type': HIGH_SHELF, 'freq': 120.0, 'gain': -3.0, 'q': 0.7}]

    result = peqfit.fit(entry, HYPERION)

    assert not result.exact
    assert all(f.filter_type != HIGH_SHELF for f in result.filters)


def test_unknown_filter_type_is_rejected():
    with pytest.raises(peqfit.UnsupportedEntryError, match='AllPass'):
        peqfit.fit([{'type': 'AllPass', 'freq': 10.0, 'gain': 0.0, 'q': 1.0}], HYPERION)


def test_malformed_filter_is_rejected():
    with pytest.raises(peqfit.UnsupportedEntryError, match='invalid values'):
        peqfit.fit([{'type': PEAKING, 'gain': 0.0, 'q': 1.0}], HYPERION)


def test_linear_solver():
    solution = peqfit._solve_linear([[4.0, 1.0], [2.0, 3.0]], [1.0, 2.0])

    assert solution == pytest.approx([0.1, 0.6])
    assert not any(math.isnan(v) for v in solution)
