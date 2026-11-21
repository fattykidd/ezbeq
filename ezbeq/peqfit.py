"""
Fits a BEQ filter set onto a device with a fixed parametric EQ layout.

Some processors cannot take a catalogue entry filter for filter: they offer a single low shelf and a handful of peaking
bands per channel, while BEQ entries regularly stack several low shelves at different frequencies. Instead of copying
the filters, this module reproduces the entry's combined magnitude response with the bands the device does have.

The response is fitted with a Levenberg-Marquardt least squares solver over a log spaced frequency grid. It is written
in plain Python so that it adds no dependencies to ezbeq; each fit takes well under a second.

Filters follow the RBJ Audio EQ Cookbook, the same definitions as ezbeq.iir, without the rounding ezbeq.iir applies.
"""
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LOW_SHELF = 'LowShelf'
HIGH_SHELF = 'HighShelf'
PEAKING = 'PeakingEQ'
FILTER_TYPES = (LOW_SHELF, HIGH_SHELF, PEAKING)

DEFAULT_FS = 48000
DEFAULT_MAX_FREQ = 300.0
GRID_POINTS_PER_OCTAVE = 24

# bounds placed on fitted filters, the range the device is known to accept
MIN_GAIN = -24.0
MAX_GAIN = 24.0
MIN_Q = 0.2
MAX_Q = 10.0
MAX_FREQ = 20000.0

# solver tuning
EXACT_ENOUGH_DB = 0.05
MAX_ITERATIONS = 150
FD_STEP = 1e-5
INITIAL_DAMPING = 1e-2


@dataclass(frozen=True)
class Filter:
    filter_type: str
    freq: float
    gain: float
    q: float

    def as_dict(self) -> dict:
        return {'type': self.filter_type, 'freq': self.freq, 'gain': self.gain, 'q': self.q}


@dataclass(frozen=True)
class Layout:
    """
    What a device can do on each channel.

    :param has_low_shelf: whether one band can be a low shelf.
    :param peaking_bands: how many peaking bands are available.
    :param min_freq: the lowest frequency to reproduce and to place a filter at.
    :param max_boost: if set, the response is never allowed to exceed this many dB.
    :param fs: the sample rate the device's filters run at.
    """
    has_low_shelf: bool
    peaking_bands: int
    min_freq: float = 5.0
    max_boost: float | None = None
    fs: int = DEFAULT_FS


@dataclass(frozen=True)
class FitResult:
    low_shelf: Filter | None
    peaks: list[Filter]
    max_error_db: float
    exact: bool

    @property
    def filters(self) -> list[Filter]:
        return ([self.low_shelf] if self.low_shelf else []) + self.peaks


class UnsupportedEntryError(ValueError):
    pass


def expand_filters(entry_filters: Iterable[dict]) -> list[Filter]:
    """
    Converts catalogue entry filters (which may carry a repeat count) into a flat list of filters.
    """
    expanded = []
    for idx, f in enumerate(entry_filters):
        filter_type = f.get('type')
        if filter_type not in FILTER_TYPES:
            raise UnsupportedEntryError(f'Unsupported filter type {filter_type} at index {idx}')
        try:
            parsed = Filter(filter_type, float(f['freq']), float(f['gain']), float(f['q']))
            count = int(f.get('count', 1))
        except (KeyError, TypeError, ValueError) as e:
            raise UnsupportedEntryError(f'Filter {idx} has invalid values') from e
        expanded.extend([parsed] * count)
    return expanded


def fit(entry_filters: Iterable[dict], layout: Layout) -> FitResult:
    """
    Produces the device filters which best reproduce the given catalogue entry filters.
    """
    filters = expand_filters(entry_filters)
    exact = _try_exact(filters, layout)
    if exact:
        return exact

    grid = Grid(layout.min_freq, DEFAULT_MAX_FREQ, layout.fs)
    target = grid.response(filters)
    if layout.max_boost is not None:
        target = [min(t, layout.max_boost) for t in target]

    entry_peaks = sum(1 for f in filters if f.filter_type == PEAKING)
    best = None
    for n_peaks in _peak_counts(entry_peaks, layout.peaking_bands):
        result = _solve(grid, target, filters, layout, n_peaks)
        if best is None or result.max_error_db < best.max_error_db:
            best = result
        if best.max_error_db <= 0.25:
            break
    return best


def _peak_counts(entry_peaks: int, available: int) -> list[int]:
    first = min(available, entry_peaks + 3)
    return [first] if first == available else [first, available]


def _try_exact(filters: list[Filter], layout: Layout) -> FitResult | None:
    """
    Uses the entry filters as they are when the device can hold them without any approximation.
    """
    low_shelves = [f for f in filters if f.filter_type == LOW_SHELF]
    peaks = [f for f in filters if f.filter_type == PEAKING]
    if any(f.filter_type == HIGH_SHELF for f in filters):
        return None
    if len(low_shelves) > (1 if layout.has_low_shelf else 0) or len(peaks) > layout.peaking_bands:
        return None
    if any(not _within_bounds(f, layout) for f in filters):
        return None
    if layout.max_boost is not None:
        grid = Grid(layout.min_freq, DEFAULT_MAX_FREQ, layout.fs)
        if max(grid.response(filters)) > layout.max_boost:
            return None
    return FitResult(low_shelves[0] if low_shelves else None, peaks, 0.0, True)


def _within_bounds(f: Filter, layout: Layout) -> bool:
    return (layout.min_freq <= f.freq <= MAX_FREQ
            and MIN_GAIN <= f.gain <= MAX_GAIN
            and MIN_Q <= f.q <= MAX_Q)


class Grid:
    """
    The log spaced frequencies a fit is evaluated at, with the trigonometry for each one precomputed.
    """

    def __init__(self, min_freq: float, max_freq: float, fs: int):
        octaves = math.log2(max_freq / min_freq)
        count = max(2, math.ceil(octaves * GRID_POINTS_PER_OCTAVE) + 1)
        self.freqs = [min_freq * (max_freq / min_freq) ** (i / (count - 1)) for i in range(count)]
        self.fs = fs
        omegas = [2.0 * math.pi * f / fs for f in self.freqs]
        self.cos_w = [math.cos(w) for w in omegas]
        self.cos_2w = [math.cos(2.0 * w) for w in omegas]

    def filter_response(self, filter_type: str, freq: float, gain: float, q: float) -> list[float]:
        b, a = coefficients(filter_type, freq, gain, q, self.fs)
        num_c = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
        num_1 = 2.0 * (b[0] * b[1] + b[1] * b[2])
        num_2 = 2.0 * b[0] * b[2]
        den_c = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        den_1 = 2.0 * (a[0] * a[1] + a[1] * a[2])
        den_2 = 2.0 * a[0] * a[2]
        return [
            10.0 * math.log10((num_c + num_1 * c1 + num_2 * c2) / (den_c + den_1 * c1 + den_2 * c2))
            for c1, c2 in zip(self.cos_w, self.cos_2w, strict=True)
        ]

    def response(self, filters: Sequence[Filter]) -> list[float]:
        total = [0.0] * len(self.freqs)
        for f in filters:
            for i, v in enumerate(self.filter_response(f.filter_type, f.freq, f.gain, f.q)):
                total[i] += v
        return total


def coefficients(filter_type: str, freq: float, gain: float, q: float, fs: int) -> tuple[list[float], list[float]]:
    """
    RBJ Audio EQ Cookbook coefficients as (b, a), unnormalised.
    """
    a_lin = 10.0 ** (gain / 40.0)
    w0 = 2.0 * math.pi * freq / fs
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    if filter_type == PEAKING:
        return ([1.0 + alpha * a_lin, -2.0 * cos_w0, 1.0 - alpha * a_lin],
                [1.0 + alpha / a_lin, -2.0 * cos_w0, 1.0 - alpha / a_lin])
    sqrt_alpha = 2.0 * math.sqrt(a_lin) * alpha
    if filter_type == LOW_SHELF:
        return ([a_lin * ((a_lin + 1) - (a_lin - 1) * cos_w0 + sqrt_alpha),
                 2.0 * a_lin * ((a_lin - 1) - (a_lin + 1) * cos_w0),
                 a_lin * ((a_lin + 1) - (a_lin - 1) * cos_w0 - sqrt_alpha)],
                [(a_lin + 1) + (a_lin - 1) * cos_w0 + sqrt_alpha,
                 -2.0 * ((a_lin - 1) + (a_lin + 1) * cos_w0),
                 (a_lin + 1) + (a_lin - 1) * cos_w0 - sqrt_alpha])
    if filter_type == HIGH_SHELF:
        return ([a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 + sqrt_alpha),
                 -2.0 * a_lin * ((a_lin - 1) + (a_lin + 1) * cos_w0),
                 a_lin * ((a_lin + 1) + (a_lin - 1) * cos_w0 - sqrt_alpha)],
                [(a_lin + 1) - (a_lin - 1) * cos_w0 + sqrt_alpha,
                 2.0 * ((a_lin - 1) - (a_lin + 1) * cos_w0),
                 (a_lin + 1) - (a_lin - 1) * cos_w0 - sqrt_alpha])
    raise UnsupportedEntryError(f'Unsupported filter type {filter_type}')


# --- solver ---------------------------------------------------------------------------------------------------------
#
# Each filter is held as three parameters: log(freq), gain and log(q). Working in logs keeps frequency and Q steps
# proportional to their size. The solver itself works on unbounded internal values, each mapped into its allowed range
# with a sine transform, so a parameter sitting at a limit never stalls the search.

class _Model:

    def __init__(self, grid: Grid, layout: Layout, kinds: list[str]):
        self.grid = grid
        self.kinds = kinds
        self.lower, self.upper = [], []
        for _ in kinds:
            self.lower += [math.log(layout.min_freq), MIN_GAIN, math.log(MIN_Q)]
            self.upper += [math.log(DEFAULT_MAX_FREQ), MAX_GAIN, math.log(MAX_Q)]

    def to_internal(self, params: list[float]) -> list[float]:
        internal = []
        for p, lo, hi in zip(params, self.lower, self.upper, strict=True):
            # start just inside the range: exactly on a limit the transform is flat and the value could never move
            ratio = min(max((p - lo) / (hi - lo), 1e-3), 1.0 - 1e-3)
            internal.append(math.asin(2.0 * ratio - 1.0))
        return internal

    def to_params(self, internal: list[float]) -> list[float]:
        return [lo + (hi - lo) * (math.sin(u) + 1.0) / 2.0
                for u, lo, hi in zip(internal, self.lower, self.upper, strict=True)]

    def one(self, idx: int, internal: list[float]) -> list[float]:
        """
        The response of a single filter, the only one a change to its own three parameters affects.
        """
        window = slice(3 * idx, 3 * idx + 3)
        log_f, gain, log_q = [
            lo + (hi - lo) * (math.sin(u) + 1.0) / 2.0
            for u, lo, hi in zip(internal[window], self.lower[window], self.upper[window], strict=True)
        ]
        return self.grid.filter_response(self.kinds[idx], math.exp(log_f), gain, math.exp(log_q))

    def to_filters(self, internal: list[float]) -> list[Filter]:
        params = self.to_params(internal)
        return [
            Filter(kind, round(math.exp(params[3 * i]), 2), round(params[3 * i + 1], 2),
                   round(math.exp(params[3 * i + 2]), 3))
            for i, kind in enumerate(self.kinds)
        ]


def _solve(grid: Grid, target: list[float], filters: list[Filter], layout: Layout, n_peaks: int) -> FitResult:
    kinds = ([LOW_SHELF] if layout.has_low_shelf else []) + [PEAKING] * n_peaks
    model = _Model(grid, layout, kinds)
    internal = model.to_internal(_initial_params(grid, target, filters, layout, n_peaks))
    internal = _levenberg_marquardt(model, internal, target)
    fitted = model.to_filters(internal)
    # report the error of the rounded values which are actually sent to the device
    error = max(abs(r - t) for r, t in zip(grid.response(fitted), target, strict=True))
    low_shelf = fitted[0] if layout.has_low_shelf else None
    peaks = fitted[1:] if layout.has_low_shelf else fitted
    return FitResult(low_shelf, peaks, error, False)


def _initial_params(grid: Grid, target: list[float], filters: list[Filter], layout: Layout,
                    n_peaks: int) -> list[float]:
    """
    Starts from the entry itself: its low shelves merged into one and its peaks as they are. Any spare peaking bands
    are then placed, one at a time, wherever the starting point is furthest from the target.
    """
    params = []
    start: list[Filter] = []
    if layout.has_low_shelf:
        shelves = [f for f in filters if f.filter_type == LOW_SHELF]
        total_gain = sum(f.gain for f in shelves)
        if shelves and total_gain:
            weights = [abs(f.gain) for f in shelves]
            log_f = sum(w * math.log(f.freq) for w, f in zip(weights, shelves, strict=True)) / sum(weights)
            shelf_gain = max(min(total_gain, MAX_GAIN), MIN_GAIN)
            start.append(Filter(LOW_SHELF, max(math.exp(log_f), layout.min_freq), shelf_gain, 0.707))
            excess = total_gain - shelf_gain
            if excess and n_peaks:
                # the shelf cannot provide it all, so a wide peak at the bottom of the range makes up the difference
                start.append(Filter(PEAKING, layout.min_freq * 1.5, max(min(excess, MAX_GAIN), MIN_GAIN), 0.5))
        else:
            start.append(Filter(LOW_SHELF, max(20.0, layout.min_freq), 0.0, 0.707))
    spare = n_peaks - (len(start) - (1 if layout.has_low_shelf else 0))
    peaks = sorted((f for f in filters if f.filter_type == PEAKING), key=lambda f: -abs(f.gain))[:spare]
    start += [Filter(PEAKING, max(f.freq, layout.min_freq), f.gain, f.q) for f in peaks]
    for f in start:
        params += [math.log(f.freq), f.gain, math.log(f.q)]

    current = grid.response(start)
    for _ in range(n_peaks - (len(start) - (1 if layout.has_low_shelf else 0))):
        peak = _best_spare_peak(grid, target, current)
        params += [math.log(peak.freq), peak.gain, math.log(peak.q)]
        current = [c + v for c, v in zip(current, grid.filter_response(*_unpack(peak)), strict=True)]
    return params


def _best_spare_peak(grid: Grid, target: list[float], current: list[float]) -> Filter:
    """
    Picks the single peaking filter which most reduces the remaining error. A few widths are tried at the point of
    largest error; a wide one matters when the low shelf alone cannot reach the boost an entry asks for.
    """
    residual = [t - c for t, c in zip(target, current, strict=True)]
    idx = max(range(len(residual)), key=lambda i: abs(residual[i]))
    gain = max(min(residual[idx], MAX_GAIN), MIN_GAIN)
    best, best_cost = None, math.inf
    for q in (0.3, 0.7, 1.5, 4.0):
        candidate = Filter(PEAKING, grid.freqs[idx], gain, q)
        response = grid.filter_response(*_unpack(candidate))
        cost = sum((r - v) ** 2 for r, v in zip(residual, response, strict=True))
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def _unpack(f: Filter) -> tuple[str, float, float, float]:
    return f.filter_type, f.freq, f.gain, f.q


def _levenberg_marquardt(model: _Model, internal: list[float], target: list[float]) -> list[float]:
    n_filters = len(model.kinds)
    parts = [model.one(i, internal) for i in range(n_filters)]

    def residuals(responses: list[list[float]]) -> list[float]:
        return [sum(col) - t for *col, t in zip(*responses, target, strict=True)]

    res = residuals(parts)
    cost = sum(r * r for r in res)
    damping = INITIAL_DAMPING
    for _ in range(MAX_ITERATIONS):
        if max(abs(r) for r in res) <= EXACT_ENOUGH_DB:
            break
        jacobian = _jacobian(model, internal, parts)
        jtj = [[sum(a * b for a, b in zip(ci, cj, strict=True)) for cj in jacobian] for ci in jacobian]
        jtr = [sum(a * r for a, r in zip(ci, res, strict=True)) for ci in jacobian]
        improved = False
        while damping < 1e10:
            system = [row[:] for row in jtj]
            for i in range(len(system)):
                system[i][i] += damping * max(system[i][i], 1e-9)
            step = _solve_linear(system, [-v for v in jtr])
            trial = [u + s for u, s in zip(internal, step, strict=True)]
            trial_parts = [model.one(i, trial) for i in range(n_filters)]
            trial_res = residuals(trial_parts)
            trial_cost = sum(r * r for r in trial_res)
            if trial_cost < cost:
                reduction = (cost - trial_cost) / cost if cost else 0.0
                internal, parts, res, cost = trial, trial_parts, trial_res, trial_cost
                damping = max(damping / 3.0, 1e-7)
                improved = reduction > 1e-9
                break
            damping *= 4.0
        if not improved:
            break
    return internal


def _jacobian(model: _Model, internal: list[float], parts: list[list[float]]) -> list[list[float]]:
    """
    One column per parameter, by forward differences. Each parameter only moves its own filter, so only that
    filter's response is recomputed.
    """
    columns = []
    for p in range(len(internal)):
        nudged = internal[:]
        nudged[p] += FD_STEP
        moved = model.one(p // 3, nudged)
        base = parts[p // 3]
        columns.append([(m - b) / FD_STEP for m, b in zip(moved, base, strict=True)])
    return columns


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """
    Gaussian elimination with partial pivoting; the system is small (three unknowns per filter).
    """
    n = len(rhs)
    m = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-300:
            continue
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    solution = [0.0] * n
    for r in range(n - 1, -1, -1):
        if abs(m[r][r]) < 1e-300:
            continue
        solution[r] = (m[r][n] - sum(m[r][c] * solution[c] for c in range(r + 1, n))) / m[r][r]
    return solution
