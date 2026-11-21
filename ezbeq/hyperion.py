"""
AudioControl Hyperion APR-16 support.

Each Hyperion speaker has 16 parametric EQ bands: band 1 is a low shelf, band 8 a high shelf and the rest are peaking
filters. BEQ entries often stack several low shelves at different frequencies, which that layout cannot hold, so the
entry's response is fitted onto the bands ezbeq is allowed to use (see ezbeq.peqfit) and written to every configured
speaker through the Hyperion's REST API.
"""
import logging
import threading
from collections.abc import Callable

import requests

from ezbeq import peqfit
from ezbeq.apis.ws import WsServer
from ezbeq.catalogue import CatalogueEntry, CatalogueProvider
from ezbeq.device import (
    DeviceState,
    InvalidRequestError,
    PersistentDevice,
    SlotState,
    UnableToPatchDeviceError,
)

logger = logging.getLogger('ezbeq.hyperion')

SLOT_ID = 'Hyperion'
DEFAULT_PORT = 8080
LOW_SHELF_BAND = 1
HIGH_SHELF_BAND = 8
ALL_BANDS = range(1, 17)
DEFAULT_PEAK_BANDS = [b for b in ALL_BANDS if b not in (LOW_SHELF_BAND, HIGH_SHELF_BAND)]

# the values a band is reset to; any frequency would do as the gain is 0 dB
FLAT_SHELF = peqfit.Filter(peqfit.LOW_SHELF, 20.0, 0.0, 0.707)
FLAT_PEAK = peqfit.Filter(peqfit.PEAKING, 100.0, 0.0, 1.0)


class HyperionSlotState(SlotState):

    def __init__(self):
        super().__init__(SLOT_ID)


class HyperionState(DeviceState):

    def __init__(self, name: str):
        self.__name = name
        self.slot = HyperionSlotState()
        self.slot.active = True

    def serialise(self) -> dict:
        return {
            'type': 'hyperion',
            'name': self.__name,
            'slots': [self.slot.as_dict()]
        }


class Hyperion(PersistentDevice[HyperionState]):
    SUPPORTED_OPS = frozenset({'activate', 'load_filter', 'clear_filter'})
    FIXED_SLOT_ID = SLOT_ID

    def __init__(self, name: str, config_path: str, cfg: dict, ws_server: WsServer, catalogue: CatalogueProvider):
        super().__init__(config_path, name, ws_server)
        self.__catalogue = catalogue
        self.__api = HyperionApi(resolve_url(cfg), float(cfg.get('timeout', 10.0)))
        self.__bands = BandConfig(cfg)
        self.__layout = peqfit.Layout(
            has_low_shelf=self.__bands.shelf is not None,
            peaking_bands=len(self.__bands.peaks),
            min_freq=float(cfg.get('minFrequency', 5.0)),
            max_boost=float(cfg['maxBoost']) if cfg.get('maxBoost') is not None else None,
            fs=int(cfg.get('sampleRate', peqfit.DEFAULT_FS)),
        )
        self.__max_fit_error = float(cfg.get('maxFitError', 1.5))
        self.__speakers = cfg.get('speakers') or []
        if not self.__speakers:
            raise ValueError('Hyperion requires at least one entry in speakers')
        if not 1.0 <= self.__layout.min_freq <= 100.0:
            raise ValueError('Hyperion minFrequency must be between 1 and 100 Hz')
        self.__watcher: StateWatcher | None = None
        if cfg.get('autoclear', False):
            interval = float(cfg.get('autoclearInterval', 5.0))
            if interval <= 0:
                raise ValueError('Hyperion autoclearInterval must be greater than 0')
            watcher_api = HyperionApi(resolve_url(cfg), float(cfg.get('timeout', 10.0)))
            self.__watcher = StateWatcher(watcher_api, interval, self.__on_state_change)
            self.__watcher.start()

    def _load_initial_state(self) -> HyperionState:
        return HyperionState(self.name)

    def _merge_state(self, loaded: HyperionState, cached: dict) -> HyperionState:
        for slot in cached.get('slots', []):
            if slot.get('id') == SLOT_ID:
                loaded.slot.merge_with(slot)
        loaded.slot.active = True
        return loaded

    @property
    def device_type(self) -> str:
        return 'hyperion'

    def update(self, params: dict) -> bool:
        any_update = False
        for slot in params.get('slots', []):
            if slot.get('id') == SLOT_ID and 'entry' in slot:
                if slot['entry']:
                    match = self.__catalogue.find(slot['entry'])
                    if not match:
                        raise UnableToPatchDeviceError(f'Unknown catalogue entry {slot["entry"]}', True)
                    self.load_filter(SLOT_ID, match)
                else:
                    self.clear_filter(SLOT_ID)
                any_update = True
        return any_update

    def activate(self, slot: str) -> None:
        self.__check_slot(slot)

        def __do_it():
            self._current_state.slot.active = True

        self._hydrate_cache_broadcast(__do_it)

    def load_filter(self, slot: str, entry: CatalogueEntry, mv_adjust: float = 0.0) -> None:
        self.__check_slot(slot)
        try:
            result = peqfit.fit(entry.filters, self.__layout)
        except peqfit.UnsupportedEntryError as e:
            raise InvalidRequestError(f'Unable to load {entry.formatted_title}: {e}') from e
        if result.max_error_db > self.__max_fit_error:
            raise InvalidRequestError(
                f'{entry.formatted_title} cannot be reproduced within {self.__max_fit_error} dB on the configured '
                f'bands (best fit is {result.max_error_db:.2f} dB out)'
            )
        logger.info(f'Loading {entry.formatted_title}: {"exact" if result.exact else "fitted"}, '
                    f'{len(result.filters)} bands, max error {result.max_error_db:.2f} dB')
        writes = self.__bands.assign(result)
        self._hydrate_cache_broadcast(lambda: self.__write(writes, entry.formatted_title, entry.author))

    def clear_filter(self, slot: str) -> None:
        self.__check_slot(slot)
        writes = self.__bands.assign(None)

        def __do_it():
            self.__write(writes, None, None)
            self._current_state.slot.active = True

        self._hydrate_cache_broadcast(__do_it)

    def __write(self, writes: dict[int, peqfit.Filter], title: str | None, author: str | None) -> None:
        try:
            speaker_ids = self.__api.resolve_speakers(self.__speakers)
            # cuts before boosts, so a partly applied change never adds more boost than the finished one
            ordered = sorted(writes.items(), key=lambda kv: kv[1].gain > 0)
            for band, f in ordered:
                for speaker_id in speaker_ids:
                    self.__api.set_band(speaker_id, band, f)
        except Exception:
            self._current_state.slot.last = 'ERROR'
            self._current_state.slot.last_author = None
            raise
        if title:
            self._current_state.slot.last = title
            self._current_state.slot.last_author = author
        else:
            self._current_state.slot.clear()

    def __on_state_change(self, reason: str) -> None:
        if self.state().slot.last == 'Empty':
            logger.debug(f'Hyperion {reason} changed, no BEQ loaded')
            return
        logger.info(f'Clearing BEQ as the Hyperion {reason} changed')
        try:
            self.clear_filter(SLOT_ID)
        except Exception:
            logger.exception('Unable to clear BEQ from the Hyperion')

    def stop(self) -> None:
        """
        Stops watching the Hyperion, if autoclear is enabled.
        """
        if self.__watcher:
            self.__watcher.stop()

    def load_biquads(self, slot: str, overwrite: bool, inputs: list[int], outputs: list[int],
                     biquads: list[dict]) -> None:
        raise NotImplementedError()

    def send_commands(self, slot: str, inputs: list[int], outputs: list[int], commands: list[str]) -> None:
        raise NotImplementedError()

    def mute(self, slot: str | None, channel: int | None) -> None:
        raise NotImplementedError()

    def unmute(self, slot: str | None, channel: int | None) -> None:
        raise NotImplementedError()

    def set_gain(self, slot: str | None, channel: int | None, gain: float) -> None:
        raise NotImplementedError()

    def levels(self) -> dict:
        return {}

    @staticmethod
    def __check_slot(slot: str) -> None:
        if slot != SLOT_ID:
            raise InvalidRequestError(f'Unknown Hyperion slot {slot}')


class BandConfig:
    """
    The bands ezbeq may change on each speaker.
    """

    def __init__(self, cfg: dict):
        shelf = cfg.get('shelfBand', LOW_SHELF_BAND)
        self.shelf: int | None = int(shelf) if shelf is not None else None
        self.peaks: list[int] = [int(b) for b in cfg.get('peakBands', DEFAULT_PEAK_BANDS)]
        if self.shelf is not None and self.shelf != LOW_SHELF_BAND:
            raise ValueError(f'Hyperion shelfBand must be {LOW_SHELF_BAND} (the low shelf) or null')
        bad = [b for b in self.peaks if b not in ALL_BANDS or b in (LOW_SHELF_BAND, HIGH_SHELF_BAND)]
        if bad:
            raise ValueError(f'Hyperion peakBands must be peaking bands (2-7, 9-16), not {bad}')
        if len(set(self.peaks)) != len(self.peaks):
            raise ValueError('Hyperion peakBands contains duplicates')
        if self.shelf is None and not self.peaks:
            raise ValueError('Hyperion needs at least one band to load filters into')

    def assign(self, result: peqfit.FitResult | None) -> dict[int, peqfit.Filter]:
        """
        Maps a fit onto band numbers; every configured band is included so that anything left over from a previous
        entry is cleared. With no result, every configured band is set flat.
        """
        writes = {}
        if self.shelf is not None:
            writes[self.shelf] = result.low_shelf if result and result.low_shelf else FLAT_SHELF
        peaks = result.peaks if result else []
        for idx, band in enumerate(self.peaks):
            writes[band] = peaks[idx] if idx < len(peaks) else FLAT_PEAK
        return writes


class StateWatcher:
    """
    Polls the Hyperion's power state and active input, reporting each change. The first reading only sets the
    baseline, and while the Hyperion cannot be reached the last known values are kept, so a change made in the meantime
    is still reported once it is back.
    """

    def __init__(self, api: 'HyperionApi', interval: float, on_change: Callable[[str], None]):
        self.__api = api
        self.__interval = interval
        self.__on_change = on_change
        self.__stopped = threading.Event()
        self.__thread = threading.Thread(target=self.__run, name='hyperion-autoclear', daemon=True)
        self.__power: bool | None = None
        self.__input: str | None = None

    def start(self) -> None:
        self.__thread.start()

    def stop(self) -> None:
        self.__stopped.set()
        self.__thread.join(timeout=self.__interval + 5)

    def __run(self) -> None:
        unreachable = False
        while not self.__stopped.is_set():
            try:
                self.poll()
                if unreachable:
                    logger.info('Hyperion is reachable again')
                unreachable = False
            except (requests.RequestException, HyperionError) as e:
                if not unreachable:
                    logger.warning(f'Unable to read the Hyperion state, will keep trying: {e}')
                unreachable = True
            self.__stopped.wait(self.__interval)

    def poll(self) -> None:
        power = self.__api.power_on()
        active_input = self.__api.active_input()
        changes = []
        if self.__power is not None and power != self.__power:
            changes.append('power state')
        if self.__input is not None and active_input != self.__input:
            changes.append('active input')
        self.__power, self.__input = power, active_input
        if changes:
            self.__on_change(' and '.join(changes))


class HyperionApi:

    def __init__(self, base_url: str, timeout: float):
        self.__base_url = base_url
        self.__timeout = timeout
        self.__session = requests.Session()

    def resolve_speakers(self, configured: list) -> list[str]:
        """
        Converts the configured speakers, given by id or by name, into speaker ids.
        """
        if all(isinstance(s, int) or str(s).isdigit() for s in configured):
            return [str(s) for s in configured]
        speakers = self.__get('/v1/speakerstate').get('speakers', [])
        ids = []
        for wanted in configured:
            if isinstance(wanted, int) or str(wanted).isdigit():
                ids.append(str(wanted))
                continue
            match = next((s for s in speakers if wanted.casefold() in (str(s.get('customName', '')).casefold(),
                                                                         str(s.get('speaker', '')).casefold())), None)
            if match is None:
                raise HyperionError(f'Hyperion has no speaker named {wanted}')
            ids.append(str(match['id']))
        return ids

    def power_on(self) -> bool:
        return str(self.__get('/v1/power').get('state')) == '1'

    def active_input(self) -> str:
        body = self.__get('/api/atlas/input/active')
        value = body.get('inputId', body.get('input'))
        if value is None:
            raise HyperionError('Hyperion did not report its active input')
        return str(value)

    def set_band(self, speaker_id: str, band: int, f: peqfit.Filter) -> None:
        payload = {
            'speakerId': speaker_id,
            'eqId': band,
            'frequency': f.freq,
            'gain': f.gain,
            'q': f.q,
            'enabled': True,
        }
        body = self.__post('/v1/speaker/eq', payload)
        if body.get('eqId') not in (None, band):
            raise HyperionError(f'Hyperion updated band {body.get("eqId")} instead of {band}')

    def __get(self, path: str) -> dict:
        rsp = self.__session.get(self.__base_url + path, timeout=self.__timeout)
        return self.__parse(rsp, path)

    def __post(self, path: str, payload: dict) -> dict:
        rsp = self.__session.post(self.__base_url + path, json=payload, timeout=self.__timeout)
        return self.__parse(rsp, path)

    @staticmethod
    def __parse(rsp: requests.Response, path: str) -> dict:
        if not rsp.ok:
            raise HyperionError(f'Hyperion returned HTTP {rsp.status_code} for {path}')
        try:
            return rsp.json()
        except ValueError as e:
            raise HyperionError(f'Hyperion returned a non-JSON response for {path}') from e


def resolve_url(cfg: dict) -> str:
    if 'url' in cfg:
        return str(cfg['url']).rstrip('/')
    ip = cfg.get('ip')
    if not ip:
        raise ValueError('Hyperion requires either url or ip')
    ip = str(ip).rstrip('/')
    if not ip.startswith(('http://', 'https://')):
        ip = f'http://{ip}'
    host_part = ip.split('://', 1)[1]
    return ip if ':' in host_part else f'{ip}:{DEFAULT_PORT}'


class HyperionError(Exception):
    pass
