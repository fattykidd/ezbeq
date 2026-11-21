import json
import os

import pytest
import yaml
from conftest import HyperionSpyConfig
from werkzeug import Response

from ezbeq import main, peqfit
from ezbeq.hyperion import (
    DEFAULT_PEAK_BANDS,
    SLOT_ID,
    BandConfig,
    Hyperion,
    resolve_url,
)

SPEAKERS = [
    {'id': 4, 'speaker': 'Sub 1', 'type': 'SW', 'customName': ''},
    {'id': 15, 'speaker': 'Sub 2', 'type': 'SW', 'customName': 'Front Sub'},
    {'id': 16, 'speaker': 'Sub 3', 'type': 'SW', 'customName': ''},
]


def test_devices(hyperion_client):
    r = hyperion_client.get("/api/1/devices")

    assert r.status_code == 200
    assert r.json == {
        'type': 'hyperion',
        'name': 'master',
        'slots': [
            {
                'id': SLOT_ID,
                'last': 'Empty',
                'active': True,
                'author': None
            }
        ]
    }


def test_load_writes_every_band_to_every_speaker(hyperion_client, httpserver):
    writes = expect_band_writes(httpserver)

    r = load(hyperion_client)

    assert r.status_code == 200
    assert r.json['slots'][0]['last'] == 'Alien Resurrection'
    assert r.json['slots'][0]['author'] == 'aron7awol'
    bands = [1] + DEFAULT_PEAK_BANDS
    assert len(writes) == len(bands) * 2
    for speaker in ('4', '15'):
        assert sorted(w['eqId'] for w in writes if w['speakerId'] == speaker) == bands
    assert all(w['enabled'] is True for w in writes)
    assert 8 not in {w['eqId'] for w in writes}, 'the high shelf band must never be touched'


def test_loaded_bands_reproduce_the_entry(hyperion_client, httpserver):
    writes = expect_band_writes(httpserver)

    load(hyperion_client)

    # five stacked 33 Hz shelves are 25 dB, more than one band can hold, so this entry has to be fitted
    loaded = [as_filter(w) for w in writes if w['speakerId'] == '4']
    entry = [peqfit.Filter(peqfit.LOW_SHELF, 33.0, 5.0, 0.9)] * 5
    grid = peqfit.Grid(5.0, 300.0, peqfit.DEFAULT_FS)
    error = max(abs(a - b) for a, b in zip(grid.response(loaded), grid.response(entry), strict=True))
    assert error < 1.0


def test_cuts_are_written_before_boosts(hyperion_client, httpserver):
    writes = expect_band_writes(httpserver)

    load(hyperion_client)

    gains = [w['gain'] for w in writes]
    first_boost = next(i for i, g in enumerate(gains) if g > 0)
    assert all(g > 0 for g in gains[first_boost:])


def test_patch_loads_known_entry(hyperion_client, httpserver):
    writes = expect_band_writes(httpserver)

    r = hyperion_client.patch("/api/3/devices/master",
                              data=json.dumps({'slots': [{'id': SLOT_ID, 'entry': '123456_0'}]}),
                              content_type='application/json')

    assert r.status_code == 200
    assert r.json['slots'][0]['last'] == 'Alien Resurrection'
    assert writes


def test_patch_rejects_unknown_entry(hyperion_client):
    r = hyperion_client.patch("/api/3/devices/master",
                              data=json.dumps({'slots': [{'id': SLOT_ID, 'entry': 'missing'}]}),
                              content_type='application/json')

    assert r.status_code == 400
    assert r.json == {'message': 'Update failed : Unknown catalogue entry missing'}


def test_clear_flattens_every_configured_band(hyperion_client, httpserver):
    writes = expect_band_writes(httpserver)
    assert load(hyperion_client).status_code == 200
    writes.clear()

    r = hyperion_client.delete(f"/api/1/devices/master/filter/{SLOT_ID}")

    assert r.status_code == 200
    assert r.json['slots'][0]['last'] == 'Empty'
    assert len(writes) == (1 + len(DEFAULT_PEAK_BANDS)) * 2
    assert all(w['gain'] == 0.0 for w in writes)


def test_activate_slot(hyperion_client):
    r = hyperion_client.put(f"/api/1/devices/master/config/{SLOT_ID}/active")

    assert r.status_code == 200
    assert r.json['slots'][0]['active'] is True


def test_http_error_marks_the_slot_as_failed(hyperion_client, httpserver):
    httpserver.expect_request("/v1/speaker/eq", method="POST").respond_with_data('boom', status=500)

    r = load(hyperion_client)

    assert r.status_code == 500
    assert r.json['slots'][0]['last'] == 'ERROR'


def test_non_json_response_marks_the_slot_as_failed(hyperion_client, httpserver):
    httpserver.expect_request("/v1/speaker/eq", method="POST").respond_with_data('ok', content_type='text/plain')

    r = load(hyperion_client)

    assert r.status_code == 500
    assert r.json['slots'][0]['last'] == 'ERROR'


def test_wrong_band_in_response_marks_the_slot_as_failed(hyperion_client, httpserver):
    httpserver.expect_request("/v1/speaker/eq", method="POST").respond_with_json({'eqId': 99})

    r = load(hyperion_client)

    assert r.status_code == 500
    assert r.json['slots'][0]['last'] == 'ERROR'


def test_poor_fit_is_refused_without_touching_the_device(httpserver, tmp_path):
    writes = expect_band_writes(httpserver)
    client = make_client(httpserver, tmp_path, maxFitError=0.001)

    r = load(client)

    assert r.status_code == 400
    assert r.json['slots'][0]['last'] == 'Empty'
    assert writes == []


def test_speakers_can_be_named(httpserver, tmp_path):
    writes = expect_band_writes(httpserver)
    httpserver.expect_request("/v1/speakerstate", method="GET").respond_with_json({'speakers': SPEAKERS})
    client = make_client(httpserver, tmp_path, speakers=['sub 1', 'Front Sub', 16])

    r = load(client)

    assert r.status_code == 200
    assert {w['speakerId'] for w in writes} == {'4', '15', '16'}


def test_unknown_speaker_name_fails_the_load(httpserver, tmp_path):
    writes = expect_band_writes(httpserver)
    httpserver.expect_request("/v1/speakerstate", method="GET").respond_with_json({'speakers': SPEAKERS})
    client = make_client(httpserver, tmp_path, speakers=['Sub 9'])

    r = load(client)

    assert r.status_code == 500
    assert r.json['slots'][0]['last'] == 'ERROR'
    assert writes == []


def test_only_configured_bands_are_written(httpserver, tmp_path):
    writes = expect_band_writes(httpserver)
    client = make_client(httpserver, tmp_path, shelfBand=None, peakBands=[9, 10, 11, 12, 13, 14, 15, 16])

    r = load(client)

    assert r.status_code == 200
    assert {w['eqId'] for w in writes} == {9, 10, 11, 12, 13, 14, 15, 16}


def test_min_frequency_is_respected(httpserver, tmp_path):
    writes = expect_band_writes(httpserver)
    client = make_client(httpserver, tmp_path, minFrequency=18)

    r = load(client)

    assert r.status_code == 200
    assert min(w['frequency'] for w in writes) >= 18.0


@pytest.mark.parametrize('band_cfg,message', [
    ({'shelfBand': 2}, 'shelfBand must be 1'),
    ({'peakBands': [8]}, 'peakBands must be peaking bands'),
    ({'peakBands': [1]}, 'peakBands must be peaking bands'),
    ({'peakBands': [17]}, 'peakBands must be peaking bands'),
    ({'peakBands': [2, 2]}, 'duplicates'),
    ({'shelfBand': None, 'peakBands': []}, 'at least one band'),
])
def test_rejects_bad_band_config(band_cfg, message):
    with pytest.raises(ValueError, match=message):
        BandConfig(band_cfg)


def test_assign_fills_unused_bands_with_flat_filters():
    result = peqfit.FitResult(peqfit.Filter(peqfit.LOW_SHELF, 25.0, 6.0, 0.7),
                              [peqfit.Filter(peqfit.PEAKING, 30.0, -2.0, 3.0)], 0.0, True)

    writes = BandConfig({'peakBands': [9, 10]}).assign(result)

    assert writes[1] == peqfit.Filter(peqfit.LOW_SHELF, 25.0, 6.0, 0.7)
    assert writes[9] == peqfit.Filter(peqfit.PEAKING, 30.0, -2.0, 3.0)
    assert writes[10].gain == 0.0


def test_resolves_url_from_config():
    assert resolve_url({'ip': '192.0.2.10'}) == 'http://192.0.2.10:8080'
    assert resolve_url({'ip': '192.0.2.10:9000/'}) == 'http://192.0.2.10:9000'
    assert resolve_url({'ip': 'https://hyperion.example'}) == 'https://hyperion.example:8080'
    assert resolve_url({'url': 'http://hyperion.example:8080/'}) == 'http://hyperion.example:8080'
    with pytest.raises(ValueError, match='url or ip'):
        resolve_url({})


def test_rejects_missing_speakers(httpserver, tmp_path):
    with pytest.raises(ValueError, match='speakers'):
        main.create_app(HyperionSpyConfig(httpserver.host, httpserver.port, tmp_path, speakers=[]))


def test_example_config_creates_a_device(tmp_path):
    example = os.path.join(os.path.dirname(__file__), '..', 'examples', 'ezbeq_hyperion.yml')
    with open(example) as f:
        devices = yaml.safe_load(f)['devices']

    device = Hyperion('hyperion', str(tmp_path), devices['hyperion'], None, None)

    assert device.device_type == 'hyperion'


def load(client):
    return client.put(f"/api/1/devices/master/filter/{SLOT_ID}",
                      data=json.dumps({'entryId': '123456_0'}),
                      content_type='application/json')


def make_client(httpserver, tmp_path, **device_cfg):
    app, _ws = main.create_app(HyperionSpyConfig(httpserver.host, httpserver.port, tmp_path, **device_cfg))
    return app.test_client()


def expect_band_writes(httpserver) -> list[dict]:
    """
    Acts as the Hyperion's band endpoint, echoing each write back the way the real device does.
    """
    writes = []

    def handler(request):
        body = request.get_json()
        writes.append(body)
        echo = {k: body[k] for k in ('speakerId', 'eqId', 'frequency', 'gain', 'q')}
        return Response(json.dumps(echo), status=200, content_type='application/json')

    httpserver.expect_request("/v1/speaker/eq", method="POST").respond_with_handler(handler)
    return writes


def as_filter(write: dict) -> peqfit.Filter:
    kind = peqfit.LOW_SHELF if write['eqId'] == 1 else peqfit.PEAKING
    return peqfit.Filter(kind, write['frequency'], write['gain'], write['q'])
