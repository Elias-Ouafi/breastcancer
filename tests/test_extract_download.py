"""How a DBT download counts success and failure.

``nbia.downloadSeries`` swallows every exception and returns normally, so a failed
series used to be counted as downloaded and marked present. These tests replace the two
nbia calls and check that only a folder actually written counts. Needs tcia_utils,
which ExtractData imports; CI does not install it, the local suite does.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("tcia_utils", reason="TCIA client not installed")

import ExtractData  # noqa: E402
import http_timeouts  # noqa: E402


def fake_nbia(monkeypatch, series_by_patient, writes):
    def get_series(collection, patientId):
        return series_by_patient.get(patientId)

    def download_series(series, path):
        uid = series[0]["SeriesInstanceUID"]
        if uid in writes:
            os.makedirs(os.path.join(path, uid))
            with open(os.path.join(path, uid, "1-1.dcm"), "wb") as fh:
                fh.write(b"\0" * 10)
        # otherwise: return normally, exactly like tcia_utils after a caught exception

    monkeypatch.setattr(ExtractData.nbia, "getSeries", get_series)
    monkeypatch.setattr(ExtractData.nbia, "downloadSeries", download_series)


def test_importing_the_module_installs_the_http_timeouts():
    assert isinstance(ExtractData.nbia.requests, http_timeouts.RequestsWithTimeout)
    assert ExtractData.nbia.requests.timeout == (http_timeouts.CONNECT_TIMEOUT_S,
                                                 http_timeouts.READ_TIMEOUT_S)


def test_a_series_whose_folder_was_not_written_is_not_counted(tmp_path, monkeypatch):
    fake_nbia(monkeypatch, {"P1": [{"SeriesInstanceUID": "1.1"}, {"SeriesInstanceUID": "1.2"}]},
              writes={"1.1"})
    assert ExtractData.download_dbt_series_for(["P1"], download_dir=str(tmp_path)) == 1


def test_failures_raise_when_asked_after_the_rest_is_fetched(tmp_path, monkeypatch):
    fake_nbia(monkeypatch, {"P1": [{"SeriesInstanceUID": "1.1"}],
                            "P2": [{"SeriesInstanceUID": "2.1"}]}, writes={"2.1"})
    with pytest.raises(ExtractData.DownloadIncomplete) as err:
        ExtractData.download_dbt_series_for(["P1", "P2"], download_dir=str(tmp_path),
                                            raise_on_failure=True)
    assert err.value.failed == ["1.1"]
    assert os.path.isdir(tmp_path / "2.1")  # the failure did not stop the next patient


def test_a_patient_whose_series_list_failed_is_a_failure(tmp_path, monkeypatch):
    fake_nbia(monkeypatch, {}, writes=set())  # getSeries returns None
    with pytest.raises(ExtractData.DownloadIncomplete) as err:
        ExtractData.download_dbt_series_for(["P9"], download_dir=str(tmp_path),
                                            raise_on_failure=True)
    assert err.value.failed == ["P9"]


def test_the_normal_sample_is_the_same_one_the_download_uses(monkeypatch):
    import TransformData

    status = {f"P{i:03d}": "normal" for i in range(20)} | {"PX": "cancer"}
    monkeypatch.setattr(TransformData, "dbt_patient_status", lambda labels: status)
    chosen, available = ExtractData.choose_normal_patients(["labels.csv"], 5, seed=0)
    again, _ = ExtractData.choose_normal_patients(["labels.csv"], 5, seed=0)

    assert available == 20 and len(chosen) == 5 and chosen == again
    assert "PX" not in chosen
    fetched = []
    monkeypatch.setattr(ExtractData, "download_dbt_series_for",
                        lambda patients, **kw: fetched.extend(patients) or 0)
    ExtractData.download_normal_dbt_series(["labels.csv"], max_patients=5, seed=0)
    assert fetched == chosen
