"""Fixtures for the Phase 3 ingestion tests.

Fixtures under ``tests/data/fixtures/`` were extracted mechanically from the real
SMART-DS v1.0 files by ``scripts/extract_fixtures.py``. They are verbatim
excerpts, not hand-written approximations -- see ``FIXTURE_PROVENANCE.json``.

:func:`fixture_layout` builds a real directory tree from those excerpts so the
adapter can run against genuine SMART-DS text without the 228 MiB dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from energy_intelligence.data.smartds import SmartDsLayout

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    if not FIXTURE_DIR.is_dir():
        pytest.skip(
            "fixtures missing; run 'uv run python scripts/extract_fixtures.py' with "
            "the dataset present under data/raw/smart_ds"
        )
    return FIXTURE_DIR


@pytest.fixture
def layout(tmp_path: Path, fixture_dir: Path) -> SmartDsLayout:
    """A real SMART-DS feeder tree built from extracted fixtures.

    The tree mirrors the authoritative nested layout exactly
    (``<root>/<version>/<year>/<region>/<subregion>/...``) so ``SmartDsLayout``
    resolves the same paths it would for a full download.
    """
    subregion = tmp_path / "v1.0" / "2018" / "AUS" / "P1U"
    scenario = subregion / "scenarios" / "base_timeseries"
    feeder = (
        scenario / "opendss" / "p1uhs0_1247" / "p1uhs0_1247--p1udt12703"
    )
    feeder.mkdir(parents=True)
    (feeder / "analysis").mkdir()
    (subregion / "profiles").mkdir(parents=True)

    shutil.copy(FIXTURE_DIR / "loads_excerpt.dss", feeder / "Loads.dss")
    shutil.copy(FIXTURE_DIR / "loadshapes_excerpt.dss", feeder / "LoadShapes.dss")
    shutil.copy(FIXTURE_DIR / "master_excerpt.dss", feeder / "Master.dss")
    shutil.copy(FIXTURE_DIR / "lines_excerpt.dss", feeder / "Lines.dss")
    shutil.copy(FIXTURE_DIR / "buscoords_excerpt.dss", feeder / "Buscoords.dss")
    shutil.copy(
        FIXTURE_DIR / "summary_data.csv", feeder / "analysis" / "Summary_data.csv"
    )
    shutil.copy(
        FIXTURE_DIR / "profile_res_kw_17316.csv",
        subregion / "profiles" / "res_kw_17316_pu.csv",
    )

    return SmartDsLayout(
        root=tmp_path,
        version="v1.0",
        year="2018",
        region="AUS",
        subregion="P1U",
        scenario="base_timeseries",
        substation="p1uhs0_1247",
        feeder="p1uhs0_1247--p1udt12703",
    )


@pytest.fixture
def real_dataset_available() -> bool:
    """Whether the full dataset is present locally.

    Integration tests that need it skip rather than fail, so the suite is
    runnable on a clean checkout that has not acquired the data.
    """
    return (
        Path("data/raw/smart_ds/v1.0/2018/AUS/P1U/profiles").is_dir()
        and Path(
            "data/raw/smart_ds/v1.0/2018/AUS/P1U/scenarios/base_timeseries"
            "/opendss/p1uhs0_1247/p1uhs0_1247--p1udt12703/Loads.dss"
        ).is_file()
    )
