"""Extract test fixtures from the real SMART-DS files.

Fixtures must preserve the *real* semantics of the source, so they are cut
mechanically from the downloaded dataset rather than hand-written. A
hand-written fixture would be a guess at SMART-DS's format, and a guess is
exactly what Phase 3 exists to eliminate.

Everything written here is a verbatim excerpt. Two fixtures are explicitly
truncated and say so in their own header:

* ``profile_res_kw_17316.csv`` - the first 96 values (4 hours) of a real
  35040-point profile. It is a genuine prefix of real data, so its maximum is
  NOT 1.0 (that normalisation holds over the whole year, not over a prefix).
  Tests using it must not assert per-unit normalisation.
* ``summary_data.csv`` - the real single-row published summary, unmodified.

Run:  python scripts/extract_fixtures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RAW = Path("data/raw/smart_ds/v1.0")
FIXTURES = Path("tests/data/fixtures")

SCENARIO = RAW / "2018/AUS/P1U/scenarios/base_timeseries"
FEEDER = SCENARIO / "opendss/p1uhs0_1247/p1uhs0_1247--p1udt12703"
PV_SCENARIO = (
    RAW / "2018/AUS/P1U/scenarios/solar_high_batteries_low_timeseries/opendss"
    "/p1uhs0_1247/p1uhs0_1247--p1udt12703"
)
PROFILES = RAW / "2018/AUS/P1U/profiles"


def write_lines(source: Path, target: Path, count: int, *, skip_blank: bool = True) -> None:
    """Copy the first ``count`` non-empty lines verbatim."""
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if skip_blank and not raw.strip():
            continue
        lines.append(raw)
        if len(lines) >= count:
            break
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_line_containing(source: Path, target: Path, needle: str) -> bool:
    """Append the first line of ``source`` containing ``needle``, verbatim.

    Used to guarantee the excerpt contains the Circuit's source bus. Without that
    line the truncated graph has no point of common coupling, and the honest
    mapper correctly refuses to invent one -- which would leave the mapper's
    success path untestable on a clean checkout.
    """
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if needle in raw and raw.strip():
            lines.append(raw)
            break
    if not lines:
        return False
    with target.open("a", encoding="utf-8") as handle:
        handle.write(lines[0] + "\n")
    return True


def circuit_source_bus(master: Path) -> str:
    """Read the source bus straight out of the real ``Circuit`` directive."""
    for raw in master.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.lower().startswith("new circuit"):
            for token in stripped.split():
                if token.lower().startswith("bus1="):
                    return token.split("=", 1)[1].split(".")[0].strip()
    raise SystemExit("no Circuit bus1 found in Master.dss")


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    write_lines(FEEDER / "Loads.dss", FIXTURES / "loads_excerpt.dss", 12)
    written.append("loads_excerpt.dss (12 real Load directives)")

    write_lines(FEEDER / "LoadShapes.dss", FIXTURES / "loadshapes_excerpt.dss", 4)
    written.append("loadshapes_excerpt.dss (4 real Loadshape directives)")

    write_lines(FEEDER / "Master.dss", FIXTURES / "master_excerpt.dss", 40)
    written.append("master_excerpt.dss (real circuit definition)")

    write_lines(FEEDER / "Lines.dss", FIXTURES / "lines_excerpt.dss", 10)
    source_bus = circuit_source_bus(FEEDER / "Master.dss")
    if append_line_containing(FEEDER / "Lines.dss", FIXTURES / "lines_excerpt.dss", source_bus):
        written.append(
            "lines_excerpt.dss (10 real Line directives plus the one incident to the "
            f"Circuit source bus {source_bus}, read from Master.dss)"
        )
    else:
        written.append("lines_excerpt.dss (10 real Line directives)")

    write_lines(FEEDER / "Buscoords.dss", FIXTURES / "buscoords_excerpt.dss", 10)
    written.append("buscoords_excerpt.dss (10 real coordinate rows)")

    write_lines(PV_SCENARIO / "Storage.dss", FIXTURES / "storage_excerpt.dss", 6)
    written.append("storage_excerpt.dss (6 real Storage directives)")

    write_lines(PV_SCENARIO / "PVSystems.dss", FIXTURES / "pvsystems_excerpt.dss", 6)
    written.append("pvsystems_excerpt.dss (6 real PVSystem directives)")

    write_lines(RAW / "User_Guide/Readme.md", FIXTURES / "user_guide_excerpt.md", 20)
    written.append("user_guide_excerpt.md (documentation excerpt)")

    # Summary_data.csv is copied whole: it is one row and is the published
    # reference the balance validation checks against.
    summary = FEEDER / "analysis/Summary_data.csv"
    (FIXTURES / "summary_data.csv").write_text(
        summary.read_text(encoding="utf-8"), encoding="utf-8"
    )
    written.append("summary_data.csv (complete, unmodified)")

    # A genuine 96-value prefix of a real profile.
    profile = PROFILES / "res_kw_17316_pu.csv"
    prefix = profile.read_text(encoding="utf-8").split()[:96]
    (FIXTURES / "profile_res_kw_17316.csv").write_text(
        "\n".join(prefix) + "\n", encoding="utf-8"
    )
    written.append(f"profile_res_kw_17316.csv (first 96 real values of 35040)")

    # Header of a real solar_data file, plus 4 real rows.
    solar = RAW / "2018/AUS/P1U/solar_data/AUS_30.3459_-97.8095_25_180_full.csv"
    write_lines(solar, FIXTURES / "solar_excerpt.csv", 5)
    written.append("solar_excerpt.csv (header + 4 real rows)")

    # A real placement JSON, truncated to a few feeders so it stays small.
    placement = RAW / "placements/AUS/P1U/battery_customer=L.json"
    payload = json.loads(placement.read_text(encoding="utf-8"))
    trimmed = dict(list(payload.items())[:2])
    (FIXTURES / "placement_battery_customer_L.json").write_text(
        json.dumps(trimmed, indent=2), encoding="utf-8"
    )
    written.append(
        f"placement_battery_customer_L.json (2 of {len(payload)} real keys)"
    )

    manifest = {
        "note": "Fixtures extracted mechanically from real SMART-DS v1.0 files.",
        "source_root": str(RAW),
        "feeder": "AUS/P1U 2018 base_timeseries p1uhs0_1247/p1uhs0_1247--p1udt12703",
        "files": written,
        "truncated": [
            "profile_res_kw_17316.csv is a 96-value prefix; its maximum is not 1.0",
            "placement_battery_customer_L.json keeps 2 of the real keys",
            "loadshapes_excerpt.dss references profiles not shipped as fixtures; "
            "tests must stub the profile directory",
            "lines_excerpt.dss keeps the first 10 real lines plus the line incident "
            "to the Circuit source bus, so the excerpt has a resolvable boundary",
        ],
    }
    (FIXTURES / "FIXTURE_PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    written.append("FIXTURE_PROVENANCE.json")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
