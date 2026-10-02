# SMART-DS Acquisition Record

Authoritative source, method and integrity for the dataset this project ingests.
Every fact here was established by querying the source directly, not assumed.

## Dataset

| Field | Value |
|---|---|
| Dataset | SMART-DS (Synthetic Models for Advanced, Realistic Testing: Distribution Systems and Scenarios) |
| Version | **1.0**, confirmed by reading the `SMART-DS_version.txt` shipped inside each scenario folder, whose entire content is `1.0` |
| Official source | OEDI data lake, public S3: `https://oedi-data-lake.s3.amazonaws.com/SMART-DS/` |
| Documentation | `SMART-DS/v1.0/User_Guide/Readme.md` (60,380 bytes) and `Readme.pdf` |
| Access | **Unauthenticated public HTTPS.** No credentials, no registration, no mirror |
| Licence | **UNKNOWN.** Not stated in the User Guide excerpt retrieved. Must be confirmed before any redistribution or publication |
| Alternatives present | `SMART-DS/v0.9/` also exists in the bucket; **v1.0 was chosen** as the newer release |

No community or Kaggle mirror was used. Every byte came from
`oedi-data-lake.s3.amazonaws.com`.

## Region and subset selection

Regions in v1.0: `SAF` (Santa Fe), `GSO` (Greensboro), `SFO` (San Francisco
Bay Area, the 40-sub-region flagship), `AUS` (Austin). Confirmed by listing the
bucket.

**Selected: `AUS/P1U`, year 2018.** P1U is the smallest urban sub-region, so
the acquired subset is tractable while still being genuine distribution data.

Scenarios available under `2018/AUS/P1U/scenarios/` (15 of them, confirmed by
listing): `base_timeseries`, and `solar_{none,low,medium,high,extreme}_batteries_{none,low,high}_timeseries`.

| Configured value | Meaning |
|---|---|
| `dataset` | `SMART-DS` |
| `version` | `v1.0` |
| `year` | `2018` |
| `region` / `subregion` | `AUS` / `P1U` |
| `scenario` | `base_timeseries` (switchable to a solar+battery scenario) |
| `substation` / `feeder` | `p1uhs0_1247` / `p1uhs0_1247--p1udt12703` |

## Files acquired

359 files, 228.5 MiB, staged under `data/raw/smart_ds/` (git-ignored per
decisions.md D-016, so no large binary is committed).

| Content | Count | Notes |
|---|---|---|
| `profiles/*.csv` | 341 | One per loadshape referenced by the feeder |
| `User_Guide/Readme.md` | 1 | Authoritative documentation |
| `metrics.csv` | 1 | 96 feeder rows, 71 metrics per row |
| Feeder `.dss` files | 9 | Master, Loads, LoadShapes, Lines, LineCodes, Transformers, Buscoords, Capacitors |
| `analysis/Summary_data.csv` | 1 | Published peak-time aggregate |
| `solar_data/*.csv` | 1 | Header + full 35,040-row solar file |
| PV scenario `.dss` | 2 | `PVSystems.dss` (1,216 PV), `Storage.dss` (93 batteries) |
| `placements/*.json` | 2 | `battery_customer=L.json`, `ev_residential=L.json` |

## Integrity

A `DatasetManifest` is produced by `energy-intel data manifest`, recording for
every file: S3 key, byte size, and a **computed** SHA-256. No checksum is
*claimed* unless it was calculated or published by the source.

Keys carry the full `SMART-DS/v1.0/` prefix so each entry is fetchable from the
authoritative bucket, and `acquisition.source` is the full base URL rather than a
local path (D-054).

```
SMART-DS v1.0 | 359 files | 228.5 MiB | digest a4384881d6e3
```

> The digest changed from `067783021396` when remote keys were corrected to
> include the `SMART-DS/v1.0/` prefix (D-054). The file *bytes* never changed;
> the recorded locator did.

`manifest.digest()` is a SHA-256 over the deterministic body and **excludes the
acquisition timestamp**, so two acquisitions of identical bytes agree regardless
of when they happened. That is the value to compare across machines.

The manifest carries **no secrets**; the bucket is unauthenticated, so there are
none to record. A test asserts no secret-like key appears in the serialised
manifest.

## Acquisition method

Plain HTTPS `GET` against the S3 REST endpoint. Objects are placed at the same
relative paths they occupy in the bucket, so provenance locators read like the
authoritative layout:

```text
data/raw/smart_ds/v1.0/2018/AUS/P1U/scenarios/base_timeseries/
    opendss/p1uhs0_1247/p1uhs0_1247--p1udt12703/Loads.dss
```

Directory listings use the S3 `list-type=2` API. No third-party S3 client was
installed.

## Reproducing the acquisition

```powershell
# 1. Write the subset manifest (lists every object with size and role)
# 2. Fetch each object to its mirrored relative path
# 3. Verify with the computed digests
uv run energy-intel data manifest
```

Raw data is deliberately **not** committed. Re-fetching from the authoritative
source plus verifying digests is the reproducibility contract; committing 228
MiB per contributor would be worse.

## Raw versus processed

```text
data/raw/smart_ds/...      NEVER modified in place; a byte-for-byte copy of the source
data/processed/             (empty in Phase 3)
artifacts/data/            reports: manifest, schema, quality, balance, mapping
```

Raw data is git-ignored. The only transformation applied in Phase 3 is
reconstruction of demand from ratings and per-unit profiles, which is
deterministic and recorded as a `ProcessingStep` on every provenance chain.

## Known acquisition gaps

* The licence is **UNKNOWN** and must be resolved before publication.
* Only one sub-region, one year, and two scenarios were acquired. A wider
  acquisition is a Phase 3 follow-up, not a Phase 1-3 blocker.
* `load_data/*.parquet` (per-customer end-use breakdown) was **not** acquired:
  the files are 1-4 MiB each and thousands exist. Phase 3 follow-up; the User
  Guide documents 34 columns including heating, cooling, lighting, motors and
  appliance-level loads, which is valuable for flexible-load work later.
* Battery **dispatch cannot be acquired** because it does not exist in the
  dataset at all. This is a property of SMART-DS, not an acquisition failure.
  See `docs/gaps_report.md`.
