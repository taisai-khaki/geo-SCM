from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ATLAS_GRAPHQL_URL = "https://atlas.hks.harvard.edu/api/graphql"
DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
DEFAULT_START_YEAR = 2012
OECD_TIVA_QUERY_PREFIX = (
    "https://sdmx.oecd.org/sti-public/rest/data/"
    "OECD.STI.PIE,DSD_TIVA_MAINSH@DF_MAINSH,1.1/"
    "EXGR_DVA+EXGR_FVA+FEXGR_DVA+FFD_DVA.._T.W..A"
)


@dataclass(frozen=True)
class SourceFile:
    name: str
    out_name: str


ATLAS_REQUIRED_FILES: tuple[SourceFile, ...] = (
    SourceFile(name="hs92_country_year", out_name="atlas_hs92_country_year.csv"),
    SourceFile(
        name="hs92_country_country_year", out_name="atlas_hs92_country_country_year.csv"
    ),
)


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def build_oecd_tiva_url(start_year: int, end_year: int) -> str:
    return (
        f"{OECD_TIVA_QUERY_PREFIX}"
        f"?startPeriod={start_year}&endPeriod={end_year}&format=csvfile"
    )


def fetch_json(url: str, *, method: str = "GET", **kwargs) -> dict:
    if method.upper() == "GET":
        response = requests.get(url, timeout=300, **kwargs)
    else:
        response = requests.post(url, timeout=300, **kwargs)
    response.raise_for_status()
    return response.json()


def query_atlas_downloads_table() -> list[dict]:
    query = """
    query {
      downloadsTable {
        tableName
        dvFileId
        dvFileName
        dvPublicationDate
        yearMin
        yearMax
        repo
      }
    }
    """
    payload = {"query": query}
    obj = fetch_json(ATLAS_GRAPHQL_URL, method="POST", json=payload)
    if "errors" in obj:
        raise RuntimeError(f"Atlas GraphQL error: {obj['errors']}")
    return obj["data"]["downloadsTable"]


def query_atlas_location_country() -> pd.DataFrame:
    query = """
    query {
      locationCountry {
        countryId
        iso3Code
        iso2Code
        nameShortEn
      }
    }
    """
    payload = {"query": query}
    obj = fetch_json(ATLAS_GRAPHQL_URL, method="POST", json=payload)
    if "errors" in obj:
        raise RuntimeError(f"Atlas GraphQL error: {obj['errors']}")
    rows = obj["data"]["locationCountry"]
    df = pd.DataFrame(rows)
    df["country_id"] = (
        df["countryId"].astype(str).str.replace("country-", "", regex=False).astype(int)
    )
    df = df.rename(
        columns={
            "iso3Code": "country_iso3_code",
            "iso2Code": "country_iso2_code",
            "nameShortEn": "country_name",
        }
    )[["country_id", "country_iso3_code", "country_iso2_code", "country_name"]]
    return df


def build_atlas_file_map(download_rows: Iterable[dict]) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    for row in download_rows:
        table_name = row.get("tableName")
        if table_name:
            by_name[table_name] = row
    return by_name


def download_file(url: str, out_path: Path) -> None:
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def resolve_atlas_year_bounds(table_map: dict[str, dict]) -> tuple[int, int]:
    mins: list[int] = []
    maxes: list[int] = []
    for source in ATLAS_REQUIRED_FILES:
        row = table_map[source.name]
        year_min = row.get("yearMin")
        year_max = row.get("yearMax")
        if year_min is None or year_max is None:
            raise RuntimeError(f"Missing year bounds for Atlas table: {source.name}")
        mins.append(int(year_min))
        maxes.append(int(year_max))
    return max(mins), min(maxes)


def download_raw_sources(
    raw_dir: Path,
    metadata_dir: Path,
    force: bool,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
) -> dict:
    ensure_dirs(raw_dir, metadata_dir)
    downloads_table = query_atlas_downloads_table()
    (metadata_dir / "atlas_downloads_table.json").write_text(
        json.dumps(downloads_table, indent=2), encoding="utf-8"
    )

    table_map = build_atlas_file_map(downloads_table)
    atlas_source_manifest: list[dict] = []

    for source in ATLAS_REQUIRED_FILES:
        if source.name not in table_map:
            raise RuntimeError(f"Could not find Atlas table: {source.name}")

        row = table_map[source.name]
        file_id = row.get("dvFileId")
        if file_id is None:
            raise RuntimeError(f"Missing dvFileId for Atlas table: {source.name}")

        out_path = raw_dir / source.out_name
        if force or not out_path.exists():
            download_file(DATAVERSE_FILE_URL.format(file_id=file_id), out_path)

        atlas_source_manifest.append(
            {
                "table_name": source.name,
                "out_file": str(out_path),
                "dv_file_id": file_id,
                "dv_file_name": row.get("dvFileName"),
                "dv_publication_date": row.get("dvPublicationDate"),
                "year_min": row.get("yearMin"),
                "year_max": row.get("yearMax"),
                "repo": row.get("repo"),
            }
        )

    atlas_year_min, atlas_year_max = resolve_atlas_year_bounds(table_map)
    effective_end_year = atlas_year_max if end_year is None else min(end_year, atlas_year_max)
    if start_year > effective_end_year:
        raise ValueError(
            f"Invalid year range: start_year={start_year}, end_year={effective_end_year}"
        )

    oecd_url = build_oecd_tiva_url(start_year=start_year, end_year=effective_end_year)
    oecd_out = raw_dir / f"oecd_tiva_mainsh_{start_year}_{effective_end_year}_selected.csv"
    if force or not oecd_out.exists():
        download_file(oecd_url, oecd_out)

    source_manifest = {
        "atlas_sources": atlas_source_manifest,
        "year_window": {
            "requested_start_year": start_year,
            "requested_end_year": end_year,
            "effective_start_year": start_year,
            "effective_end_year": effective_end_year,
            "atlas_available_min_year": atlas_year_min,
            "atlas_available_max_year": atlas_year_max,
        },
        "oecd_tiva_source": {
            "url": oecd_url,
            "out_file": str(oecd_out),
        },
    }
    (metadata_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    return source_manifest


def compute_export_recovery_index(df_country_year: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df_country_year[df_country_year["year"].between(2015, 2017)]
        .groupby("country_iso3_code", as_index=False)["export_value"]
        .mean()
        .rename(columns={"export_value": "baseline_export_2015_2017"})
    )
    merged = df_country_year.merge(baseline, on="country_iso3_code", how="left")
    merged["export_recovery_index"] = (
        merged["export_value"] / merged["baseline_export_2015_2017"]
    )
    return merged


def compute_partner_diversification(df_bilateral: pd.DataFrame) -> pd.DataFrame:
    work = df_bilateral.copy()
    work = work[work["country_iso3_code"] != work["partner_iso3_code"]].copy()
    work["export_value"] = pd.to_numeric(work["export_value"], errors="coerce")
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")

    totals = work.groupby(["country_iso3_code", "year"], as_index=False)[
        "export_value"
    ].sum()
    totals = totals.rename(columns={"export_value": "total_exports_to_partners"})
    work = work.merge(totals, on=["country_iso3_code", "year"], how="left")

    work = work[work["total_exports_to_partners"] > 0].copy()
    work["export_share"] = work["export_value"] / work["total_exports_to_partners"]
    work["export_share_sq"] = work["export_share"] ** 2

    hhi = (
        work.groupby(["country_iso3_code", "year"], as_index=False)["export_share_sq"]
        .sum()
        .rename(columns={"export_share_sq": "partner_hhi_export"})
    )
    hhi = hhi.merge(totals, on=["country_iso3_code", "year"], how="left")
    hhi["partner_diversification_1_minus_hhi"] = 1.0 - hhi["partner_hhi_export"]
    return hhi


def transform_oecd_tiva(df_tiva: pd.DataFrame) -> pd.DataFrame:
    keep = df_tiva[
        (df_tiva["ACTIVITY"] == "_T")
        & (df_tiva["COUNTERPART_AREA"] == "W")
        & (df_tiva["FREQ"] == "A")
    ].copy()
    keep["TIME_PERIOD"] = pd.to_numeric(keep["TIME_PERIOD"], errors="coerce").astype(
        "Int64"
    )
    keep["OBS_VALUE"] = pd.to_numeric(keep["OBS_VALUE"], errors="coerce")

    pivot = keep.pivot_table(
        index=["REF_AREA", "TIME_PERIOD"],
        columns="MEASURE",
        values="OBS_VALUE",
        aggfunc="first",
    ).reset_index()

    pivot = pivot.rename(
        columns={
            "REF_AREA": "country_iso3_code",
            "TIME_PERIOD": "year",
            "EXGR_DVA": "tiva_exgr_dva_share",
            "EXGR_FVA": "tiva_exgr_fva_share",
            "FEXGR_DVA": "tiva_fexgr_dva_share",
            "FFD_DVA": "tiva_ffd_dva_share",
        }
    )
    pivot["country_iso3_code"] = pivot["country_iso3_code"].astype(str)
    return pivot


def build_final_databank(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    source_manifest: dict,
) -> dict[str, int]:
    ensure_dirs(processed_dir, reports_dir)

    country_year_path = raw_dir / "atlas_hs92_country_year.csv"
    bilateral_path = raw_dir / "atlas_hs92_country_country_year.csv"
    year_window = source_manifest.get("year_window", {})
    start_year = int(year_window.get("effective_start_year", DEFAULT_START_YEAR))
    end_year = int(year_window.get("effective_end_year", 2022))
    span_label = f"{start_year}_{end_year}"

    tiva_path = Path(source_manifest["oecd_tiva_source"]["out_file"])

    df_country_year = pd.read_csv(country_year_path)
    df_bilateral = pd.read_csv(bilateral_path)
    df_tiva = pd.read_csv(tiva_path)
    df_locations = query_atlas_location_country()

    df_country_year = df_country_year[df_country_year["year"].between(start_year, end_year)].copy()
    df_bilateral = df_bilateral[df_bilateral["year"].between(start_year, end_year)].copy()
    df_country_year = df_country_year.merge(
        df_locations, on=["country_id", "country_iso3_code"], how="left"
    )

    df_country_year = compute_export_recovery_index(df_country_year)
    df_div = compute_partner_diversification(df_bilateral)
    df_tiva_wide = transform_oecd_tiva(df_tiva)

    databank = df_country_year.merge(
        df_div, on=["country_iso3_code", "year"], how="left"
    ).merge(df_tiva_wide, on=["country_iso3_code", "year"], how="left")

    databank = databank.sort_values(["country_iso3_code", "year"]).reset_index(drop=True)
    databank["delta_tiva_fexgr_dva_share"] = databank.groupby("country_iso3_code")[
        "tiva_fexgr_dva_share"
    ].diff()
    databank["delta_tiva_ffd_dva_share"] = databank.groupby("country_iso3_code")[
        "tiva_ffd_dva_share"
    ].diff()

    source_country_year_out = processed_dir / f"source_atlas_country_year_{span_label}.csv"
    source_bilateral_out = (
        processed_dir / f"source_atlas_country_country_year_{span_label}.csv"
    )
    source_tiva_out = processed_dir / f"source_oecd_tiva_mainsh_{span_label}_selected.csv"
    databank_out = processed_dir / f"databank_country_year_{span_label}.csv"

    df_country_year.to_csv(source_country_year_out, index=False)
    df_bilateral.to_csv(source_bilateral_out, index=False)
    df_tiva.to_csv(source_tiva_out, index=False)
    databank.to_csv(databank_out, index=False)

    metrics = {
        "start_year": start_year,
        "end_year": end_year,
        "country_year_rows": int(len(df_country_year)),
        "bilateral_rows": int(len(df_bilateral)),
        "tiva_rows": int(len(df_tiva)),
        "databank_rows": int(len(databank)),
        "databank_countries": int(databank["country_iso3_code"].nunique()),
        "databank_years": int(databank["year"].nunique()),
    }

    tiva_years = pd.to_numeric(df_tiva.get("TIME_PERIOD"), errors="coerce")
    tiva_year_min = int(tiva_years.min()) if tiva_years.notna().any() else None
    tiva_year_max = int(tiva_years.max()) if tiva_years.notna().any() else None

    quality = {
        "metrics": metrics,
        "missing_counts": {
            k: int(v)
            for k, v in databank[
                [
                    "eci",
                    "coi",
                    "export_recovery_index",
                    "partner_diversification_1_minus_hhi",
                    "tiva_fexgr_dva_share",
                    "delta_tiva_fexgr_dva_share",
                ]
            ]
            .isna()
            .sum()
            .to_dict()
            .items()
        },
        "source_coverage": {
            "atlas_country_year_min": int(df_country_year["year"].min()),
            "atlas_country_year_max": int(df_country_year["year"].max()),
            "atlas_bilateral_min": int(df_bilateral["year"].min()),
            "atlas_bilateral_max": int(df_bilateral["year"].max()),
            "tiva_min_year": tiva_year_min,
            "tiva_max_year": tiva_year_max,
            "post_2022_available_atlas": bool(df_country_year["year"].max() > 2022),
            "post_2022_available_tiva": bool(tiva_year_max is not None and tiva_year_max > 2022),
        },
    }
    (reports_dir / "data_quality_summary.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    dictionary_lines = [
        "column,description",
        "country_id,Numeric country code from Atlas (UN M49 aligned).",
        "country_iso3_code,ISO3 country code.",
        "country_iso2_code,ISO2 country code from Atlas location table.",
        "country_name,Country short name from Atlas location table.",
        "year,Calendar year.",
        "export_value,Total exports (HS92) from Atlas source.",
        "import_value,Total imports (HS92) from Atlas source.",
        "eci,Economic Complexity Index from Atlas.",
        "coi,Complexity Outlook Index from Atlas.",
        "diversity,Country export diversity from Atlas.",
        "growth_proj,Growth projection field from Atlas rankings source.",
        "baseline_export_2015_2017,Average exports over 2015-2017.",
        "export_recovery_index,export_value / baseline_export_2015_2017.",
        "total_exports_to_partners,Sum of bilateral exports by year.",
        "partner_hhi_export,HHI based on export partner shares.",
        "partner_diversification_1_minus_hhi,1 - partner_hhi_export.",
        "tiva_exgr_dva_share,TiVA EXGR_DVA share measure.",
        "tiva_exgr_fva_share,TiVA EXGR_FVA share measure.",
        "tiva_fexgr_dva_share,TiVA FEXGR_DVA share measure (forward-linkage candidate).",
        "tiva_ffd_dva_share,TiVA FFD_DVA share measure (forward-linkage alternative).",
        "delta_tiva_fexgr_dva_share,Year-over-year change in tiva_fexgr_dva_share.",
        "delta_tiva_ffd_dva_share,Year-over-year change in tiva_ffd_dva_share.",
    ]
    (reports_dir / "data_dictionary.csv").write_text(
        "\n".join(dictionary_lines), encoding="utf-8"
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build country-level databank from Atlas, TiVA, and trade sources."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory. Defaults to repo root.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw files even if present.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First year to include in the databank.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to include. If omitted, uses latest Atlas-available year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    reports_dir = base_dir / "reports"
    metadata_dir = base_dir / "metadata"

    print(f"Using base dir: {base_dir}")
    source_manifest = download_raw_sources(
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        force=args.force_download,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    metrics = build_final_databank(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        source_manifest=source_manifest,
    )
    print("Build complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
