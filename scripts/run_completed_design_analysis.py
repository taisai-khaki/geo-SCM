from __future__ import annotations

"""Complete the pre-specified redesign tasks for the geo-SCM study.

This script leaves the earlier capability-conversion package unchanged.  It
adds a second, auditable package that (1) freezes historical World Bank income
groups and ECI/COI regimes before the shock, (2) creates an HS6 tariff-weighted
US-market diversion exposure, (3) runs corrected-outcome and formal
heterogeneity tests, and (4) applies one BH correction over every reported
confirmatory and exploratory coefficient/contrast test.
"""

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy import stats

from run_capability_conversion_analysis import (
    BASELINE_END,
    BASELINE_START,
    H6_TERMS,
    add_pre_control_year_terms,
    add_linear_did_terms,
    bh_fdr_adjust,
    contrast_statistics,
    equivalence_and_mde,
    fit_twfe,
    freeze_pre_shock_constructs,
    h6_test_rows,
    make_pretrend_adjusted_recovery,
    prepare_piecewise_data,
    select_h6_threshold,
    wild_bootstrap_threshold_distribution,
    wild_cluster_bootstrap_contrast,
    winsorize,
    zscore,
)


POLICY_START = 2018
BOOTSTRAP_DEFAULT = 999
SOURCE_DIR_NAME = "capability_redesign_sources"

LIST_SPECS = {
    "list1": {
        "file": "ustr_2018_list1_notice.pdf",
        "pages": range(5, 10),
        "expected_lines": 818,
        "source": "83 FR 28710 (June 20, 2018)",
    },
    "list2": {
        "file": "ustr_2018_list2_notice.pdf",
        "pages": range(4, 6),
        "expected_lines": 279,
        "source": "83 FR 40823 (August 16, 2018)",
    },
    "list3": {
        "file": "ustr_2018_list3_notice.pdf",
        "pages": range(4, 30),
        "expected_lines": 5733,
        "source": "83 FR 47974 (September 21, 2018), with later amendments",
    },
    "list4a": {
        "file": "ustr_2019_list4_original_notice.pdf",
        "pages": range(4, 27),
        "expected_lines": 3233,
        "source": "84 FR 43304 (August 20, 2019), Annex A only",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_hts8(value: str) -> str:
    return value.replace(".", "")


def extract_tariff_lines(path: Path, pages: Iterable[int]) -> set[str]:
    """Extract formal HTS8 entries from the Annex pages in the USTR notice."""

    reader = PdfReader(str(path))
    result: set[str] = set()
    pattern = re.compile(r"(?<!\d)(\d{4}\.\d{2}\.\d{2})(?!\d)")
    for page_number in pages:
        text = reader.pages[page_number - 1].extract_text(extraction_mode="layout") or ""
        for value in pattern.findall(text):
            code = clean_hts8(value)
            # Chapter 98/99 entries are customs provisions, not BACI products.
            if not code.startswith(("98", "99")):
                result.add(code)
    return result


def build_tariff_schedule(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    line_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for list_name, spec in LIST_SPECS.items():
        source_path = source_dir / str(spec["file"])
        codes = extract_tariff_lines(source_path, spec["pages"])
        for hts8 in sorted(codes):
            line_rows.append(
                {
                    "tariff_list": list_name,
                    "hts8": hts8,
                    "hs6": hts8[:6],
                    "source_notice": spec["source"],
                }
            )
        audit_rows.append(
            {
                "tariff_list": list_name,
                "expected_official_hts8_line_count": int(spec["expected_lines"]),
                "extracted_hts8_line_count_after_chapter_98_99_exclusion": len(codes),
                "difference_from_official_count": len(codes) - int(spec["expected_lines"]),
                "source_file": str(spec["file"]),
                "source_sha256": file_hash(source_path),
                "source_notice": spec["source"],
            }
        )

    lines = pd.DataFrame(line_rows)
    membership = (
        lines.assign(member=1)
        .pivot_table(index="hs6", columns="tariff_list", values="member", aggfunc="max", fill_value=0)
        .reset_index()
    )
    for list_name in LIST_SPECS:
        if list_name not in membership:
            membership[list_name] = 0
    membership["number_of_lists_at_hs6"] = membership[list(LIST_SPECS)].sum(axis=1)
    membership["multi_list_hs6"] = (membership["number_of_lists_at_hs6"] > 1).astype(int)

    # HS6 flows cannot distinguish partial HTS8 coverage.  Assign a deterministic
    # highest-rate list so a product flow is never counted twice across channels.
    order = ["list1", "list2", "list3", "list4a"]
    membership["dominant_tariff_list"] = np.select(
        [membership[name].eq(1) for name in order], order, default="none"
    )
    return lines, membership, pd.DataFrame(audit_rows)


def inclusive_days(start: date, end: date) -> int:
    return (end - start).days + 1


def rate_for_period(year: int, starts: list[tuple[date, float]]) -> float:
    """Annual-average additional duty from a dated sequence of rate changes."""

    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    segments = sorted(starts, key=lambda item: item[0])
    total = 0.0
    for index, (start, rate) in enumerate(segments):
        end = (
            date.fromordinal(segments[index + 1][0].toordinal() - 1)
            if index + 1 < len(segments)
            else year_end
        )
        if end < year_start or start > year_end:
            continue
        clipped_start = max(start, year_start)
        clipped_end = min(end, year_end)
        total += inclusive_days(clipped_start, clipped_end) * rate
    days = 366 if date(year, 12, 31).timetuple().tm_yday == 366 else 365
    return total / days


def annual_tariff_rates(years: Iterable[int]) -> pd.DataFrame:
    lists = {
        "list1": [(date(2018, 7, 6), 0.25)],
        "list2": [(date(2018, 8, 23), 0.25)],
        "list3": [(date(2018, 9, 24), 0.10), (date(2019, 5, 10), 0.25)],
        "list4a": [(date(2019, 9, 1), 0.15), (date(2020, 2, 14), 0.075)],
    }
    rows = []
    for year in sorted(set(int(value) for value in years)):
        row: dict[str, Any] = {"year": year}
        for list_name, starts in lists.items():
            row[f"{list_name}_annual_additional_duty"] = rate_for_period(year, starts)
        rows.append(row)
    return pd.DataFrame(rows)


def historical_income_groups(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze income status using the 2015-2017 modal World Bank analytical group."""

    raw = pd.read_excel(workbook, sheet_name="Country Analytical History", header=None)
    calendar_row = raw.iloc[5]
    columns: dict[int, int] = {}
    for year in (2015, 2016, 2017):
        matches = [idx for idx, value in calendar_row.items() if str(value).strip() == str(year)]
        if not matches:
            raise ValueError(f"Could not find calendar year {year} in World Bank history.")
        columns[year] = matches[0]
    labels = {"L": ("LIC", "Low income"), "LM": ("LMC", "Lower-middle income"), "UM": ("UMC", "Upper-middle income"), "H": ("HIC", "High income")}
    rows: list[dict[str, Any]] = []
    for _, source in raw.iloc[6:].iterrows():
        iso3 = str(source.iloc[0]).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", iso3):
            continue
        values = [str(source.iloc[columns[year]]).strip() for year in (2015, 2016, 2017)]
        valid = [value for value in values if value in labels]
        if not valid:
            continue
        counts = Counter(valid)
        max_count = max(counts.values())
        candidates = {value for value, count in counts.items() if count == max_count}
        # A rare tie is broken with the latest pre-shock GNI classification.
        selected = values[-1] if values[-1] in candidates else sorted(candidates)[0]
        group_id, group_name = labels[selected]
        rows.append(
            {
                "country_iso3_code": iso3,
                "wb_income_id": group_id,
                "wb_income_name": group_name,
                "income_group_2015": labels.get(values[0], (np.nan, np.nan))[0],
                "income_group_2016": labels.get(values[1], (np.nan, np.nan))[0],
                "income_group_2017": labels.get(values[2], (np.nan, np.nan))[0],
                "historical_income_group_rule": "modal_2015_2017_tie_break_latest_2017",
                "historical_income_group_tie": int(len(candidates) > 1),
            }
        )
    result = pd.DataFrame(rows)
    audit = pd.DataFrame(
        [
            {"metric": "historical_income_countries", "value": int(result["country_iso3_code"].nunique())},
            {"metric": "income_group_ties", "value": int(result["historical_income_group_tie"].sum())},
            {"metric": "source_workbook_sha256", "value": file_hash(workbook)},
            {"metric": "source_workbook", "value": workbook.name},
        ]
    )
    return result, audit


def read_baci_country_codes(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("country_codes_V202601.csv") as stream:
            mapping = pd.read_csv(stream, dtype={"country_code": str, "country_iso3": str})
    return dict(zip(mapping["country_code"].astype(str), mapping["country_iso3"].astype(str)))


def build_tariff_weighted_exposure(
    zip_path: Path,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct fixed HS6 product-overlap channels without extracting BACI."""

    code_to_iso3 = read_baci_country_codes(zip_path)
    usa_code = next(code for code, iso3 in code_to_iso3.items() if iso3 == "USA")
    china_code = next(code for code, iso3 in code_to_iso3.items() if iso3 == "CHN")
    dominant = dict(zip(membership["hs6"], membership["dominant_tariff_list"]))

    us_total: defaultdict[str, float] = defaultdict(float)
    china_total: defaultdict[str, float] = defaultdict(float)
    us_by_list: defaultdict[tuple[str, str], float] = defaultdict(float)
    us_competition_by_list: defaultdict[tuple[str, str], float] = defaultdict(float)
    china_by_list: defaultdict[tuple[str, str], float] = defaultdict(float)
    us_import_total_by_hs6: defaultdict[str, float] = defaultdict(float)
    china_us_export_by_hs6: defaultdict[str, float] = defaultdict(float)
    us_country_product: defaultdict[tuple[str, str], float] = defaultdict(float)
    rows_scanned = 0
    rows_to_us = 0
    rows_from_china = 0

    with zipfile.ZipFile(zip_path) as archive:
        for year in range(BASELINE_START, BASELINE_END + 1):
            member = f"BACI_HS12_Y{year}_V202601.csv"
            with archive.open(member) as binary:
                header = binary.readline().decode("utf-8").strip().split(",")
                positions = {name: index for index, name in enumerate(header)}
                for raw_line in binary:
                    parts = raw_line.decode("utf-8").rstrip("\n").split(",")
                    if len(parts) != len(header):
                        continue
                    rows_scanned += 1
                    exporter = parts[positions["i"]]
                    importer = parts[positions["j"]]
                    if exporter != china_code and importer != usa_code:
                        continue
                    try:
                        value = float(parts[positions["v"]])
                    except ValueError:
                        continue
                    if value <= 0:
                        continue
                    hs6 = parts[positions["k"]].zfill(6)
                    tariff_list = dominant.get(hs6, "none")

                    if importer == usa_code:
                        iso3 = code_to_iso3.get(exporter)
                        if iso3:
                            us_total[iso3] += value
                            rows_to_us += 1
                            if tariff_list != "none":
                                us_by_list[(iso3, tariff_list)] += value
                                us_country_product[(iso3, hs6)] += value
                                us_import_total_by_hs6[hs6] += value
                                if exporter == china_code:
                                    china_us_export_by_hs6[hs6] += value

                    # This is a separate supplier-network profile. It is not a
                    # direct US tariff liability for the importing third country.
                    if exporter == china_code and importer != usa_code:
                        iso3 = code_to_iso3.get(importer)
                        if iso3:
                            china_total[iso3] += value
                            rows_from_china += 1
                            if tariff_list != "none":
                                china_by_list[(iso3, tariff_list)] += value

    # A country has more diversion opportunity when its affected US export
    # products were also supplied substantially by China before the tariff.
    for (iso3, hs6), value in us_country_product.items():
        tariff_list = dominant[hs6]
        denominator = us_import_total_by_hs6[hs6]
        china_share = china_us_export_by_hs6[hs6] / denominator if denominator > 0 else 0.0
        us_competition_by_list[(iso3, tariff_list)] += value * china_share

    countries = sorted(set(us_total) | set(china_total))
    rows: list[dict[str, Any]] = []
    for iso3 in countries:
        row: dict[str, Any] = {
            "country_iso3_code": iso3,
            "us_export_value_pre_baci_thousand_usd": us_total.get(iso3, 0.0),
            "china_import_value_pre_baci_thousand_usd": china_total.get(iso3, 0.0),
        }
        for list_name in LIST_SPECS:
            row[f"us_export_share_{list_name}_pre"] = (
                us_by_list[(iso3, list_name)] / us_total[iso3] if us_total[iso3] else np.nan
            )
            row[f"us_export_competition_share_{list_name}_pre"] = (
                us_competition_by_list[(iso3, list_name)] / us_total[iso3]
                if us_total[iso3]
                else np.nan
            )
            row[f"china_import_share_{list_name}_pre"] = (
                china_by_list[(iso3, list_name)] / china_total[iso3]
                if china_total[iso3]
                else np.nan
            )
        row["us_tariffed_export_share_pre"] = sum(
            row[f"us_export_share_{list_name}_pre"]
            for list_name in LIST_SPECS
            if pd.notna(row[f"us_export_share_{list_name}_pre"])
        )
        row["us_china_competition_tariffed_share_pre"] = sum(
            row[f"us_export_competition_share_{list_name}_pre"]
            for list_name in LIST_SPECS
            if pd.notna(row[f"us_export_competition_share_{list_name}_pre"])
        )
        row["china_tariffed_import_share_pre"] = sum(
            row[f"china_import_share_{list_name}_pre"]
            for list_name in LIST_SPECS
            if pd.notna(row[f"china_import_share_{list_name}_pre"])
        )
        rows.append(row)

    market_total = float(sum(us_import_total_by_hs6.values()))
    china_market_total = float(sum(china_us_export_by_hs6.values()))
    audit = pd.DataFrame(
        [
            {"metric": "baci_rows_scanned_2015_2017", "value": rows_scanned},
            {"metric": "baci_rows_exporter_to_us", "value": rows_to_us},
            {"metric": "baci_rows_china_to_third_country", "value": rows_from_china},
            {"metric": "countries_with_baci_tariff_profile", "value": len(countries)},
            {"metric": "covered_hs6_with_us_imports", "value": len(us_import_total_by_hs6)},
            {"metric": "china_share_of_covered_us_imports_pre", "value": china_market_total / market_total if market_total > 0 else np.nan},
            {"metric": "baci_archive_sha256", "value": file_hash(zip_path)},
        ]
    )
    return pd.DataFrame(rows), audit


def attach_tariff_exposure(
    panel: pd.DataFrame,
    frozen: pd.DataFrame,
    tariff_profiles: pd.DataFrame,
    rates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = frozen.merge(tariff_profiles, on="country_iso3_code", how="left")
    for source_prefix, target_prefix in [
        ("us_export_competition_share", "tariff_weighted_us_diversion"),
        ("us_export_share", "tariff_weighted_us_covered_basket"),
        ("china_import_share", "tariff_weighted_china_network"),
    ]:
        for _, rate_row in rates.iterrows():
            year = int(rate_row["year"])
            value = pd.Series(0.0, index=frozen.index)
            for list_name in LIST_SPECS:
                share = pd.to_numeric(
                    frozen.get(f"{source_prefix}_{list_name}_pre"), errors="coerce"
                ).fillna(0.0)
                value += share * float(rate_row[f"{list_name}_annual_additional_duty"])
            frozen[f"{target_prefix}_{year}"] = value

    new_columns = ["country_iso3_code"] + [
        column for column in frozen.columns if column != "country_iso3_code" and column not in panel.columns
    ]
    out = panel.merge(frozen[new_columns], on="country_iso3_code", how="left")
    out["post_tariff"] = (out["year"] >= POLICY_START).astype(int)
    for target_prefix in [
        "tariff_weighted_us_diversion",
        "tariff_weighted_us_covered_basket",
        "tariff_weighted_china_network",
    ]:
        out[target_prefix] = [
            row.get(f"{target_prefix}_{int(row.year)}", np.nan)
            for _, row in out.iterrows()
        ]
        post_values = out.loc[out["post_tariff"].eq(1), target_prefix]
        scale = float(post_values.std(ddof=0, skipna=True))
        out[f"s_{target_prefix}"] = out[target_prefix] / scale if scale > 0 else np.nan
    return out, frozen


def add_frozen_regimes(frozen: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = frozen.copy()
    thresholds: list[dict[str, Any]] = []
    for raw_col, name in [("eci_pre_raw", "eci_pre_regime"), ("coi_pre_raw", "coi_pre_regime")]:
        values = pd.to_numeric(frozen[raw_col], errors="coerce")
        low = float(values.quantile(1.0 / 3.0))
        high = float(values.quantile(2.0 / 3.0))
        frozen[name] = pd.cut(
            values,
            bins=[-np.inf, low, high, np.inf],
            labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype("string")
        thresholds.extend(
            [
                {"construct": raw_col, "quantile": "q33", "value": low},
                {"construct": raw_col, "quantile": "q67", "value": high},
            ]
        )
    keys = ["country_iso3_code", "eci_pre_regime", "coi_pre_regime"]
    panel = panel.merge(frozen[keys], on="country_iso3_code", how="left")
    return panel, frozen, pd.DataFrame(thresholds)


def add_corrected_gvc_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    baseline = (
        out[out["year"].between(BASELINE_START, BASELINE_END)]
        .groupby("country_iso3_code", as_index=False)["tiva_fexgr_dva_share"]
        .mean()
        .rename(columns={"tiva_fexgr_dva_share": "fwd_linkage_pre"})
    )
    out = out.merge(baseline, on="country_iso3_code", how="left")
    out["gvc_signed_change"] = out["delta_tiva_fexgr_dva_share"]
    out["gvc_deviation_from_pre"] = out["tiva_fexgr_dva_share"] - out["fwd_linkage_pre"]
    out["gvc_adverse_deviation_stability"] = np.minimum(out["gvc_deviation_from_pre"], 0.0)
    out["gvc_absolute_deviation_stability"] = -np.abs(out["gvc_deviation_from_pre"])
    return out


def add_tariff_linear_terms(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    out = df.copy()
    out["tariff_treatment"] = out[treatment_col]
    out["post_x_eci_pre"] = out["post_tariff"] * out["z_eci_pre_raw"]
    out["eci_x_tariff_treatment"] = out["z_eci_pre_raw"] * out["tariff_treatment"]
    return out


def contrast_row(
    fit: Any,
    contrast: np.ndarray,
    *,
    test_id: str,
    hypothesis: str,
    outcome: str,
    expected_direction: str,
    alternative: str,
    bootstrap_reps: int,
    seed: int,
    family: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "test_id": test_id,
        "hypothesis": hypothesis,
        "outcome": outcome,
        "expected_direction": expected_direction,
        "alternative": alternative,
        "multiplicity_family": family,
        **contrast_statistics(fit, contrast, alternative=alternative),
        **wild_cluster_bootstrap_contrast(
            fit, contrast, reps=bootstrap_reps, seed=seed, alternative=alternative
        ),
        "n_obs": fit.n_obs,
        "n_countries": fit.n_countries,
        "n_years": fit.n_years,
    }
    if extra:
        row.update(extra)
    row["pvalue_for_fdr"] = row["p_wild_bootstrap"]
    return row


def target_contrast(fit: Any, term: str) -> np.ndarray:
    result = np.zeros(len(fit.term_names), dtype=float)
    result[fit.term_index(term)] = 1.0
    return result


def wild_cluster_bootstrap_contrast(
    fit: Any,
    contrast: np.ndarray,
    reps: int,
    seed: int,
    alternative: str,
) -> dict[str, float | int]:
    """Vectorized Rademacher wild-cluster bootstrap-t with a restricted null."""

    observed = contrast_statistics(fit, contrast, alternative=alternative)
    if not np.isfinite(observed["tvalue"]):
        return {
            "p_wild_bootstrap": np.nan,
            "bootstrap_reps_requested": int(reps),
            "bootstrap_reps_success": 0,
        }
    restriction_variance = float(contrast @ fit.inv_xx @ contrast)
    if restriction_variance <= 0:
        return {
            "p_wild_bootstrap": np.nan,
            "bootstrap_reps_requested": int(reps),
            "bootstrap_reps_success": 0,
        }

    restricted_beta = fit.beta - (
        (fit.inv_xx @ contrast) * (float(contrast @ fit.beta) / restriction_variance)
    )
    yhat_null = fit.x_resid @ restricted_beta
    residual_null = fit.y_resid - yhat_null
    rng = np.random.default_rng(seed)
    cluster_indices = [np.flatnonzero(fit.cluster_codes == cluster) for cluster in range(fit.n_clusters)]
    correction = (fit.n_clusters / max(fit.n_clusters - 1, 1)) * (
        (fit.n_obs - 1) / max(fit.n_obs - fit.k_full, 1)
    )
    tstars: list[float] = []
    batch_size = min(128, max(int(reps), 1))
    for start in range(0, int(reps), batch_size):
        width = min(batch_size, int(reps) - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(width, fit.n_clusters)).T
        ystar = yhat_null[:, None] + residual_null[:, None] * signs[fit.cluster_codes]
        beta_star = fit.inv_xx @ (fit.x_resid.T @ ystar)
        residual_star = ystar - fit.x_resid @ beta_star
        scores = np.zeros((fit.n_clusters, len(fit.term_names), width), dtype=float)
        for cluster, indices in enumerate(cluster_indices):
            scores[cluster] = fit.x_resid[indices].T @ residual_star[indices]
        meat = np.einsum("gkr,glr->klr", scores, scores, optimize=True)
        covariance = correction * np.einsum(
            "ab,bcr,cd->adr", fit.inv_xx, meat, fit.inv_xx, optimize=True
        )
        variances = np.einsum("a,abr,b->r", contrast, covariance, contrast, optimize=True)
        standard_errors = np.sqrt(np.maximum(variances, 0.0))
        tvalues = (contrast @ beta_star) / standard_errors
        tstars.extend(tvalues[np.isfinite(tvalues) & (standard_errors > 0)].tolist())

    tarr = np.asarray(tstars, dtype=float)
    if len(tarr) == 0:
        p_boot = np.nan
    elif alternative == "less":
        p_boot = float((1 + np.sum(tarr <= observed["tvalue"])) / (len(tarr) + 1))
    elif alternative == "greater":
        p_boot = float((1 + np.sum(tarr >= observed["tvalue"])) / (len(tarr) + 1))
    else:
        p_boot = float((1 + np.sum(np.abs(tarr) >= abs(observed["tvalue"]))) / (len(tarr) + 1))
    return {
        "p_wild_bootstrap": p_boot,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_success": int(len(tarr)),
    }

def run_confirmatory_models(
    panel: pd.DataFrame, treatment_col: str, bootstrap_reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_outcomes = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    gvc_sensitivity = [
        ("H1_sensitivity", "Signed forward-GVC linkage change", "gvc_signed_change"),
        ("H1_sensitivity", "Absolute forward-GVC deviation stability", "gvc_absolute_deviation_stability"),
    ]
    h3_sensitivity = [
        ("H3_sensitivity", "Destination entropy excluding US and China", "destination_entropy_excl_us_china"),
        ("H3_sensitivity", "Effective destinations excluding US and China", "effective_destinations_excl_us_china"),
    ]
    rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    terms = ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]
    prepared = add_tariff_linear_terms(panel, treatment_col)
    for offset, (hypothesis, label, outcome) in enumerate(primary_outcomes + gvc_sensitivity + h3_sensitivity):
        fit = fit_twfe(prepared, outcome, terms)
        target = target_contrast(fit, "eci_x_tariff_treatment")
        row = contrast_row(
            fit,
            target,
            test_id=f"{hypothesis}_{outcome}",
            hypothesis=hypothesis,
            outcome=label,
            expected_direction="positive",
            alternative="greater",
            bootstrap_reps=bootstrap_reps,
            seed=seed + offset * 101,
            family="full_reported_family",
            extra={"treatment": treatment_col},
        )
        rows.append(row)
        if hypothesis in {"H1", "H2", "H3"}:
            sample = prepared.dropna(subset=[outcome] + terms).copy()
            sample["standardized_outcome"] = zscore(sample[outcome])
            fit_std = fit_twfe(sample, "standardized_outcome", terms)
            for bound in (0.10, 0.20):
                equivalence_result = equivalence_and_mde(
                    fit_std, "eci_x_tariff_treatment", bound=bound
                )
                equivalence_rows.append(
                    {
                        "test_id": f"{hypothesis}_{outcome}_equivalence_bound_{bound:.2f}",
                        "hypothesis": f"{hypothesis} practical-equivalence diagnostic",
                        "outcome": label,
                        "test_type": "cluster_tost_equivalence",
                        "treatment": treatment_col,
                        "multiplicity_family": "full_reported_family",
                        **equivalence_result,
                        "pvalue_for_fdr": equivalence_result["tost_pvalue"],
                        "n_obs": fit_std.n_obs,
                        "n_countries": fit_std.n_countries,
                    }
                )

    h4_rows: list[dict[str, Any]] = []
    h4_terms = [
        "tariff_treatment",
        "post_x_eci_pre",
        "post_x_coi_pre",
        "eci_x_tariff_treatment",
        "coi_x_tariff_treatment",
        "post_x_eci_x_coi",
        "h4_eci_x_coi_x_tariff",
    ]
    h4 = prepared.copy()
    h4["post_x_coi_pre"] = h4["post_tariff"] * h4["z_coi_pre_raw"]
    h4["coi_x_tariff_treatment"] = h4["z_coi_pre_raw"] * h4["tariff_treatment"]
    h4["post_x_eci_x_coi"] = h4["post_tariff"] * h4["z_eci_pre_raw"] * h4["z_coi_pre_raw"]
    h4["h4_eci_x_coi_x_tariff"] = h4["z_eci_pre_raw"] * h4["z_coi_pre_raw"] * h4["tariff_treatment"]

    for offset, (_, label, outcome) in enumerate(primary_outcomes):
        fit = fit_twfe(h4, outcome, h4_terms)

        h4_rows.append(
            contrast_row(
                fit,
                target_contrast(fit, "h4_eci_x_coi_x_tariff"),
                test_id=f"H4_{outcome}",
                hypothesis="H4",
                outcome=label,
                expected_direction="positive",
                alternative="greater",
                bootstrap_reps=bootstrap_reps,
                seed=seed + 1000 + offset * 101,
                family="full_reported_family",
                extra={"treatment": treatment_col},
            )
        )
    h4_wald = h4_stacked_omnibus(h4, primary_outcomes, h4_terms)

    h5_rows: list[dict[str, Any]] = []
    observed = prepared[prepared["gpr_country_annual"].notna()].copy()
    observed["z_gpr_observed"] = zscore(observed["gpr_country_annual"])
    h5_terms = [
        "tariff_treatment",
        "post_x_eci_pre",
        "z_gpr_observed",
        "eci_x_gpr",
        "eci_x_tariff_treatment",
        "gpr_x_tariff_treatment",
        "h5_eci_x_gpr_x_tariff",
    ]
    observed["eci_x_gpr"] = observed["z_eci_pre_raw"] * observed["z_gpr_observed"]
    observed["gpr_x_tariff_treatment"] = observed["z_gpr_observed"] * observed["tariff_treatment"]
    observed["h5_eci_x_gpr_x_tariff"] = observed["z_eci_pre_raw"] * observed["z_gpr_observed"] * observed["tariff_treatment"]
    for offset, (_, label, outcome) in enumerate(primary_outcomes):
        try:
            fit = fit_twfe(observed, outcome, h5_terms)
            h5_rows.append(
                contrast_row(
                    fit,
                    target_contrast(fit, "h5_eci_x_gpr_x_tariff"),
                    test_id=f"H5_{outcome}",
                    hypothesis="H5 exploratory observed-GPR-only",
                    outcome=label,
                    expected_direction="negative",
                    alternative="less",
                    bootstrap_reps=bootstrap_reps,
                    seed=seed + 2000 + offset * 101,
                    family="full_reported_family",
                    extra={"treatment": treatment_col, "gpr_sample": "directly_observed_only"},
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            h5_rows.append({"test_id": f"H5_{outcome}", "hypothesis": "H5", "outcome": label, "error": str(exc)})
    return (
        pd.DataFrame(rows),
        pd.DataFrame(equivalence_rows),
        pd.DataFrame(h4_rows),
        pd.concat([pd.DataFrame(h5_rows), h4_wald], ignore_index=True, sort=False),
    )


def h4_stacked_omnibus(
    prepared: pd.DataFrame,
    outcomes: list[tuple[str, str, str]],
    terms: list[str],
) -> pd.DataFrame:
    """Test H4 jointly while retaining cross-outcome cluster covariance."""

    long_parts: list[pd.DataFrame] = []
    for _, label, outcome in outcomes:
        part = prepared[["country_iso3_code", "year", outcome] + terms].dropna().copy()
        part["outcome_name"] = label
        part["stacked_outcome"] = zscore(part[outcome])
        long_parts.append(part)
    long_df = pd.concat(long_parts, ignore_index=True)
    stacked_terms: list[str] = []
    label_keys: list[str] = []
    for _, label, _ in outcomes:
        label_key = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        label_keys.append(label_key)
        for term in terms:
            name = f"{label_key}__{term}"
            long_df[name] = np.where(long_df["outcome_name"].eq(label), long_df[term], 0.0)
            stacked_terms.append(name)
    long_df["entity_outcome"] = (
        long_df["country_iso3_code"].astype(str) + "__" + long_df["outcome_name"].astype(str)
    )
    long_df["year_outcome"] = (
        long_df["year"].astype(str) + "__" + long_df["outcome_name"].astype(str)
    )
    fit = fit_twfe(
        long_df,
        "stacked_outcome",
        stacked_terms,
        entity_col="entity_outcome",
        time_col="year_outcome",
        cluster_col="country_iso3_code",
    )
    idx = [fit.term_index(f"{key}__h4_eci_x_coi_x_tariff") for key in label_keys]
    beta = fit.beta[idx]
    covariance = fit.covariance[np.ix_(idx, idx)]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    pvalue = float(stats.chi2.sf(statistic, df=len(idx)))
    return pd.DataFrame(
        [{
            "test_id": "H4_omnibus",
            "hypothesis": "H4",
            "outcome": "H4 outcome-specific interactions jointly equal zero",
            "test_type": "stacked_country_clustered_wald",
            "wald_chi2": statistic,
            "df": len(idx),
            "p_cluster": pvalue,
            "pvalue_for_fdr": pvalue,
            "multiplicity_family": "full_reported_family",
            "n_obs": fit.n_obs,
            "n_country_clusters": fit.n_clusters,
            "n_entity_outcome_panels": fit.n_countries,
            "n_year_outcome_fixed_effects": fit.n_years,
        }]
    )


def run_grouped_eci_tests(
    panel: pd.DataFrame,
    *,
    group_col: str,
    group_label: str,
    treatment_col: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    prepared = add_tariff_linear_terms(panel, treatment_col)
    all_rows: list[dict[str, Any]] = []
    omnibus_rows: list[dict[str, Any]] = []
    for outcome_index, (hypothesis, label, outcome) in enumerate(outcomes):
        d = prepared.dropna(subset=[outcome, group_col, "tariff_treatment", "z_eci_pre_raw"]).copy()
        groups = [
            str(group)
            for group, count in d[["country_iso3_code", group_col]].drop_duplicates()[group_col].astype(str).value_counts().items()
            if count >= 4
        ]
        groups = sorted(groups)
        if len(groups) < 2:
            continue
        reference = "HIC" if group_col == "wb_income_id" and "HIC" in groups else ("middle" if "middle" in groups else groups[0])
        terms = ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]
        modifiers: list[str] = []
        for group in groups:
            if group == reference:
                continue
            flag = d[group_col].astype(str).eq(group).astype(int)
            for base in ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]:
                name = f"{group}_{base}_modifier"
                d[name] = flag * d[base]
                terms.append(name)
                if base == "eci_x_tariff_treatment":
                    modifiers.append(name)
        try:
            fit = fit_twfe(d, outcome, terms)
        except (ValueError, np.linalg.LinAlgError) as exc:
            all_rows.append({"test_id": f"{group_label}_{outcome}_model", "error": str(exc)})
            continue
        group_contrasts: dict[str, np.ndarray] = {}
        for group in groups:
            contrast = target_contrast(fit, "eci_x_tariff_treatment")
            if group != reference:
                contrast[fit.term_index(f"{group}_eci_x_tariff_treatment_modifier")] = 1.0
            group_contrasts[group] = contrast
            all_rows.append(
                contrast_row(
                    fit,
                    contrast,
                    test_id=f"{group_label}_{outcome}_{group}_slope",
                    hypothesis=f"{group_label} candidate slope",
                    outcome=label,
                    expected_direction="two-sided exploratory",
                    alternative="two-sided",
                    bootstrap_reps=bootstrap_reps,
                    seed=seed + outcome_index * 10000 + len(all_rows),
                    family="full_reported_family",
                    extra={"group_dimension": group_label, "group": group, "reference_group": reference},
                )
            )
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                all_rows.append(
                    contrast_row(
                        fit,
                        group_contrasts[left] - group_contrasts[right],
                        test_id=f"{group_label}_{outcome}_{left}_minus_{right}",
                        hypothesis=f"{group_label} formal slope difference",
                        outcome=label,
                        expected_direction="two-sided exploratory",
                        alternative="two-sided",
                        bootstrap_reps=bootstrap_reps,
                        seed=seed + outcome_index * 10000 + len(all_rows),
                        family="full_reported_family",
                        extra={"group_dimension": group_label, "contrast": f"{left} - {right}"},
                    )
                )
        if modifiers:
            idx = [fit.term_index(name) for name in modifiers]
            beta = fit.beta[idx]
            covariance = fit.covariance[np.ix_(idx, idx)]
            stat = float(beta.T @ np.linalg.pinv(covariance) @ beta)
            omnibus_rows.append(
                {
                    "test_id": f"{group_label}_{outcome}_omnibus",
                    "hypothesis": f"{group_label} formal slope equality",
                    "outcome": label,
                    "test_type": "cluster_wald_omnibus",
                    "wald_chi2": stat,
                    "df": len(idx),
                    "p_cluster": float(stats.chi2.sf(stat, df=len(idx))),
                    "pvalue_for_fdr": float(stats.chi2.sf(stat, df=len(idx))),
                    "multiplicity_family": "full_reported_family",
                }
            )
    return pd.DataFrame(all_rows), pd.DataFrame(omnibus_rows)


def run_gpr_income_tests(
    panel: pd.DataFrame, treatment_col: str, bootstrap_reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    prepared = add_tariff_linear_terms(panel, treatment_col)
    prepared = prepared[prepared["gpr_country_annual"].notna()].copy()
    prepared["z_gpr_observed"] = zscore(prepared["gpr_country_annual"])
    base_terms = [
        "tariff_treatment",
        "post_x_eci_pre",
        "z_gpr_observed",
        "eci_x_gpr",
        "eci_x_tariff_treatment",
        "gpr_x_tariff_treatment",
        "h5_eci_x_gpr_x_tariff",
    ]
    prepared["eci_x_gpr"] = prepared["z_eci_pre_raw"] * prepared["z_gpr_observed"]
    prepared["gpr_x_tariff_treatment"] = prepared["z_gpr_observed"] * prepared["tariff_treatment"]
    prepared["h5_eci_x_gpr_x_tariff"] = prepared["z_eci_pre_raw"] * prepared["z_gpr_observed"] * prepared["tariff_treatment"]
    rows: list[dict[str, Any]] = []
    omnibus: list[dict[str, Any]] = []
    for outcome_index, (_, label, outcome) in enumerate(outcomes):
        d = prepared.dropna(subset=[outcome, "wb_income_id"] + base_terms).copy()
        groups = [
            str(group)
            for group, count in d[["country_iso3_code", "wb_income_id"]].drop_duplicates()["wb_income_id"].astype(str).value_counts().items()
            if count >= 4
        ]
        groups = sorted(groups)
        if len(groups) < 2:
            continue
        reference = "HIC" if "HIC" in groups else groups[0]
        terms = list(base_terms)
        modifier_terms: list[str] = []
        for group in groups:
            if group == reference:
                continue
            flag = d["wb_income_id"].astype(str).eq(group).astype(int)
            for base in base_terms:
                name = f"{group}_{base}_modifier"
                d[name] = flag * d[base]
                terms.append(name)
                if base == "h5_eci_x_gpr_x_tariff":
                    modifier_terms.append(name)
        try:
            fit = fit_twfe(d, outcome, terms)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"test_id": f"gpr_income_{outcome}_model", "error": str(exc)})
            continue
        contrasts: dict[str, np.ndarray] = {}
        for group in groups:
            contrast = target_contrast(fit, "h5_eci_x_gpr_x_tariff")
            if group != reference:
                contrast[fit.term_index(f"{group}_h5_eci_x_gpr_x_tariff_modifier")] = 1.0
            contrasts[group] = contrast
            rows.append(
                contrast_row(
                    fit,
                    contrast,
                    test_id=f"gpr_income_{outcome}_{group}_slope",
                    hypothesis="GPR-by-income candidate moderation",
                    outcome=label,
                    expected_direction="two-sided exploratory",
                    alternative="two-sided",
                    bootstrap_reps=bootstrap_reps,
                    seed=seed + outcome_index * 10000 + len(rows),
                    family="full_reported_family",
                    extra={"group_dimension": "historical_income", "group": group, "reference_group": reference, "gpr_sample": "directly_observed_only"},
                )
            )
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                rows.append(
                    contrast_row(
                        fit,
                        contrasts[left] - contrasts[right],
                        test_id=f"gpr_income_{outcome}_{left}_minus_{right}",
                        hypothesis="GPR-by-income formal moderation difference",
                        outcome=label,
                        expected_direction="two-sided exploratory",
                        alternative="two-sided",
                        bootstrap_reps=bootstrap_reps,
                        seed=seed + outcome_index * 10000 + len(rows),
                        family="full_reported_family",
                        extra={"group_dimension": "historical_income", "contrast": f"{left} - {right}", "gpr_sample": "directly_observed_only"},
                    )
                )
        if modifier_terms:
            idx = [fit.term_index(name) for name in modifier_terms]
            beta = fit.beta[idx]
            covariance = fit.covariance[np.ix_(idx, idx)]
            stat = float(beta.T @ np.linalg.pinv(covariance) @ beta)
            omnibus.append(
                {
                    "test_id": f"gpr_income_{outcome}_omnibus",
                    "hypothesis": "GPR-by-income formal moderation equality",
                    "outcome": label,
                    "test_type": "cluster_wald_omnibus",
                    "wald_chi2": stat,
                    "df": len(idx),
                    "p_cluster": float(stats.chi2.sf(stat, df=len(idx))),
                    "pvalue_for_fdr": float(stats.chi2.sf(stat, df=len(idx))),
                    "multiplicity_family": "full_reported_family",
                    "gpr_sample": "directly_observed_only",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(omnibus)


PROFILE_GPR_FEATURES = [
    "eci_pre_raw",
    "coi_pre_raw",
    "z_nexus_exposure_pre",
    "baseline_log_gdp_pc",
    "baseline_trade_open",
    "baseline_wgi",
    "baseline_rents",
]


def profile_knn_prediction(
    recipient: pd.Series,
    donors: pd.DataFrame,
    features: list[str],
    k: int,
) -> tuple[float | None, str, int]:
    """Predict GPR from frozen country profiles, prioritising same-income donors."""

    available = donors.dropna(subset=["gpr_country_annual"]).copy()
    if available.empty:
        return None, "no_donors", 0
    pool = available
    pool_type = "all_income_groups"
    recipient_income = recipient.get("wb_income_id", np.nan)
    if pd.notna(recipient_income) and "wb_income_id" in available:
        same_income = available[available["wb_income_id"].astype(str).eq(str(recipient_income))]
        if len(same_income) >= k:
            pool = same_income
            pool_type = "same_historical_income_group"
    scales = pool[features].std(skipna=True, ddof=0).replace(0, np.nan)
    distances: list[tuple[float, float]] = []
    for _, donor in pool.iterrows():
        components: list[float] = []
        for feature in features:
            recipient_value = recipient.get(feature, np.nan)
            donor_value = donor.get(feature, np.nan)
            scale = scales.get(feature, np.nan)
            if pd.notna(recipient_value) and pd.notna(donor_value) and pd.notna(scale) and scale > 0:
                components.append(((float(recipient_value) - float(donor_value)) / float(scale)) ** 2)
        if len(components) >= 3:
            distances.append((float(np.sqrt(np.mean(components))), float(donor["gpr_country_annual"])))
    if not distances:
        return None, "insufficient_profile_overlap", 0
    nearest = sorted(distances, key=lambda item: item[0])[: min(k, len(distances))]
    d = np.asarray([item[0] for item in nearest], dtype=float)
    values = np.asarray([item[1] for item in nearest], dtype=float)
    weights = 1.0 / (d + 1e-6)
    return float(np.sum(weights * values) / np.sum(weights)), pool_type, len(nearest)


def validate_profile_gpr(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-country-out validation uses no outcomes and only frozen profiles."""

    rows: list[dict[str, Any]] = []
    for k in (3, 5, 10):
        for year, group in panel[panel["gpr_country_annual"].notna()].groupby("year"):
            for index, recipient in group.iterrows():
                donors = group.drop(index=index)
                prediction, pool_type, n_donors = profile_knn_prediction(
                    recipient, donors, PROFILE_GPR_FEATURES, k
                )
                if prediction is None:
                    continue
                observed = float(recipient["gpr_country_annual"])
                rows.append(
                    {
                        "k": k,
                        "year": int(year),
                        "country_iso3_code": recipient["country_iso3_code"],
                        "observed_gpr": observed,
                        "predicted_gpr": prediction,
                        "absolute_error": abs(prediction - observed),
                        "squared_error": (prediction - observed) ** 2,
                        "donor_pool": pool_type,
                        "n_nearest_donors": n_donors,
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(columns=["k", "n_masked", "rmse", "mae", "correlation"])
    summary = (
        detail.groupby("k", as_index=False)
        .agg(
            n_masked=("country_iso3_code", "size"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            mae=("absolute_error", "mean"),
            correlation=("observed_gpr", lambda values: np.nan),
        )
    )
    correlations = detail.groupby("k").apply(
        lambda group: group["observed_gpr"].corr(group["predicted_gpr"]), include_groups=False
    )
    summary["correlation"] = summary["k"].map(correlations)
    return detail, summary


def add_profile_matched_gpr(panel: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only missing country-year GPR with matched-country estimates as a sensitivity."""

    out = panel.copy()
    out["gpr_profile_matched"] = out["gpr_country_annual"]
    out["gpr_profile_matched_imputed"] = 0
    rows: list[dict[str, Any]] = []
    for year, group in out.groupby("year"):
        donors = group[group["gpr_country_annual"].notna()].copy()
        for index, recipient in group[group["gpr_country_annual"].isna()].iterrows():
            prediction, pool_type, n_donors = profile_knn_prediction(
                recipient, donors, PROFILE_GPR_FEATURES, k
            )
            if prediction is None:
                continue
            out.at[index, "gpr_profile_matched"] = prediction
            out.at[index, "gpr_profile_matched_imputed"] = 1
            rows.append(
                {
                    "country_iso3_code": recipient["country_iso3_code"],
                    "year": int(year),
                    "gpr_profile_matched": prediction,
                    "donor_pool": pool_type,
                    "n_nearest_donors": n_donors,
                    "k": k,
                }
            )
    detail = pd.DataFrame(rows)
    return out, detail


def run_h5_profile_sensitivity(
    panel: pd.DataFrame,
    treatment_col: str,
    gpr_col: str,
    sample_label: str,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate H5 under an explicitly labelled GPR availability strategy."""

    outcomes = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    prepared = add_tariff_linear_terms(panel, treatment_col)
    prepared = prepared[prepared[gpr_col].notna()].copy()
    prepared["z_gpr_profile"] = zscore(prepared[gpr_col])
    prepared["eci_x_gpr"] = prepared["z_eci_pre_raw"] * prepared["z_gpr_profile"]
    prepared["gpr_x_tariff_treatment"] = prepared["z_gpr_profile"] * prepared["tariff_treatment"]
    prepared["h5_eci_x_gpr_x_tariff"] = (
        prepared["z_eci_pre_raw"] * prepared["z_gpr_profile"] * prepared["tariff_treatment"]
    )
    terms = [
        "tariff_treatment",
        "post_x_eci_pre",
        "z_gpr_profile",
        "eci_x_gpr",
        "eci_x_tariff_treatment",
        "gpr_x_tariff_treatment",
        "h5_eci_x_gpr_x_tariff",
    ]
    rows: list[dict[str, Any]] = []
    for offset, (_, label, outcome) in enumerate(outcomes):
        fit = fit_twfe(prepared, outcome, terms)
        rows.append(
            contrast_row(
                fit,
                target_contrast(fit, "h5_eci_x_gpr_x_tariff"),
                test_id=f"H5_{sample_label}_{outcome}",
                hypothesis="H5 exploratory profile-matched-GPR sensitivity",
                outcome=label,
                expected_direction="negative",
                alternative="less",
                bootstrap_reps=bootstrap_reps,
                seed=seed + offset * 101,
                family="full_reported_family",
                extra={"treatment": treatment_col, "gpr_sample": sample_label, "gpr_column": gpr_col},
            )
        )
    return pd.DataFrame(rows)


def run_targeted_eci_group_test(
    panel: pd.DataFrame,
    *,
    group_col: str,
    target_group: str,
    outcome: str,
    outcome_label: str,
    candidate_id: str,
    treatment_col: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use one pooled model for a named E1/E2 subgroup and formal differences."""

    prepared = add_tariff_linear_terms(panel, treatment_col)
    d = prepared.dropna(subset=[outcome, group_col, "tariff_treatment", "z_eci_pre_raw"]).copy()
    groups = [
        str(group)
        for group, count in d[["country_iso3_code", group_col]].drop_duplicates()[group_col].astype(str).value_counts().items()
        if count >= 4
    ]
    groups = sorted(groups)
    if target_group not in groups:
        return pd.DataFrame([{"test_id": candidate_id, "error": f"Target group {target_group} is not estimable."}]), pd.DataFrame()
    reference = "HIC" if group_col == "wb_income_id" and "HIC" in groups else ("middle" if "middle" in groups else groups[0])
    terms = ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]
    modifiers: list[str] = []
    for group in groups:
        if group == reference:
            continue
        flag = d[group_col].astype(str).eq(group).astype(int)
        for base in ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]:
            name = f"{group}_{base}_modifier"
            d[name] = flag * d[base]
            terms.append(name)
            if base == "eci_x_tariff_treatment":
                modifiers.append(name)
    fit = fit_twfe(d, outcome, terms)
    contrasts: dict[str, np.ndarray] = {}
    for group in groups:
        contrast = target_contrast(fit, "eci_x_tariff_treatment")
        if group != reference:
            contrast[fit.term_index(f"{group}_eci_x_tariff_treatment_modifier")] = 1.0
        contrasts[group] = contrast
    rows = [
        contrast_row(
            fit,
            contrasts[target_group],
            test_id=f"{candidate_id}_{target_group}_slope",
            hypothesis=f"{candidate_id} targeted subgroup slope",
            outcome=outcome_label,
            expected_direction="two-sided exploratory",
            alternative="two-sided",
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            family="full_reported_family",
            extra={"group_dimension": group_col, "target_group": target_group, "reference_group": reference},
        )
    ]
    for offset, group in enumerate(groups):
        if group == target_group:
            continue
        rows.append(
            contrast_row(
                fit,
                contrasts[target_group] - contrasts[group],
                test_id=f"{candidate_id}_{target_group}_minus_{group}",
                hypothesis=f"{candidate_id} formal subgroup difference",
                outcome=outcome_label,
                expected_direction="two-sided exploratory",
                alternative="two-sided",
                bootstrap_reps=bootstrap_reps,
                seed=seed + 100 + offset,
                family="full_reported_family",
                extra={"group_dimension": group_col, "contrast": f"{target_group} - {group}", "reference_group": reference},
            )
        )
    idx = [fit.term_index(name) for name in modifiers]
    beta = fit.beta[idx]
    covariance = fit.covariance[np.ix_(idx, idx)]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    omnibus = pd.DataFrame(
        [{
            "test_id": f"{candidate_id}_omnibus",
            "hypothesis": f"{candidate_id} formal equality across available groups",
            "outcome": outcome_label,
            "test_type": "cluster_wald_omnibus",
            "wald_chi2": statistic,
            "df": len(idx),
            "p_cluster": float(stats.chi2.sf(statistic, df=len(idx))),
            "pvalue_for_fdr": float(stats.chi2.sf(statistic, df=len(idx))),
            "multiplicity_family": "full_reported_family",
            "n_obs": fit.n_obs,
            "n_countries": fit.n_countries,
            "available_groups": ",".join(groups),
        }]
    )
    return pd.DataFrame(rows), omnibus


def run_targeted_gpr_income_test(
    panel: pd.DataFrame,
    *,
    gpr_col: str,
    sample_label: str,
    treatment_col: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test E3 in a pooled income-interaction model, not separate regressions."""

    outcome = "partner_diversification_excl_us_china"
    outcome_label = "Partner diversification excluding US and China"
    prepared = add_tariff_linear_terms(panel, treatment_col)
    prepared = prepared[prepared[gpr_col].notna()].copy()
    prepared["z_gpr_target"] = zscore(prepared[gpr_col])
    prepared["eci_x_gpr"] = prepared["z_eci_pre_raw"] * prepared["z_gpr_target"]
    prepared["gpr_x_tariff_treatment"] = prepared["z_gpr_target"] * prepared["tariff_treatment"]
    prepared["h5_eci_x_gpr_x_tariff"] = (
        prepared["z_eci_pre_raw"] * prepared["z_gpr_target"] * prepared["tariff_treatment"]
    )
    base_terms = [
        "tariff_treatment",
        "post_x_eci_pre",
        "z_gpr_target",
        "eci_x_gpr",
        "eci_x_tariff_treatment",
        "gpr_x_tariff_treatment",
        "h5_eci_x_gpr_x_tariff",
    ]
    d = prepared.dropna(subset=[outcome, "wb_income_id"] + base_terms).copy()
    groups = [
        str(group)
        for group, count in d[["country_iso3_code", "wb_income_id"]].drop_duplicates()["wb_income_id"].astype(str).value_counts().items()
        if count >= 4
    ]
    groups = sorted(groups)
    target_group = "HIC"
    if target_group not in groups:
        return pd.DataFrame([{"test_id": f"E3_{sample_label}", "error": "HIC is not estimable."}]), pd.DataFrame()
    reference = target_group
    terms = list(base_terms)
    modifiers: list[str] = []
    for group in groups:
        if group == reference:
            continue
        flag = d["wb_income_id"].astype(str).eq(group).astype(int)
        for base in base_terms:
            name = f"{group}_{base}_modifier"
            d[name] = flag * d[base]
            terms.append(name)
            if base == "h5_eci_x_gpr_x_tariff":
                modifiers.append(name)
    fit = fit_twfe(d, outcome, terms)
    contrasts: dict[str, np.ndarray] = {}
    for group in groups:
        contrast = target_contrast(fit, "h5_eci_x_gpr_x_tariff")
        if group != reference:
            contrast[fit.term_index(f"{group}_h5_eci_x_gpr_x_tariff_modifier")] = 1.0
        contrasts[group] = contrast
    rows = [
        contrast_row(
            fit,
            contrasts[target_group],
            test_id=f"E3_{sample_label}_{target_group}_slope",
            hypothesis="E3 high-income GPR moderation",
            outcome=outcome_label,
            expected_direction="negative exploratory",
            alternative="two-sided",
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            family="full_reported_family",
            extra={"gpr_sample": sample_label, "gpr_column": gpr_col, "target_group": target_group},
        )
    ]
    for offset, group in enumerate(groups):
        if group == target_group:
            continue
        rows.append(
            contrast_row(
                fit,
                contrasts[target_group] - contrasts[group],
                test_id=f"E3_{sample_label}_{target_group}_minus_{group}",
                hypothesis="E3 formal GPR moderation difference",
                outcome=outcome_label,
                expected_direction="two-sided exploratory",
                alternative="two-sided",
                bootstrap_reps=bootstrap_reps,
                seed=seed + 100 + offset,
                family="full_reported_family",
                extra={"gpr_sample": sample_label, "gpr_column": gpr_col, "contrast": f"{target_group} - {group}"},
            )
        )
    idx = [fit.term_index(name) for name in modifiers]
    beta = fit.beta[idx]
    covariance = fit.covariance[np.ix_(idx, idx)]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    omnibus = pd.DataFrame(
        [{
            "test_id": f"E3_{sample_label}_omnibus",
            "hypothesis": "E3 formal equality across available income groups",
            "outcome": outcome_label,
            "test_type": "cluster_wald_omnibus",
            "wald_chi2": statistic,
            "df": len(idx),
            "p_cluster": float(stats.chi2.sf(statistic, df=len(idx))),
            "pvalue_for_fdr": float(stats.chi2.sf(statistic, df=len(idx))),
            "multiplicity_family": "full_reported_family",
            "n_obs": fit.n_obs,
            "n_countries": fit.n_countries,
            "gpr_sample": sample_label,
            "available_groups": ",".join(groups),
        }]
    )
    return pd.DataFrame(rows), omnibus


def run_baseline_control_sensitivity(
    panel: pd.DataFrame, treatment_col: str, bootstrap_reps: int, seed: int
) -> pd.DataFrame:
    """Allow frozen country characteristics to have separate year effects."""

    outcomes = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    controls, control_terms = add_pre_control_year_terms(panel)
    prepared = add_tariff_linear_terms(controls, treatment_col)
    terms = ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"] + control_terms
    rows: list[dict[str, Any]] = []
    for offset, (hypothesis, label, outcome) in enumerate(outcomes):
        fit = fit_twfe(prepared, outcome, terms)
        rows.append(
            contrast_row(
                fit,
                target_contrast(fit, "eci_x_tariff_treatment"),
                test_id=f"{hypothesis}_frozen_controls_x_year",
                hypothesis=f"{hypothesis} frozen-control sensitivity",
                outcome=label,
                expected_direction="positive",
                alternative="greater",
                bootstrap_reps=bootstrap_reps,
                seed=seed + offset * 101,
                family="full_reported_family",
                extra={"treatment": treatment_col, "control_strategy": "frozen_pre_shock_controls_by_year"},
            )
        )
    return pd.DataFrame(rows)

def run_h6_tariff_suite(
    panel: pd.DataFrame, treatment_col: str, bootstrap_reps: int, threshold_reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run H6 at a pre-specified median and report outcome-selected threshold exploration separately."""

    required = ["log_export_recovery", "eci_pre_raw", treatment_col, "country_iso3_code", "year"]
    primary_sample = panel.dropna(subset=required).copy()
    country_eci = primary_sample[["country_iso3_code", "eci_pre_raw"]].drop_duplicates()
    prespecified_threshold = float(country_eci["eci_pre_raw"].median())

    quantiles = np.arange(0.20, 0.801, 0.05)
    exploratory_threshold, search, fits = select_h6_threshold(
        panel, "log_export_recovery", treatment_col, quantiles
    )
    distribution = wild_bootstrap_threshold_distribution(
        fits, exploratory_threshold, threshold_reps, seed + 1
    )
    summary = pd.DataFrame(
        [
            {
                "threshold_type": "pre_specified_pre_shock_median",
                "primary_outcome": "log_export_recovery",
                "primary_treatment": treatment_col,
                "threshold_eci_pre": prespecified_threshold,
                "threshold_ci_low_95_wild": np.nan,
                "threshold_ci_high_95_wild": np.nan,
                "threshold_bootstrap_reps": 0,
                "threshold_grid_quantiles": "not applicable",
            },
            {
                "threshold_type": "exploratory_outcome_selected_grid",
                "primary_outcome": "log_export_recovery",
                "primary_treatment": treatment_col,
                "threshold_eci_pre": exploratory_threshold,
                "threshold_ci_low_95_wild": float(distribution["threshold_eci_pre"].quantile(0.025)),
                "threshold_ci_high_95_wild": float(distribution["threshold_eci_pre"].quantile(0.975)),
                "threshold_bootstrap_reps": threshold_reps,
                "threshold_grid_quantiles": "0.20 to 0.80 by 0.05",
            },
        ]
    )
    specs: list[tuple[str, str, str, str, str | None, str | None, pd.DataFrame | None, list[str]]] = [
        ("primary_log_tariff_weighted", "Log export recovery", "log_export_recovery", treatment_col, None, None, None, []),
        ("pretrend_log_tariff_weighted", "Pretrend-adjusted log export deviation", "pretrend_adjusted_export_recovery", treatment_col, None, None, None, []),
        ("raw_ratio_tariff_weighted", "Raw export-recovery ratio", "export_recovery_index", treatment_col, None, None, None, []),
        ("winsorized_ratio_tariff_weighted", "Winsorized export-recovery ratio", "export_recovery_winsor_1_99", treatment_col, None, None, None, []),
        ("exclude_microeconomies_log", "Log export recovery excluding bottom baseline-export decile", "log_export_recovery", treatment_col, "not_bottom_decile_export_size", None, None, []),
        ("weighted_log_tariff_weighted", "Export-size-weighted log export recovery", "log_export_recovery", treatment_col, None, "export_size_weight", None, []),
        ("covered_basket_log", "Log export recovery, covered-US-basket sensitivity", "log_export_recovery", "s_tariff_weighted_us_covered_basket", None, None, None, []),
        ("china_network_log", "Log export recovery, China-network channel sensitivity", "log_export_recovery", "s_tariff_weighted_china_network", None, None, None, []),
        ("continuous_nexus_log", "Log export recovery, continuous nexus sensitivity", "log_export_recovery", "z_nexus_exposure_pre", None, None, None, []),
    ]
    controls_df, control_terms = add_pre_control_year_terms(panel)
    specs.append(
        (
            "pre_shock_controls_x_year_log",
            "Log export recovery with frozen pre-shock controls by year",
            "log_export_recovery",
            treatment_col,
            None,
            None,
            controls_df,
            control_terms,
        )
    )

    rows: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for index, (name, label, outcome, exposure, subset, weight, source_df, extra_terms) in enumerate(specs):
        source = panel if source_df is None else source_df
        d = source if subset is None else source[source[subset].eq(1)].copy()
        try:
            fit = fit_twfe(
                prepare_piecewise_data(d, prespecified_threshold, exposure),
                outcome,
                H6_TERMS + list(extra_terms),
                weight_col=weight,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"test_id": f"H6_{name}", "spec": name, "error": str(exc)})
            continue
        for row in h6_test_rows(
            fit,
            spec=name,
            outcome_label=label,
            exposure_label=exposure,
            threshold=prespecified_threshold,
            reps=bootstrap_reps,
            seed=seed + 100 + index * 50,
        ):
            row.update(
                {
                    "test_id": f"H6_{name}_{row['test']}",
                    "hypothesis": "H6 capability-conversion threshold",
                    "threshold_type": "pre_specified_pre_shock_median",
                    "multiplicity_family": "full_reported_family",
                    "pvalue_for_fdr": row.get("p_wild_bootstrap"),
                }
            )
            rows.append(row)
        for term in H6_TERMS:
            coefficients.append(
                {
                    "spec": name,
                    "outcome": label,
                    "term": term,
                    "threshold_type": "pre_specified_pre_shock_median",
                    "threshold_eci_pre": prespecified_threshold,
                    **fit.term_row(term),
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                    "r2_within": fit.r2_within,
                }
            )
    return pd.DataFrame(rows), summary, search, distribution, pd.DataFrame(coefficients)


def run_destination_mechanisms(
    panel: pd.DataFrame, treatment_col: str, bootstrap_reps: int, seed: int
) -> pd.DataFrame:
    prepared = add_tariff_linear_terms(panel, treatment_col)
    terms = ["tariff_treatment", "post_x_eci_pre", "eci_x_tariff_treatment"]
    outcomes = [
        ("New non-US/China destination entries", "new_non_uschina_destination_count"),
        ("New non-US/China destination export share", "new_non_uschina_destination_export_share"),
        ("Persistent new non-US/China destination entries", "persistent_new_non_uschina_destination_count"),
    ]
    rows = []
    for index, (label, outcome) in enumerate(outcomes):
        fit = fit_twfe(prepared, outcome, terms)
        rows.append(
            contrast_row(
                fit,
                target_contrast(fit, "eci_x_tariff_treatment"),
                test_id=f"mechanism_{outcome}",
                hypothesis="Direct market-search and destination-entry mechanism",
                outcome=label,
                expected_direction="positive",
                alternative="greater",
                bootstrap_reps=bootstrap_reps,
                seed=seed + index * 101,
                family="full_reported_family",
                extra={"treatment": treatment_col},
            )
        )
    return pd.DataFrame(rows)


def build_full_family(tables: list[pd.DataFrame]) -> pd.DataFrame:
    usable = []
    for table in tables:
        if table is None or table.empty or "pvalue_for_fdr" not in table:
            continue
        subset = table[table["pvalue_for_fdr"].notna()].copy()
        if not subset.empty:
            usable.append(subset)
    result = pd.concat(usable, ignore_index=True, sort=False)
    result["qvalue_bh_full_reported_family"] = bh_fdr_adjust(result["pvalue_for_fdr"])
    result["fdr_significant_0_05"] = (result["qvalue_bh_full_reported_family"] < 0.05).astype(int)
    return result


def attach_full_family_q_values(table: pd.DataFrame, full_family: pd.DataFrame) -> pd.DataFrame:
    """Attach the single authoritative BH result to each report table."""

    if table.empty or "test_id" not in table or "test_id" not in full_family:
        return table
    qcols = ["test_id", "qvalue_bh_full_reported_family", "fdr_significant_0_05"]
    qtable = full_family[qcols].drop_duplicates(subset=["test_id"])
    existing = [column for column in qcols[1:] if column in table]
    return table.drop(columns=existing).merge(qtable, on="test_id", how="left")


def plot_completed_design_results(
    out_dir: Path,
    panel: pd.DataFrame,
    confirmatory: pd.DataFrame,
    h4: pd.DataFrame,
    h5_and_wald: pd.DataFrame,
    h5_profile: pd.DataFrame,
    h6_summary: pd.DataFrame,
    e1: pd.DataFrame,
    e2: pd.DataFrame,
    e3: pd.DataFrame,
) -> None:
    """Create the three report-ready plots from the completed model tables."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def find_row(table: pd.DataFrame, test_id: str) -> pd.Series | None:
        if table.empty or "test_id" not in table:
            return None
        found = table[table["test_id"].eq(test_id)]
        return None if found.empty else found.iloc[0]

    primary = [
        ("H1", "Adverse forward-GVC deviation stability", "gvc_adverse_deviation_stability"),
        ("H2", "Log export recovery", "log_export_recovery"),
        ("H3", "Partner diversification excluding US and China", "partner_diversification_excl_us_china"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3), constrained_layout=True)
    styles = [
        ("H1-H3", "#173F5F"),
        ("H4 (COI moderation)", "#007C7A"),
        ("H5 direct GPR", "#D1495B"),
        ("H5 profile-matched GPR", "#8E8E8E"),
    ]
    for ax, (hypothesis, label, outcome) in zip(axes, primary):
        rows: list[tuple[str, str, pd.Series | None]] = [
            ("H1-H3", "#173F5F", find_row(confirmatory, f"{hypothesis}_{outcome}")),
            ("H4 (COI moderation)", "#007C7A", find_row(h4, f"H4_{outcome}")),
            ("H5 direct GPR", "#D1495B", find_row(h5_and_wald, f"H5_{outcome}")),
        ]
        profile_rows = h5_profile[h5_profile.get("outcome", pd.Series(dtype=str)).eq(label)]
        rows.append(("H5 profile-matched GPR", "#8E8E8E", None if profile_rows.empty else profile_rows.iloc[0]))
        usable = [(name, color, row) for name, color, row in rows if row is not None and pd.notna(row.get("estimate", np.nan))]
        positions = np.arange(len(usable))[::-1]
        for position, (name, color, row) in zip(positions, usable):
            low = max(float(row["estimate"] - row["ci_low_95"]), 0.0)
            high = max(float(row["ci_high_95"] - row["estimate"]), 0.0)
            ax.errorbar(
                float(row["estimate"]),
                position,
                xerr=np.array([[low], [high]]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=6,
                linewidth=1.5,
            )
        ax.axvline(0, color="#454545", linewidth=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels([name for name, _, _ in usable], fontsize=8.5)
        ax.set_title(f"{hypothesis}: {label}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Coefficient with 95% cluster CI", fontsize=9)
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Tariff-weighted confirmatory and moderation estimates", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "figure_completed_confirmatory_coefficients.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    threshold_row = h6_summary[
        h6_summary.get("threshold_type", pd.Series(dtype=str)).eq("pre_specified_pre_shock_median")
    ]
    if not threshold_row.empty:
        threshold = float(threshold_row.iloc[0]["threshold_eci_pre"])
        h6_data = prepare_piecewise_data(panel, threshold, "s_tariff_weighted_us_diversion")
        try:
            h6_fit = fit_twfe(h6_data, "log_export_recovery", H6_TERMS)
            values = panel["eci_pre_raw"].dropna()
            x = np.linspace(float(values.quantile(0.02)), float(values.quantile(0.98)), 250)
            below = np.minimum(x - threshold, 0.0)
            above = np.maximum(x - threshold, 0.0)
            effects = np.empty_like(x)
            standard_errors = np.empty_like(x)
            for index, (low, high) in enumerate(zip(below, above)):
                contrast = np.zeros(len(h6_fit.term_names), dtype=float)
                contrast[h6_fit.term_index("post_x_exposure")] = 1.0
                contrast[h6_fit.term_index("h6_low_slope")] = low
                contrast[h6_fit.term_index("h6_high_slope")] = high
                effects[index] = float(contrast @ h6_fit.beta)
                standard_errors[index] = np.sqrt(max(float(contrast @ h6_fit.covariance @ contrast), 0.0))
            critical = stats.t.ppf(0.975, h6_fit.df_clusters)
            fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
            ax.plot(x, effects, color="#176B87", linewidth=2.4)
            ax.fill_between(x, effects - critical * standard_errors, effects + critical * standard_errors, color="#B9E1EA", alpha=0.7)
            ax.axhline(0, color="#454545", linewidth=0.9)
            ax.axvline(threshold, color="#C84B31", linestyle="--", linewidth=1.4)
            ax.text(threshold, ax.get_ylim()[1], " Pre-specified median ECI", color="#A13A25", va="top", fontsize=9)
            ax.set_xlabel("Frozen pre-shock ECI (2015-2017 average)")
            ax.set_ylabel("Effect of +1 SD tariff-weighted diversion opportunity\non log export recovery")
            ax.set_title("H6 capability-conversion threshold: primary pre-specified breakpoint", fontweight="bold")
            ax.grid(axis="y", alpha=0.22)
            fig.savefig(out_dir / "figure_h6_tariff_weighted_threshold.png", dpi=240, bbox_inches="tight")
            plt.close(fig)
        except (ValueError, np.linalg.LinAlgError):
            pass

    heterogeneity = pd.concat([e1, e2, e3], ignore_index=True, sort=False)
    heterogeneity = heterogeneity[
        heterogeneity.get("estimate", pd.Series(dtype=float)).notna()
        & heterogeneity.get("ci_low_95", pd.Series(dtype=float)).notna()
        & heterogeneity.get("ci_high_95", pd.Series(dtype=float)).notna()
    ].copy()
    if not heterogeneity.empty:
        color_map = {"E1": "#3B6FB6", "E2": "#D9851E", "E3": "#7C5C9E"}
        heterogeneity["family"] = heterogeneity["test_id"].astype(str).str.extract(r"^(E[123])", expand=False)
        heterogeneity = heterogeneity.iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(heterogeneity) + 1.8), constrained_layout=True)
        positions = np.arange(len(heterogeneity))
        for position, (_, row) in zip(positions, heterogeneity.iterrows()):
            color = color_map.get(str(row["family"]), "#555555")
            low = max(float(row["estimate"] - row["ci_low_95"]), 0.0)
            high = max(float(row["ci_high_95"] - row["estimate"]), 0.0)
            ax.errorbar(float(row["estimate"]), position, xerr=np.array([[low], [high]]), fmt="o", color=color, ecolor=color, capsize=3)
        ax.axvline(0, color="#454545", linewidth=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(heterogeneity["test_id"].astype(str), fontsize=8.5)
        ax.set_xlabel("Coefficient with 95% cluster CI")
        ax.set_title("Targeted exploratory heterogeneity: pooled subgroup slopes and formal contrasts", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        fig.savefig(out_dir / "figure_targeted_exploratory_heterogeneity.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

def write_summary(
    out_dir: Path,
    confirmatory: pd.DataFrame,
    h4: pd.DataFrame,
    h5_and_wald: pd.DataFrame,
    h6: pd.DataFrame,
    e1: pd.DataFrame,
    e2: pd.DataFrame,
    e3: pd.DataFrame,
    mechanisms: pd.DataFrame,
    full_family: pd.DataFrame,
    threshold: pd.DataFrame,
) -> None:
    def best(table: pd.DataFrame, identifier: str) -> str:
        match = table[table.get("test_id", pd.Series(dtype=str)).eq(identifier)]
        if match.empty or "estimate" not in match:
            return "not estimable"
        row = match.iloc[0]
        return f"b={row['estimate']:.4f}, wild p={row.get('p_wild_bootstrap', np.nan):.3f}"

    h6a = h6[h6.get("test", pd.Series(dtype=str)).eq("H6a_beta_low_lt_0")]
    h6b = h6[h6.get("test", pd.Series(dtype=str)).eq("H6b_beta_high_minus_low_gt_0")]
    h6a_primary = h6a[h6a.get("spec", pd.Series(dtype=str)).eq("primary_log_tariff_weighted")]
    h6b_primary = h6b[h6b.get("spec", pd.Series(dtype=str)).eq("primary_log_tariff_weighted")]

    def h6_result_text(table: pd.DataFrame) -> str:
        if table.empty:
            return "not estimable"
        row = table.iloc[0]
        return f"b={row['estimate']:.4f}, wild p={row['p_wild_bootstrap']:.3f}"

    pre_specified = threshold[
        threshold.get("threshold_type", pd.Series(dtype=str)).eq("pre_specified_pre_shock_median")
    ]
    exploratory = threshold[
        threshold.get("threshold_type", pd.Series(dtype=str)).eq("exploratory_outcome_selected_grid")
    ]
    if pre_specified.empty:
        pre_text = "not estimable"
    else:
        pre_text = f"{pre_specified.iloc[0]['threshold_eci_pre']:.4f} (pre-shock median; fixed before outcome estimation)"
    if exploratory.empty:
        exploratory_text = "not estimated"
    else:
        row = exploratory.iloc[0]
        exploratory_text = (
            f"{row['threshold_eci_pre']:.4f} "
            f"(wild 95% selection interval {row['threshold_ci_low_95_wild']:.4f} to {row['threshold_ci_high_95_wild']:.4f})"
        )
    h4_joint = h5_and_wald[h5_and_wald.get("test_id", pd.Series(dtype=str)).eq("H4_omnibus")]
    h4_joint_text = (
        "not estimable"
        if h4_joint.empty
        else f"cluster Wald p={h4_joint.iloc[0]['p_cluster']:.3f}"
    )

    lines = [
        "# Completed Design Analysis",
        "",
        "## Purpose",
        "This package completes the six redesign tasks without replacing earlier results. It freezes pre-shock constructs, uses a product-overlap tariff treatment, corrects outcome definitions, uses pooled formal E1-E3 contrasts, reports configured country wild-bootstrap inference, and applies Benjamini-Hochberg correction across every reported inferential test.",
        "",
        "## Design Decisions",
        "- The causal sample excludes the United States and China because they are the direct policy parties; 229 third countries remain in the panel before outcome-specific missingness.",
        "- Income group is the modal World Bank analytical classification over 2015-2017, with rare ties resolved using 2017. ECI and COI are frozen 2015-2017 averages; regime cut points are frozen pre-shock terciles.",
        "- The primary treatment is the annual Section 301 duty weighted by a country's pre-shock affected US export basket and China's pre-shock US-market share at HS6. It measures diversion opportunity, not a tariff paid by the third country.",
        "- The China-import product-overlap channel and raw covered-US-basket channel are supplementary. HS6 remains an any-covered-HTS8 approximation, fully audited in the tariff-line files.",
        "- H5 uses directly observed country GPR as the primary exploratory analysis. A same-historical-income-prioritised KNN profile match is reported separately as a no-outcome-leakage sensitivity, not as a replacement for observed data.",
        "",
        "## Confirmatory Estimates",
        f"- H1: {best(confirmatory, 'H1_gvc_adverse_deviation_stability')}",
        f"- H2: {best(confirmatory, 'H2_log_export_recovery')}",
        f"- H3: {best(confirmatory, 'H3_partner_diversification_excl_us_china')}",
        f"- H4: minimum outcome-specific wild p={h4.get('p_wild_bootstrap', pd.Series(dtype=float)).min() if not h4.empty else np.nan:.3f}; valid stacked joint test: {h4_joint_text}.",
        "- H5 remains exploratory because direct country GPR coverage is limited; both direct-observation and profile-matched sensitivity estimates are supplied.",
        "",
        "## H6 Threshold",
        f"- Primary breakpoint: {pre_text}.",
        f"- Outcome-selected breakpoint (exploratory only): {exploratory_text}.",
        f"- H6a, low-regime ECI slope < 0: {h6_result_text(h6a_primary)}",
        f"- H6b, high-minus-low ECI slope > 0: {h6_result_text(h6b_primary)}",
        "",
        "## Targeted Exploratory Heterogeneity",
        f"- E1, upper-middle-income GVC slope: {best(e1, 'E1_UMC_slope')}; use E1_upper_middle_income_gvc_omnibus.csv and the contrast rows for the formal subgroup test.",
        f"- E2, low-pre-shock-ECI export-recovery slope: {best(e2, 'E2_low_slope')}; use E2_low_eci_export_recovery_omnibus.csv and the contrast rows for the formal subgroup test.",
        f"- E3, high-income direct-GPR diversification moderation: {best(e3, 'E3_direct_observation_HIC_slope')}; the profile-matched sensitivity is separate and explicitly labelled.",
        "- A within-group p-value alone is never treated as a subgroup finding: use the pooled coefficient-difference rows and their full-family q-values.",
        "",
        "## Multiplicity And Mechanism",
        f"- Complete reported family size: {len(full_family)} tests.",
        f"- Tests with BH q < .05: {int(full_family['fdr_significant_0_05'].sum()) if not full_family.empty else 0}.",
        "- The destination-entry models directly test market search and redirection rather than inferring it only from aggregate diversification. See destination_entry_mechanism_tests.csv.",
        "",
        "## Interpretation Rule",
        "Do not elevate H6 or an exploratory subgroup to primary support unless its directional tests, formal group-difference tests, outcome corrections, and multiplicity-adjusted evidence converge. The report preserves null and contrary results alongside favorable point estimates.",
    ]
    (out_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_methods_addendum(out_dir: Path) -> None:
    text = r"""% Generated methods addendum for the completed design analysis.
\subsection{Completed Pre-shock and Tariff-weighted Design}
All capability constructs are fixed before the tariff shock. Baseline ECI and COI are country averages over 2015--2017. Historical income group is the modal World Bank analytical classification over those years, with a 2017 tie-break rule. Low, middle, and high complexity regimes are fixed pre-shock ECI terciles. The causal sample excludes the United States and China because they are the direct policy parties rather than third-country comparison units.

Our primary treatment is a continuous tariff-weighted US-market diversion opportunity. For each third country, we calculate the 2015--2017 share of its US exports in Section 301-affected HS6 products and weight each product by China's corresponding pre-shock share of US imports. The fixed product profile is multiplied by the annual-average additional Section 301 duty. This construct captures the tariff-induced opportunity to displace Chinese suppliers, not a duty paid by the third country. A raw affected-basket measure and a China-import product-overlap measure are reported as supplementary channels. Because BACI is HS6 and the legal schedules are HTS8 or more detailed, all treatment measures are any-covered-line HS6 approximations; the replication package reports coverage, overlap, line-count, and source-hash audits.

GVC stability is measured primarily as the adverse deviation from a country's 2015--2017 forward-linkage mean, \(\min(FwdLinkage_{it}-\overline{FwdLinkage}_{i,pre},0)\), so values closer to zero indicate that a country avoided an erosion of its pre-shock GVC position. Signed annual linkage change and absolute deviation are reported as sensitivity outcomes. Export recovery is the logarithm of current exports relative to the fixed 2015--2017 export baseline; with country fixed effects, this is equivalent to modelling log exports with a country-specific baseline removed. Partner diversification excludes the United States and China and is measured primarily as one minus the destination-share HHI, with entropy and effective-destination-count sensitivity outcomes.

For outcome \(Y_{it}\), the primary model is
\begin{equation}
Y_{it}=\beta_1 T_{it}+\beta_2(ECI_i^{pre}\times Post_t)+\beta_3(ECI_i^{pre}\times T_{it})+\alpha_i+\lambda_t+\varepsilon_{it},
\end{equation}
where \(T_{it}\) is the scaled tariff-weighted diversion treatment. Country and year fixed effects are included. Inference uses country-clustered standard errors and 999-replication Rademacher wild-cluster bootstrap p-values in the final run. A sensitivity permits the frozen pre-shock GDP per capita, trade openness, institutional-quality, and resource-rent profiles to have separate year effects, avoiding controls that may themselves be affected by the shock.

H6 is evaluated in a pooled segmented model at the pre-specified median of frozen ECI, with the required tests \(\beta_L<0\) and \(\beta_H-\beta_L>0\). An outcome-selected threshold grid and a 999-replication selection distribution are reported only as exploratory localization, not as primary confirmatory evidence. E1 and E2 use single pooled group-interaction models with formal coefficient-difference contrasts. E3 is estimated first using direct country GPR observations; a same-historical-income-prioritised KNN profile match based only on frozen pre-shock country characteristics provides a supplementary missing-data sensitivity. Benjamini--Hochberg q-values are calculated over the complete reported confirmatory, sensitivity, H6, E1--E3, and mechanism test family.
"""
    (out_dir / "completed_design_methods_addendum.tex").write_text(text, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the completed geo-SCM redesign analysis.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--threshold-bootstrap-reps", type=int, default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--rebuild-tariff-profile",
        action="store_true",
        help="Stream BACI again even when the audited pre-shock profile cache exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    source_dir = base_dir / "data" / "raw" / SOURCE_DIR_NAME
    out_dir = base_dir / "reports" / "final_design_completion"
    ensure_dir(out_dir)
    processed_dir = base_dir / "data" / "processed"

    panel = pd.read_csv(processed_dir / "regression_panel_2012_2022.csv")
    bilateral = pd.read_csv(processed_dir / "source_atlas_country_country_year_2012_2022.csv")
    income, income_audit = historical_income_groups(
        source_dir / "world_bank_historical_income_classifications.xlsx"
    )
    regions_path = processed_dir / "world_bank_country_regions.csv"
    regions = pd.read_csv(regions_path) if regions_path.exists() else None
    if regions is not None and "wb_region" in regions:
        regions["wb_region"] = regions["wb_region"].astype("string").str.strip()

    analysis_panel, frozen = freeze_pre_shock_constructs(panel, bilateral, income, regions)
    analysis_panel = add_corrected_gvc_outcomes(analysis_panel)
    analysis_panel, frozen, regime_thresholds = add_frozen_regimes(frozen, analysis_panel)

    tariff_lines, membership, tariff_audit = build_tariff_schedule(source_dir)
    rates = annual_tariff_rates(analysis_panel["year"].unique())
    profile_cache = processed_dir / "tariff_weighted_channel_exposure_2015_2017.csv"
    previous_baci_audit = out_dir / "baci_tariff_profile_audit.csv"
    if profile_cache.exists() and not args.rebuild_tariff_profile:
        profiles = pd.read_csv(profile_cache)
        if previous_baci_audit.exists():
            baci_audit = pd.read_csv(previous_baci_audit)
            baci_audit = baci_audit[baci_audit["metric"].ne("tariff_profile_cache_reused")].copy()
        else:
            baci_audit = pd.DataFrame()
        baci_audit = pd.concat(
            [
                baci_audit,
                pd.DataFrame([{"metric": "tariff_profile_cache_reused", "value": 1}]),
            ],
            ignore_index=True,
        )
    else:
        profiles, baci_audit = build_tariff_weighted_exposure(
            source_dir / "BACI_HS12_V202601.zip", membership
        )
        profiles.to_csv(profile_cache, index=False)
        baci_audit = pd.concat(
            [
                baci_audit,
                pd.DataFrame([{"metric": "tariff_profile_cache_reused", "value": 0}]),
            ],
            ignore_index=True,
        )
        baci_audit.to_csv(previous_baci_audit, index=False)
    analysis_panel, frozen = attach_tariff_exposure(analysis_panel, frozen, profiles, rates)
    analysis_panel["analysis_eligible_third_country"] = (
        ~analysis_panel["country_iso3_code"].isin(["USA", "CHN"])
    ).astype(int)
    sample_audit = pd.DataFrame(
        [
            {"metric": "panel_countries_before_direct_party_exclusion", "value": int(analysis_panel["country_iso3_code"].nunique())},
            {"metric": "excluded_direct_policy_parties", "value": "USA, CHN"},
            {"metric": "analysis_countries_after_direct_party_exclusion", "value": int(analysis_panel.loc[analysis_panel["analysis_eligible_third_country"].eq(1), "country_iso3_code"].nunique())},
            {"metric": "analysis_rows_after_direct_party_exclusion", "value": int(analysis_panel["analysis_eligible_third_country"].sum())},
        ]
    )
    analysis_panel = analysis_panel[analysis_panel["analysis_eligible_third_country"].eq(1)].copy()

    # The fixed-baseline panel comes with these corrected outcomes; recalculate
    # two non-linear recovery alternatives after the final merges.
    analysis_panel["log_export_recovery"] = np.where(
        (analysis_panel["export_value"] > 0) & (analysis_panel["baseline_export_2015_2017"] > 0),
        np.log(analysis_panel["export_value"]) - np.log(analysis_panel["baseline_export_2015_2017"]),
        np.nan,
    )
    analysis_panel["export_recovery_winsor_1_99"] = winsorize(analysis_panel["export_recovery_index"])
    analysis_panel["pretrend_adjusted_export_recovery"] = make_pretrend_adjusted_recovery(analysis_panel)

    treatment = "s_tariff_weighted_us_diversion"
    confirmatory, equivalence, h4, h5_and_h4_wald = run_confirmatory_models(
        analysis_panel, treatment, int(args.bootstrap_reps), int(args.seed) + 1000
    )
    h6, h6_summary, h6_search, h6_distribution, h6_coefficients = run_h6_tariff_suite(
        analysis_panel,
        treatment,
        int(args.bootstrap_reps),
        int(args.threshold_bootstrap_reps),
        int(args.seed) + 2000,
    )
    gpr_validation_detail, gpr_validation = validate_profile_gpr(analysis_panel)
    selected_gpr_k = (
        int(gpr_validation.sort_values(["rmse", "k"]).iloc[0]["k"])
        if not gpr_validation.empty
        else 5
    )
    analysis_panel, gpr_imputed_values = add_profile_matched_gpr(analysis_panel, selected_gpr_k)
    gpr_coverage = pd.DataFrame(
        [
            {
                "gpr_series": "direct_country_observation",
                "n_rows": int(analysis_panel["gpr_country_annual"].notna().sum()),
                "n_countries": int(analysis_panel.loc[analysis_panel["gpr_country_annual"].notna(), "country_iso3_code"].nunique()),
            },
            {
                "gpr_series": f"profile_matched_knn_k{selected_gpr_k}",
                "n_rows": int(analysis_panel["gpr_profile_matched"].notna().sum()),
                "n_countries": int(analysis_panel.loc[analysis_panel["gpr_profile_matched"].notna(), "country_iso3_code"].nunique()),
                "imputed_rows": int(analysis_panel["gpr_profile_matched_imputed"].sum()),
            },
        ]
    )
    h5_profile = run_h5_profile_sensitivity(
        analysis_panel,
        treatment,
        "gpr_profile_matched",
        f"profile_matched_knn_k{selected_gpr_k}",
        int(args.bootstrap_reps),
        int(args.seed) + 3000,
    )
    e1_tests, e1_omnibus = run_targeted_eci_group_test(
        analysis_panel,
        group_col="wb_income_id",
        target_group="UMC",
        outcome="gvc_adverse_deviation_stability",
        outcome_label="Adverse forward-GVC deviation stability",
        candidate_id="E1",
        treatment_col=treatment,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 4000,
    )
    e2_tests, e2_omnibus = run_targeted_eci_group_test(
        analysis_panel,
        group_col="eci_pre_regime",
        target_group="low",
        outcome="log_export_recovery",
        outcome_label="Log export recovery",
        candidate_id="E2",
        treatment_col=treatment,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 5000,
    )
    e3_observed_tests, e3_observed_omnibus = run_targeted_gpr_income_test(
        analysis_panel,
        gpr_col="gpr_country_annual",
        sample_label="direct_observation",
        treatment_col=treatment,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 6000,
    )
    e3_profile_tests, e3_profile_omnibus = run_targeted_gpr_income_test(
        analysis_panel,
        gpr_col="gpr_profile_matched",
        sample_label=f"profile_matched_knn_k{selected_gpr_k}",
        treatment_col=treatment,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 7000,
    )
    control_sensitivity = run_baseline_control_sensitivity(
        analysis_panel, treatment, int(args.bootstrap_reps), int(args.seed) + 8000
    )
    mechanisms = run_destination_mechanisms(
        analysis_panel, treatment, int(args.bootstrap_reps), int(args.seed) + 9000
    )

    full_family = build_full_family(
        [
            confirmatory,
            equivalence,
            h4,
            h5_and_h4_wald,
            h5_profile,
            h6,
            e1_tests,
            e1_omnibus,
            e2_tests,
            e2_omnibus,
            e3_observed_tests,
            e3_observed_omnibus,
            e3_profile_tests,
            e3_profile_omnibus,
            control_sensitivity,
            mechanisms,
        ]
    )
    confirmatory = attach_full_family_q_values(confirmatory, full_family)
    equivalence = attach_full_family_q_values(equivalence, full_family)
    h4 = attach_full_family_q_values(h4, full_family)
    h5_and_h4_wald = attach_full_family_q_values(h5_and_h4_wald, full_family)
    h5_profile = attach_full_family_q_values(h5_profile, full_family)
    h6 = attach_full_family_q_values(h6, full_family)
    e1_tests = attach_full_family_q_values(e1_tests, full_family)
    e1_omnibus = attach_full_family_q_values(e1_omnibus, full_family)
    e2_tests = attach_full_family_q_values(e2_tests, full_family)
    e2_omnibus = attach_full_family_q_values(e2_omnibus, full_family)
    e3_observed_tests = attach_full_family_q_values(e3_observed_tests, full_family)
    e3_observed_omnibus = attach_full_family_q_values(e3_observed_omnibus, full_family)
    e3_profile_tests = attach_full_family_q_values(e3_profile_tests, full_family)
    e3_profile_omnibus = attach_full_family_q_values(e3_profile_omnibus, full_family)
    control_sensitivity = attach_full_family_q_values(control_sensitivity, full_family)
    mechanisms = attach_full_family_q_values(mechanisms, full_family)

    income.to_csv(out_dir / "historical_income_groups_2015_2017.csv", index=False)
    income_audit.to_csv(out_dir / "historical_income_group_audit.csv", index=False)
    frozen.to_csv(out_dir / "frozen_pre_shock_constructs_completed.csv", index=False)
    regime_thresholds.to_csv(out_dir / "frozen_regime_thresholds.csv", index=False)
    analysis_panel.to_csv(out_dir / "panel_with_completed_design_constructs.csv", index=False)
    tariff_lines.to_csv(out_dir / "ustr_section301_tariff_lines_hs8.csv", index=False)
    membership.to_csv(out_dir / "ustr_section301_hs6_membership_audit.csv", index=False)
    tariff_audit.to_csv(out_dir / "ustr_tariff_line_extraction_audit.csv", index=False)
    rates.to_csv(out_dir / "annual_section301_rate_schedule.csv", index=False)
    profiles.to_csv(processed_dir / "tariff_weighted_channel_exposure_2015_2017.csv", index=False)
    baci_audit.to_csv(out_dir / "baci_tariff_profile_audit.csv", index=False)
    sample_audit.to_csv(out_dir / "third_country_sample_audit.csv", index=False)
    pd.concat([income_audit, tariff_audit, baci_audit, sample_audit, gpr_coverage], ignore_index=True, sort=False).to_csv(out_dir / "source_and_construction_audit.csv", index=False)
    confirmatory.to_csv(out_dir / "confirmatory_tariff_weighted_tests.csv", index=False)
    equivalence.to_csv(out_dir / "equivalence_and_mde_tariff_weighted.csv", index=False)
    h4.to_csv(out_dir / "h4_tariff_weighted_tests.csv", index=False)
    h5_and_h4_wald.to_csv(out_dir / "h5_and_h4_omnibus_tests.csv", index=False)
    h6.to_csv(out_dir / "h6_tariff_weighted_tests.csv", index=False)
    h6_summary.to_csv(out_dir / "h6_tariff_weighted_threshold_summary.csv", index=False)
    h6_search.to_csv(out_dir / "h6_tariff_weighted_threshold_search.csv", index=False)
    h6_distribution.to_csv(out_dir / "h6_tariff_weighted_threshold_bootstrap.csv", index=False)
    h6_coefficients.to_csv(out_dir / "h6_tariff_weighted_coefficients.csv", index=False)
    gpr_validation_detail.to_csv(out_dir / "gpr_profile_matching_validation_detail.csv", index=False)
    gpr_validation.to_csv(out_dir / "gpr_profile_matching_validation.csv", index=False)
    gpr_imputed_values.to_csv(out_dir / "gpr_profile_matched_imputed_values.csv", index=False)
    gpr_coverage.to_csv(out_dir / "gpr_coverage_audit.csv", index=False)
    h5_profile.to_csv(out_dir / "h5_profile_matched_gpr_sensitivity.csv", index=False)
    e1_tests.to_csv(out_dir / "E1_upper_middle_income_gvc_tests.csv", index=False)
    e1_omnibus.to_csv(out_dir / "E1_upper_middle_income_gvc_omnibus.csv", index=False)
    e2_tests.to_csv(out_dir / "E2_low_eci_export_recovery_tests.csv", index=False)
    e2_omnibus.to_csv(out_dir / "E2_low_eci_export_recovery_omnibus.csv", index=False)
    e3_observed_tests.to_csv(out_dir / "E3_high_income_gpr_diversification_observed.csv", index=False)
    e3_observed_omnibus.to_csv(out_dir / "E3_high_income_gpr_diversification_observed_omnibus.csv", index=False)
    e3_profile_tests.to_csv(out_dir / "E3_high_income_gpr_diversification_profile_matched.csv", index=False)
    e3_profile_omnibus.to_csv(out_dir / "E3_high_income_gpr_diversification_profile_matched_omnibus.csv", index=False)
    control_sensitivity.to_csv(out_dir / "frozen_pre_shock_controls_by_year_sensitivity.csv", index=False)
    mechanisms.to_csv(out_dir / "destination_entry_mechanism_tests.csv", index=False)
    full_family.to_csv(out_dir / "full_reported_multiplicity_family.csv", index=False)

    metadata = {
        "analysis_type": "completed_pre_shock_tariff_weighted_design",
        "baseline_years": [BASELINE_START, BASELINE_END],
        "policy_start_year": POLICY_START,
        "primary_treatment": treatment,
        "primary_treatment_interpretation": "HS6 tariff-weighted US-market diversion opportunity",
        "income_group_rule": "World Bank modal analytical classification over 2015-2017; latest-year tie break",
        "regime_rule": "2015-2017 ECI/COI terciles",
        "wild_cluster_bootstrap_reps": int(args.bootstrap_reps),
        "threshold_bootstrap_reps": int(args.threshold_bootstrap_reps),
        "full_reported_family_tests": int(len(full_family)),
        "hs6_tariff_mapping_caveat": "Any-covered-HTS8 HS6 approximation; not an exact tariff-liability measure.",
        "direct_policy_parties_excluded_from_causal_sample": ["USA", "CHN"],
        "h5_status": "exploratory; direct-country-observation analysis plus profile-matched KNN sensitivity",
        "gpr_profile_matching": f"same-historical-income-prioritised KNN, k={selected_gpr_k}; no outcome variables used",
        "direct_mechanism": "new and persistent non-US/China destination entry",
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    plot_completed_design_results(
        out_dir,
        analysis_panel,
        confirmatory,
        h4,
        h5_and_h4_wald,
        h5_profile,
        h6_summary,
        e1_tests,
        e2_tests,
        e3_observed_tests,
    )
    write_summary(out_dir, confirmatory, h4, h5_and_h4_wald, h6, e1_tests, e2_tests, e3_observed_tests, mechanisms, full_family, h6_summary)
    write_methods_addendum(out_dir)
    print(f"Wrote completed-design package: {out_dir}")
    print(f"Primary tariff treatment: {treatment}")
    print(f"Full multiplicity family: {len(full_family)} tests")


if __name__ == "__main__":
    main()
