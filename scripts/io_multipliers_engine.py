from __future__ import annotations

import sys
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional, Union, List

# Ensure project root is on sys.path for importing app modules
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.smart_read import smart_read

GeoScope = Literal["all_provinces", "within_province", "national"]
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_LATEST = Path("data/latest")
# Using .parquet for optimized memory usage and faster loads
MULT_FP = DATA_LATEST / "io_multipliers_standardized.parquet"
SU_SUBSET_FP = DATA_LATEST / "io_supply_use_subset.csv"

# --- BASKETS CONFIG PATH (ROBUST FIX) ---
# Resolve path relative to this script file to ensure stability
# Structure assumed: [root]/scripts/this_file.py -> [root]/config/io_baskets.yml
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASKETS_FP = BASE_DIR / "config" / "io_baskets.yml"
except NameError:
    # Fallback for interactive/notebook environments where __file__ is undefined
    BASKETS_FP = Path("config/io_baskets.yml")

Scope = Literal["direct", "direct_indirect", "direct_indirect_induced", "Total", "total", "Direct", "Simple"]

SCOPE_TO_MULT_TYPE: dict[str, str] = {
    "direct": "Direct multiplier",
    "Direct": "Direct multiplier",
    "direct_indirect": "Simple multiplier",
    "Simple": "Simple multiplier",
    "direct_indirect_induced": "Total multiplier",
    "Total": "Total multiplier",
    "total": "Total multiplier",
}

def normalize_code_simple(code: str) -> str:
    """Helper to strip prefixes from YAML config to match data."""
    s = str(code).upper().strip()
    for prefix in ["BS", "GS", "NP"]:
        s = s.replace(prefix, "")
    return s.rstrip("0")

@lru_cache(maxsize=1)
def load_multipliers() -> pd.DataFrame:
    """Loads the standardized multipliers file.

    Memory optimisation: string columns are converted to ``pd.Categorical``
    because they have very few unique values (3-250) repeated across 5.2 M
    rows.  This cuts in-memory size from ~2.8 GB to ~83 MB — essential for
    Streamlit Community Cloud's ~1 GB limit.
    """
    dtypes_dict = {
        'YEAR': 'Int32',
        'GEO': 'category',
        'industry_code': 'category',
        'join_code': 'category',
        'industry_name': 'category',
        'multiplier_type': 'category',
        'variable': 'category',
        'value': 'float32'
    }

    try:
        df = smart_read(MULT_FP, dtype=dtypes_dict)
    except FileNotFoundError:
        alt_fp = MULT_FP.with_suffix("")
        if alt_fp.exists():
            df = smart_read(alt_fp, dtype=dtypes_dict)
        else:
            return pd.DataFrame()

    # --- THE GHOSTBUSTER FIX ---
    # Immediately drop rows where the value is missing (NaN).
    # This removes legacy rows (like 1114 in 2014+) so they don't trigger overlap warnings.
    if not df.empty and "value" in df.columns:
        df = df.dropna(subset=["value"])

    # --- GEO SCOPE TAGGING (replaces aggressive deduplication) ---
    # StatCan Table 36-10-0595 publishes TWO rows per provincial metric:
    #   Row 1 = "All provinces" (includes inter-provincial supply chain)
    #   Row 2 = "Within province" (impacts retained within the province)
    # We tag each row so the UI can let the user choose.
    if not df.empty:
        subset_cols = ['YEAR', 'GEO', 'join_code', 'variable', 'multiplier_type']
        existing_cols = [c for c in subset_cols if c in df.columns]

        if len(existing_cols) == 5:
            # Mark duplicates: first occurrence = all_provinces, second = within_province
            is_dup = df.duplicated(subset=existing_cols, keep='first')
            has_dup = df.duplicated(subset=existing_cols, keep=False)
            # Rows that ARE duplicated and are the first occurrence
            df['geo_scope'] = 'national'  # default for non-duplicated rows
            df.loc[has_dup & ~is_dup, 'geo_scope'] = 'all_provinces'
            df.loc[is_dup, 'geo_scope'] = 'within_province'
        else:
            df['geo_scope'] = 'national'

    # --- MEMORY OPTIMISATION ---
    # String columns have very few unique values (3-250) but 5.2M rows.
    # We now enforce 'category' dtypes directly in pd.read_csv() above, 
    # which prevents the 3GB intermediate memory spike during loading,
    # and reduces the final size to ~85 MB.
    if not df.empty and 'geo_scope' in df.columns:
        df['geo_scope'] = df['geo_scope'].astype('category')

    # Add MultiIndex for fast loc-based filtering
    if not df.empty:
        df.set_index(["GEO", "multiplier_type"], drop=False, inplace=True)
        df.sort_index(inplace=True)

    return df

@lru_cache(maxsize=1)
def load_supply_use_subset() -> pd.DataFrame:
    """Loads the historical output/supply data."""
    if not SU_SUBSET_FP.exists(): return pd.DataFrame()
    return smart_read(SU_SUBSET_FP)

@lru_cache(maxsize=1)
def _get_actual_output_lookup() -> pd.Series:
    """Pre-aggregates supply use data for O(1) lookups."""
    df = load_supply_use_subset()
    if df.empty: return pd.Series()
    # Ensure types for safe multi-index lookup
    df['GEO'] = df['GEO'].astype(str)
    df['YEAR'] = df['YEAR'].astype(int)
    df['join_code'] = df['join_code'].astype(str)
    return df.groupby(['GEO', 'YEAR', 'join_code'])['VALUE'].sum()

@lru_cache(maxsize=1)
def load_baskets_config() -> dict:
    """Loads industry baskets from YAML with fallback pathing."""
    # FIX: Added encoding="utf-8" to handle emojis (⚠️) in the config file
    # 1. Try default relative path
    if BASKETS_FP.exists():
        with open(BASKETS_FP, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("baskets", {})
            
    # 2. Try looking one level up (common issue in Streamlit cloud vs local)
    fallback = Path("../config/io_baskets.yml")
    if fallback.exists():
        with open(fallback, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("baskets", {})
            
    return {}


@lru_cache(maxsize=1)
def load_oag_config() -> dict:
    """Loads OAG ground-truth data config for apple/tender fruit calibration."""
    oag_fp = BASE_DIR / "config" / "oag_apple_data.yml"
    if oag_fp.exists():
        with open(oag_fp, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# --- HELPER FUNCTIONS ---

def available_geographies() -> List[str]:
    df = load_multipliers()
    if df.empty: return []
    if "GEO" not in df.columns: return []
    return sorted(df["GEO"].unique().tolist())

def available_years(geo: Optional[str] = None) -> List[int]:
    df = load_multipliers()
    if df.empty: return []
    if "YEAR" not in df.columns: return []
    if geo:
        df = df[df["GEO"] == geo]
    return sorted(df["YEAR"].unique().astype(int).tolist())

def available_industries(geo: str, year: int) -> List[str]:
    df = load_multipliers()
    if df.empty: return []
    
    mask = (df["GEO"] == geo) & (df["YEAR"] == int(year))
    col = "join_code" if "join_code" in df.columns else "industry_code"
    
    subset = df.loc[mask, [col, "industry_name"]].drop_duplicates()
    if subset.empty: return []
    
    return sorted([f"{r['industry_name']} [{r[col]}]" for _, r in subset.iterrows()])

@lru_cache(maxsize=32)
def industries_for_basket(basket_key: str) -> pd.DataFrame:
    baskets = load_baskets_config()
    conf = baskets.get(basket_key)
    if not conf: return pd.DataFrame()

    df = load_multipliers()
    if df.empty: return pd.DataFrame()
    
    all_rows = df[["industry_name", "join_code"]].drop_duplicates()
    
    match_conf = conf.get("match", {})
    prefixes = [normalize_code_simple(p) for p in match_conf.get("any_code_prefix", [])]
    exacts = [normalize_code_simple(e) for e in match_conf.get("any_code_exact", [])]
    
    ex_conf = conf.get("exclude", {})
    ex_exacts = [normalize_code_simple(e) for e in ex_conf.get("any_code_exact", [])]
    
    matches = []
    for _, row in all_rows.iterrows():
        code = str(row["join_code"])
        
        if code in ex_exacts: continue
            
        is_match = False
        if code in exacts: is_match = True
        elif any(code.startswith(str(p)) for p in prefixes): is_match = True
            
        if is_match:
            matches.append(row)

    # --- PARENT DOMINANCE FILTER (Anti-Double-Counting) ---
    # If we have selected both "312" (Parent) and "3121" (Child),
    # we must drop the Child to avoid summing the same money twice.
    
    if matches:
        # 1. Extract just the codes
        selected_codes = set(str(row["join_code"]) for row in matches)
        final_matches = []
        
        for row in matches:
            code = str(row["join_code"])
            # Check if this code is a "child" of another code currently in our selection
            # (e.g., Drop "3121" if "312" is also in the list)
            is_child = False
            for other in selected_codes:
                if other != code and code.startswith(other):
                    is_child = True
                    break
            
            if not is_child:
                final_matches.append(row)
        
        matches = final_matches

    return pd.DataFrame(matches).sort_values("join_code")

def get_available_historical_codes(geo: str, year: int) -> List[str]:
    df = load_supply_use_subset()
    if df.empty: return []
    mask = (df["YEAR"] == int(year)) & (df["GEO"] == geo)
    return df.loc[mask, "join_code"].unique().tolist()

def get_actual_output(geo: str, year: int, join_codes: List[str]) -> float:
    lookup = _get_actual_output_lookup()
    if lookup.empty: return 0.0

    total = 0.0
    for c in join_codes:
        key = (str(geo), int(year), str(c))
        if key in lookup.index:
            total += float(lookup.loc[key])
            
    return total

def compute_impacts(shock: Union[float, dict], geo: str, year: Union[int, None], scope: Scope, join_codes: List[str], geo_scope: GeoScope = "all_provinces"):
    df = load_multipliers()
    if df.empty: return pd.DataFrame(), pd.DataFrame()

    if scope not in SCOPE_TO_MULT_TYPE:
        raise ValueError(f"Unrecognized scope '{scope}'. Valid options: {list(SCOPE_TO_MULT_TYPE.keys())}")
    mtype = SCOPE_TO_MULT_TYPE[scope]

    # 1. Filter by Geography and Industry using Fast Index
    try:
        geo_subset = df.loc[(geo, mtype)]
        # loc can return a Series if there's only 1 row, ensure DataFrame
        if isinstance(geo_subset, pd.Series):
            geo_subset = geo_subset.to_frame().T
    except KeyError:
        return pd.DataFrame(), pd.DataFrame()

    mask = geo_subset["join_code"].isin(join_codes)
    
    # 0. Filter by geo_scope (all_provinces vs within_province)
    if geo != "Canada" and 'geo_scope' in geo_subset.columns:
        mask = mask & geo_subset['geo_scope'].isin([geo_scope, 'national'])

    broad_subset = geo_subset[mask].copy() 
    if broad_subset.empty: return pd.DataFrame(), pd.DataFrame()

    # 2. SMART ROW SELECTION (Per Industry)
    final_rows = []
    target_year = int(year) if year is not None else None
    # Q5 FIX: Collect per-code best_year BEFORE overwriting YEAR,
    # so weight lookups use the actual data vintage, not a future target.
    code_weight_year: dict[str, int] = {}

    for code in join_codes:
        ind_data = broad_subset[broad_subset["join_code"] == code]
        if ind_data.empty: continue
        
        avail_years = sorted(ind_data["YEAR"].unique())
        
        if target_year and target_year in avail_years:
            best_year = target_year
        elif avail_years:
            best_year = avail_years[-1]
        else:
            continue

        code_weight_year[code] = best_year  # Q5: remember actual vintage
        selected = ind_data[ind_data["YEAR"] == best_year].copy()
        if target_year:
            selected["YEAR"] = target_year
            
        final_rows.append(selected)
    
    if not final_rows: return pd.DataFrame(), pd.DataFrame()
    
    subset = pd.concat(final_rows)

    rows_dfs = []
    for yr, group_by_year in subset.groupby("YEAR"):
        unique_codes = group_by_year["join_code"].unique()
        n = len(unique_codes)
        if n == 0: continue
        
        # --- WEIGHTED SHOCK DISTRIBUTION ---
        val_map = {}
        if isinstance(shock, dict):
            # Manual precision (no weighting needed)
            val_map = {c: shock.get(c, 0.0) for c in unique_codes}
        else:
            # Q5 FIX: Fetch weights using each code's best_year (actual
            # data vintage), NOT the overwritten target_year.
            weights = {}
            total_basket_output = 0.0
            for c in unique_codes:
                weight_yr = code_weight_year.get(c, int(yr))
                w = get_actual_output(geo, weight_yr, [c])
                weights[c] = w
                total_basket_output += w
                
            for c in unique_codes:
                if total_basket_output > 0:
                    share = weights.get(c, 0.0) / total_basket_output
                    val_map[c] = shock * share
                else:
                    val_map[c] = shock / n
        
        # --- VECTORIZED COMPUTATION ---
        # Copy to avoid SettingWithCopyWarning
        group_by_year = group_by_year.copy()
        
        # Map shock values
        group_by_year["val_to_apply"] = group_by_year["join_code"].map(val_map)
        
        # Compute denominator
        is_jobs = group_by_year["variable"].astype(str).str.contains("Jobs", case=False, na=False)
        denom = np.where(is_jobs, 1_000_000.0, 1.0)
        
        # Calculate impact
        group_by_year["impact"] = (group_by_year["val_to_apply"] / denom) * group_by_year["value"]
        
        # Format output
        group_by_year["Year"] = int(yr)
        group_by_year["multiplier"] = group_by_year["value"]
        group_by_year["metric"] = group_by_year["variable"]
        
        if "industry_code" not in group_by_year.columns:
            group_by_year["industry_code"] = group_by_year["join_code"]
            
        needed_cols = ["Year", "industry_code", "industry_name", "metric", "multiplier", "impact"]
        
        # Cast categoricals to strings to avoid Pandas union_all bottlenecks during concat
        for col in ["industry_code", "industry_name", "metric", "multiplier"]:
            if col in group_by_year.columns and hasattr(group_by_year[col], "cat"):
                group_by_year[col] = group_by_year[col].astype(str)
                
        rows_dfs.append(group_by_year[needed_cols])

    if not rows_dfs: return pd.DataFrame(), pd.DataFrame()
    per_ind = pd.concat(rows_dfs, ignore_index=True)

    summ = per_ind.groupby(["metric", "Year"], as_index=False, observed=False)["impact"].sum()
    summ.rename(columns={"impact": "impact_dollars"}, inplace=True)
    return per_ind, summ

def compute_breakdown(shock: Union[float, dict], geo: str, year: int, join_codes: List[str], geo_scope: GeoScope = "all_provinces") -> pd.DataFrame:
    df = load_multipliers()
    if df.empty: return pd.DataFrame()

    target_types = ["Direct multiplier", "Simple multiplier", "Total multiplier"]
    
    mask = (
        (df["GEO"] == geo) &
        (df["multiplier_type"].isin(target_types)) &
        (df["join_code"].isin(join_codes))
    )
    
    if geo != "Canada" and "geo_scope" in df.columns:
        mask = mask & df["geo_scope"].isin([geo_scope, "national"])


    # 3. SMART ROW SELECTION (Per Industry)
    final_rows = []
    # N3 FIX: Port the code_weight_year pattern from compute_impacts().
    # Collect per-code best_year BEFORE overwriting YEAR, so weight
    # lookups use the actual data vintage, not a future target.
    code_weight_year: dict[str, int] = {}
    
    # We loop through requested codes and find best match for each
    broad_subset = df[mask].copy()
    if broad_subset.empty: return pd.DataFrame()

    for code in join_codes:
        ind_data = broad_subset[broad_subset["join_code"] == code]
        if ind_data.empty: continue
        
        avail_years = sorted(ind_data["YEAR"].unique())
        
        if year in avail_years:
            best_year = year
        elif avail_years:
            best_year = avail_years[-1]
        else:
            continue

        code_weight_year[code] = best_year  # N3: remember actual vintage
        selected = ind_data[ind_data["YEAR"] == best_year].copy()
        selected["YEAR"] = year  # Align to target year for UI
        final_rows.append(selected)
        
    if not final_rows: return pd.DataFrame()
    
    subset = pd.concat(final_rows)
    # ---------------------------

    unique_codes = subset["join_code"].unique()
    if len(unique_codes) == 0: return pd.DataFrame()
    
    # --- WEIGHTED SHOCK DISTRIBUTION ---
    if isinstance(shock, dict):
        is_precision = True
        weights = {}
        total_basket_output = 0.0
    else:
        is_precision = False
        weights = {}
        total_basket_output = 0.0
        for c in unique_codes:
            # N3 FIX: use vintage year for weight lookup
            weight_yr = code_weight_year.get(c, int(year))
            w = get_actual_output(geo, weight_yr, [c])
            weights[c] = w
            total_basket_output += w
    # --------------------------------------------
    
    rows_dfs = []
    
    # Pre-compute shock values per jcode
    val_map = {}
    if is_precision:
        val_map = {c: shock.get(c, 0.0) for c in unique_codes}
    else:
        for c in unique_codes:
            if total_basket_output > 0:
                share = weights.get(c, 0.0) / total_basket_output
                val_map[c] = shock * share
            else:
                val_map[c] = shock / len(unique_codes)

    # We need to compute Direct, Indirect, Induced for each join_code & variable.
    # Pivot to make multiplier types columns
    subset = subset.reset_index(drop=True)
    pivot_df = subset.pivot_table(
        index=["join_code", "variable"], 
        columns="multiplier_type", 
        values="value", 
        aggfunc='sum', 
        fill_value=0,
        observed=False,
    ).reset_index()
    
    for col in ["Direct multiplier", "Simple multiplier", "Total multiplier"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0.0

    pivot_df["Direct"] = pivot_df["Direct multiplier"]
    
    # N2 FIX: Guard against missing/suppressed Simple multiplier.
    pivot_df["Indirect"] = np.where(
        (pivot_df["Simple multiplier"] == 0) & (pivot_df["Total multiplier"] > 0),
        np.maximum(0, pivot_df["Total multiplier"] - pivot_df["Direct multiplier"]),
        np.maximum(0, pivot_df["Simple multiplier"] - pivot_df["Direct multiplier"])
    )
    
    pivot_df["Induced"] = np.where(
        (pivot_df["Simple multiplier"] == 0) & (pivot_df["Total multiplier"] > 0),
        0.0,
        np.maximum(0, pivot_df["Total multiplier"] - pivot_df["Simple multiplier"])
    )
    
    pivot_df["val_to_apply"] = pivot_df["join_code"].map(val_map)
    is_jobs = pivot_df["variable"].astype(str).str.contains("Jobs", case=False, na=False)
    pivot_df["denom"] = np.where(is_jobs, 1_000_000.0, 1.0)
    
    for t in ["Direct", "Indirect", "Induced"]:
        temp = pivot_df[["variable", t, "val_to_apply", "denom"]].copy()
        temp["Type"] = t
        temp["impact"] = (temp["val_to_apply"] / temp["denom"]) * temp[t]
        temp = temp.rename(columns={"variable": "metric"})
        rows_dfs.append(temp[["metric", "Type", "impact"]])
        
    final_df = pd.concat(rows_dfs, ignore_index=True)
    # Group by metric and Type to sum across all the underlying impacted industries
    return final_df.groupby(["metric", "Type"], as_index=False, observed=False)["impact"].sum()


def compute_hlrf_breakdown(
    shock: float,
    geo: str,
    year: int,
    join_codes: List[str],
    hlrf_overrides: dict,
    geo_scope: GeoScope = "all_provinces",
) -> pd.DataFrame:
    """Compute D/I/I breakdown with Analysis-by-Parts (ABP) & HLRF calibration.

    Implements the three-stage Analysis-by-Parts (ABP) calibration verified via
    Forensic Macroeconomic Audit (Oct 2026):
      1. Direct Value Added Specification: Replaces uncalibrated grain-dominated
         NAICS 111A Direct Jobs, Direct Wages, Direct GDP (basic & market prices),
         Direct GOS/GMI, Direct Employers' Social Contributions, and Direct Net
         Taxes on Production with OMAFRA COP empirical orchard anchors.
      2. Intermediate Supply-Chain (Indirect) Scaling: Scales Indirect impacts by
         the intermediate input compression ratio (e.g., $43.5M / $73.9M = 0.588633)
         so upstream supply-chain impacts reflect orchard intermediate purchases.
      3. Remittance-Adjusted Induced Scaling: Scales Induced impacts by the
         remittance-adjusted domestic induced scalar (e.g., 3.475x, accounting for
         the indirect wage denominator and 60% SAWP foreign remittances), or falls
         back to the linear wage ratio if no ABP config is present.
    """
    # 1. Get the uncalibrated breakdown
    bd = compute_breakdown(shock, geo, year, join_codes, geo_scope=geo_scope)
    if bd.empty:
        return bd

    bd = bd.copy()

    # Check if we can enrich hlrf_overrides with ABP parameters from oag_apple_data.yml
    abp_params = dict(hlrf_overrides)
    abp_cfg_keys = [
        "direct_gdp_basic_M",
        "direct_gdp_market_M",
        "direct_gos_M",
        "direct_taxes_prod_M",
        "direct_emp_social_M",
        "indirect_scale_factor",
        "induced_scale_factor",
        "indirect_output_M",
        "indirect_gdp_basic_M",
        "indirect_gdp_market_M",
        "indirect_wages_M",
        "indirect_emp_social_M",
        "indirect_jobs",
        "induced_output_M",
        "induced_gdp_basic_M",
        "induced_gdp_market_M",
        "induced_wages_M",
        "induced_emp_social_M",
        "induced_jobs",
    ]
    oag_cfg = load_oag_config().get("oag_employment_2024", {})
    for _, comm_cfg in oag_cfg.items():
        if isinstance(comm_cfg, dict) and comm_cfg.get("ftes") == abp_params.get("Jobs"):
            for k in abp_cfg_keys:
                if k in comm_cfg and k not in abp_params:
                    abp_params[k] = comm_cfg[k]
            break

    # 2. Identify the wage metric row for fallback linear ratio calculation
    wage_key = None
    for k in hlrf_overrides:
        if "wage" in k.lower() or "salaries" in k.lower():
            wage_key = k
            break

    wage_ratio = 1.0
    if wage_key is not None:
        wage_mask = bd["metric"].astype(str).str.contains("wages", case=False, na=False) | (
            bd["metric"].astype(str).str.contains("salaries", case=False, na=False)
        )
        direct_wage_rows = bd[wage_mask & (bd["Type"] == "Direct")]
        if not direct_wage_rows.empty:
            model_direct_wages = direct_wage_rows["impact"].sum()
            if model_direct_wages > 0:
                wage_ratio = hlrf_overrides[wage_key] / model_direct_wages

    # 3. Scale Indirect impacts if indirect_scale_factor is configured (ABP Stage 2)
    indirect_scale = abp_params.get("indirect_scale_factor")
    if indirect_scale is not None:
        ind_mask = bd["Type"] == "Indirect"
        bd.loc[ind_mask, "impact"] = bd.loc[ind_mask, "impact"] * float(indirect_scale)

    # 4. Scale Induced impacts by remittance-adjusted scalar (ABP Stage 3) or fallback wage_ratio
    induced_scale = abp_params.get("induced_scale_factor", wage_ratio)
    induced_mask = bd["Type"] == "Induced"
    bd.loc[induced_mask, "impact"] = bd.loc[induced_mask, "impact"] * float(induced_scale)

    # 5. Apply Direct overrides (ABP Stage 1)
    abp_meta_keys = set(abp_cfg_keys)
    for override_key, override_val in hlrf_overrides.items():
        if override_key in abp_meta_keys:
            continue
        key_lower = override_key.lower()
        if "job" in key_lower:
            match_mask = bd["metric"].astype(str).str.contains("Jobs", case=False, na=False)
        elif "wage" in key_lower or "salaries" in key_lower:
            match_mask = bd["metric"].astype(str).str.contains("wages", case=False, na=False) | (
                bd["metric"].astype(str).str.contains("salaries", case=False, na=False)
            )
        else:
            match_mask = bd["metric"].astype(str).str.contains(override_key, case=False, na=False)

        direct_mask = match_mask & (bd["Type"] == "Direct")
        if direct_mask.any():
            bd.loc[direct_mask, "impact"] = override_val

    # Apply Direct Value Added rebalancing when ABP parameters are present
    if "direct_gdp_basic_M" in abp_params:
        m = (bd["metric"] == "Gross domestic product (GDP) at basic prices") & (bd["Type"] == "Direct")
        bd.loc[m, "impact"] = float(abp_params["direct_gdp_basic_M"]) * 1e6
    if "direct_gdp_market_M" in abp_params:
        m = (bd["metric"] == "Gross domestic product (GDP) at market prices") & (bd["Type"] == "Direct")
        bd.loc[m, "impact"] = float(abp_params["direct_gdp_market_M"]) * 1e6
    if "direct_gos_M" in abp_params:
        m = (bd["metric"] == "Gross operating surplus") & (bd["Type"] == "Direct")
        bd.loc[m, "impact"] = float(abp_params["direct_gos_M"]) * 1e6
    if "direct_emp_social_M" in abp_params:
        m = (bd["metric"] == "Employers' social contributions") & (bd["Type"] == "Direct")
        bd.loc[m, "impact"] = float(abp_params["direct_emp_social_M"]) * 1e6
    if "direct_taxes_prod_M" in abp_params:
        m_tp = (bd["metric"] == "Taxes on production") & (bd["Type"] == "Direct")
        bd.loc[m_tp, "impact"] = float(abp_params["direct_taxes_prod_M"]) * 1e6
        # Zero-rated basic groceries at farm gate and net production tax treatment
        m_tprod = (bd["metric"] == "Taxes on products") & (bd["Type"] == "Direct")
        bd.loc[m_tprod, "impact"] = 0.0
        m_sprod = (bd["metric"] == "Subsidies on production") & (bd["Type"] == "Direct")
        bd.loc[m_sprod, "impact"] = 0.0

    # Apply explicit Indirect & Induced ABP vector targets when present
    indirect_abp_map = {
        "Output": ("indirect_output_M", 1e6),
        "Gross domestic product (GDP) at basic prices": ("indirect_gdp_basic_M", 1e6),
        "Gross domestic product (GDP) at market prices": ("indirect_gdp_market_M", 1e6),
        "Wages and salaries": ("indirect_wages_M", 1e6),
        "Employers' social contributions": ("indirect_emp_social_M", 1e6),
        "Jobs": ("indirect_jobs", 1.0),
    }
    for metric_name, (cfg_key, mult) in indirect_abp_map.items():
        if cfg_key in abp_params:
            m = (bd["metric"] == metric_name) & (bd["Type"] == "Indirect")
            bd.loc[m, "impact"] = float(abp_params[cfg_key]) * mult

    induced_abp_map = {
        "Output": ("induced_output_M", 1e6),
        "Gross domestic product (GDP) at basic prices": ("induced_gdp_basic_M", 1e6),
        "Gross domestic product (GDP) at market prices": ("induced_gdp_market_M", 1e6),
        "Wages and salaries": ("induced_wages_M", 1e6),
        "Employers' social contributions": ("induced_emp_social_M", 1e6),
        "Jobs": ("induced_jobs", 1.0),
    }
    for metric_name, (cfg_key, mult) in induced_abp_map.items():
        if cfg_key in abp_params:
            m = (bd["metric"] == metric_name) & (bd["Type"] == "Induced")
            bd.loc[m, "impact"] = float(abp_params[cfg_key]) * mult

    # Synchronize Labour income = Wages and salaries + Employers' social contributions when ABP vectors are used
    if "indirect_wages_M" in abp_params or "induced_wages_M" in abp_params or "direct_emp_social_M" in abp_params:
        for t in ["Direct", "Indirect", "Induced"]:
            w = bd.loc[(bd["metric"] == "Wages and salaries") & (bd["Type"] == t), "impact"].sum()
            e = bd.loc[(bd["metric"] == "Employers' social contributions") & (bd["Type"] == t), "impact"].sum()
            bd.loc[(bd["metric"] == "Labour income") & (bd["Type"] == t), "impact"] = w + e

    return bd


def compute_subsidy_impact(
    subsidy_dollars: float,
    geo: str,
    year: int,
    join_codes: List[str],
    geo_scope: GeoScope = "all_provinces",
) -> pd.DataFrame:
    """Compute Induced-only impact of government subsidy payments.

    Government transfers (Production Insurance, SDRM, AgriStability) do not
    generate supply-chain (Indirect) impacts — they are income transfers that
    farmers spend in their communities. This function isolates only the
    Induced (household spending) channel.
    """
    bd = compute_breakdown(subsidy_dollars, geo, year, join_codes, geo_scope=geo_scope)
    if bd.empty:
        return bd

    bd = bd.copy()

    # Zero out Direct and Indirect — subsidies only flow through Induced
    non_induced_mask = bd['Type'].isin(['Direct', 'Indirect'])
    bd.loc[non_induced_mask, 'impact'] = 0.0

    return bd


# ==============================================================================
# DERIVED FISCAL IMPACT MODULE (STAGE-SPECIFIC TAX INCIDENCE)
# ==============================================================================
# Methodology: Apply empirically calibrated, stage-specific effective tax rates
# to IO output vectors to estimate the full government revenue footprint.
#
# Scalars verified via Forensic Macroeconomic & Fiscal Audit (Oct 2026) against:
# - Income Tax Act s. 118.94 (SAWP 90% world income rule: BPA & spousal credits)
# - Ontario Taxation Act, 2007 s. 21.1 (Ontario LIFT credit residency rules)
# - Excise Tax Act s. 169 & Schedule VI Part III (100% ITC recovery on zero-rated
#   groceries; 13% HST on hard cider, <600 mL juice, agri-tourism & exempt inputs)
# - CCPC Small Business Deduction (12.2%) vs. General M&P/Farming CIT (25.0%)
# - Ontario Assessment Act s. 3(1) ¶19 (fruit tree exemption) & County FT Ratios
# ==============================================================================

# Stage-specific effective Personal Income Tax (PIT) rates on Wages & Salaries
_FISCAL_PIT_DIRECT_GROWING = 0.0452     # 4.52% blended primary orchard PIT
_FISCAL_PIT_DIRECT_PACKING = 0.1460     # 14.60% full-time wholesale (NAICS 411)
_FISCAL_PIT_DIRECT_PROCESSING = 0.1388  # 13.88% full-time food mfg (NAICS 3114)
_FISCAL_PIT_RATE = 0.1231               # 12.31% general economy / Indirect & Induced

# Stage-specific effective Corporate Income Tax (CIT) rates on Gross Operating Surplus
_FISCAL_CIT_DIRECT_GROWING = 0.0403     # 4.03% (0.33 taxable ratio × 12.2% CCPC rate)
_FISCAL_CIT_DIRECT_DOWNSTREAM = 0.0780  # 7.80% (corporate M&P / wholesale benchmark)
_FISCAL_CIT_GOS_RATE = 0.0549           # 5.49% (0.45 taxable ratio × 12.2% CCPC rate)

# Statutory Employer Payroll Tax rates (CPP + EI + EHT) on Direct Wages
_FISCAL_PAYROLL_DIRECT_GROWING = 0.0760     # 7.60% (CPP + EI + 0.35% small-employer EHT)
_FISCAL_PAYROLL_DIRECT_DOWNSTREAM = 0.0902  # 9.02% (CPP 5.57% + EI 1.70% + EHT 1.75%)

# Composite HST and Municipal Farm Property Tax scalars
_FISCAL_HST_RATE = 0.0195                   # 1.95% composite value-chain HST on Output
_FISCAL_FARM_PROP_TAX_PER_ACRE = 75.0       # $75.00/acre ($38.35 bare land + $36.65 buildings)


def compute_derived_fiscal_impact(
    io_results: pd.DataFrame,
    farm_acres: float = 0,
    segment_type: str = "default",
) -> pd.DataFrame:
    """Compute the full derived fiscal impact from IO model outputs.

    Supports both stage-specific breakdown DataFrames (containing 'Type' in
    ['Direct', 'Indirect', 'Induced'] and/or a 'segment' column in
    ['growing', 'packing', 'processing', 'transfers']) and flat IO output
    DataFrames from compute_impacts().
    """
    if io_results is None or io_results.empty:
        return pd.DataFrame()

    df = io_results.copy()
    has_type = "Type" in df.columns
    has_segment = "segment" in df.columns

    pit_Rates = {
        "growing": _FISCAL_PIT_DIRECT_GROWING,
        "packing": _FISCAL_PIT_DIRECT_PACKING,
        "processing": _FISCAL_PIT_DIRECT_PROCESSING,
        "transfers": _FISCAL_PIT_RATE,
        "default": _FISCAL_PIT_RATE,
    }
    cit_Rates = {
        "growing": _FISCAL_CIT_DIRECT_GROWING,
        "packing": _FISCAL_CIT_DIRECT_DOWNSTREAM,
        "processing": _FISCAL_CIT_DIRECT_DOWNSTREAM,
        "transfers": _FISCAL_CIT_GOS_RATE,
        "default": _FISCAL_CIT_GOS_RATE,
    }
    pay_Rates = {
        "growing": _FISCAL_PAYROLL_DIRECT_GROWING,
        "packing": _FISCAL_PAYROLL_DIRECT_DOWNSTREAM,
        "processing": _FISCAL_PAYROLL_DIRECT_DOWNSTREAM,
    }

    def _sum_metric(sub_df: pd.DataFrame, metric_name: str, impact_types: Optional[List[str]] = None) -> float:
        m = sub_df["metric"] == metric_name
        if impact_types is not None and "Type" in sub_df.columns:
            m = m & sub_df["Type"].isin(impact_types)
        return float(sub_df.loc[m, "impact"].sum()) if m.any() else 0.0

    # Determine segment partitions
    if has_segment:
        segments = [(str(seg).lower(), grp) for seg, grp in df.groupby("segment", observed=False)]
    else:
        segments = [(segment_type.lower(), df)]

    total_pit = 0.0
    total_cit = 0.0
    total_hst = 0.0
    total_payroll = 0.0
    total_io_prod = 0.0
    prop_tax_dollars = float(farm_acres) * _FISCAL_FARM_PROP_TAX_PER_ACRE if farm_acres > 0 else 0.0

    for seg_name, seg_df in segments:
        p_rate = pit_Rates.get(seg_name, _FISCAL_PIT_RATE)
        c_rate = cit_Rates.get(seg_name, _FISCAL_CIT_GOS_RATE)

        if has_type:
            w_dir = _sum_metric(seg_df, "Wages and salaries", ["Direct"])
            w_ind_inc = _sum_metric(seg_df, "Wages and salaries", ["Indirect", "Induced"])
            total_pit += w_dir * p_rate + w_ind_inc * _FISCAL_PIT_RATE

            gos_dir = _sum_metric(seg_df, "Gross operating surplus", ["Direct"])
            gos_ind_inc = _sum_metric(seg_df, "Gross operating surplus", ["Indirect", "Induced"])
            total_cit += gos_dir * c_rate + gos_ind_inc * _FISCAL_CIT_GOS_RATE

            if seg_name in pay_Rates:
                total_payroll += (
                    w_dir * pay_Rates[seg_name]
                    + _sum_metric(seg_df, "Employers' social contributions", ["Indirect", "Induced"])
                )
            else:
                total_payroll += _sum_metric(seg_df, "Employers' social contributions")

            # Exclude Induced 'Taxes on products' from IO Production Taxes when applying
            # the 1.95% composite HST scalar across total Output, preventing double-counting
            # of household retail sales taxes.
            t_prod = _sum_metric(seg_df, "Taxes on production")
            t_products_di = _sum_metric(seg_df, "Taxes on products", ["Direct", "Indirect"])
            total_io_prod += t_prod + t_products_di
        else:
            w_all = _sum_metric(seg_df, "Wages and salaries")
            gos_all = _sum_metric(seg_df, "Gross operating surplus")
            total_pit += w_all * p_rate
            total_cit += gos_all * c_rate
            total_payroll += _sum_metric(seg_df, "Employers' social contributions")
            total_io_prod += _sum_metric(seg_df, "Taxes on production") + _sum_metric(seg_df, "Taxes on products")

        total_hst += _sum_metric(seg_df, "Output") * _FISCAL_HST_RATE

    # Deduct acreage-based farm property tax from Direct Taxes on Production when
    # Growing is evaluated via ABP (where $1.50M Direct T_net already includes the
    # $1.19M municipal orchard property tax), preventing double-counting.
    if prop_tax_dollars > 0 and has_type and any(s == "growing" for s, _ in segments):
        total_io_prod = max(0.0, total_io_prod - prop_tax_dollars)

    total_wages = _sum_metric(df, "Wages and salaries")
    total_gos = _sum_metric(df, "Gross operating surplus")
    eff_pit_scalar = (total_pit / total_wages) if total_wages > 0 else _FISCAL_PIT_RATE
    eff_cit_scalar = (total_cit / total_gos) if total_gos > 0 else _FISCAL_CIT_GOS_RATE

    rows = [
        {
            "tax_channel": "Personal Income Tax",
            "amount": total_pit,
            "method": "Stage-specific PIT (4.52% Orchard / 13.88%–14.60% Downstream / 12.31% Ind & Inc)",
            "confidence": "High",
            "io_vector_used": "Wages and salaries",
            "scalar_applied": eff_pit_scalar,
        },
        {
            "tax_channel": "Corporate Income Tax",
            "amount": total_cit,
            "method": "Stage-specific CIT (4.03% Orchard / 7.80% Downstream / 5.49% Ind & Inc)",
            "confidence": "High",
            "io_vector_used": "Gross operating surplus",
            "scalar_applied": eff_cit_scalar,
        },
        {
            "tax_channel": "HST (net of zero-rating & ITC)",
            "amount": total_hst,
            "method": "1.95% composite rate (13% on cider, <600mL juice, agri-tourism & exempt inputs)",
            "confidence": "High",
            "io_vector_used": "Output",
            "scalar_applied": _FISCAL_HST_RATE,
        },
        {
            "tax_channel": "Employer Payroll Taxes",
            "amount": total_payroll,
            "method": "Statutory CPP/EI/EHT on Direct (7.60% Orchard / 9.02% Downstream) + IO Ind/Inc",
            "confidence": "Very High",
            "io_vector_used": "Wages and salaries / Employers' social contributions",
            "scalar_applied": 0.0,
        },
        {
            "tax_channel": "IO Production Taxes",
            "amount": total_io_prod,
            "method": "IO Taxes on production & D/I products (net of farm property tax)",
            "confidence": "Very High",
            "io_vector_used": "Taxes on production + Taxes on products (Direct/Indirect)",
            "scalar_applied": 0.0,
        },
    ]

    if farm_acres > 0:
        rows.append({
            "tax_channel": "Municipal Property Tax (Farm)",
            "amount": prop_tax_dollars,
            "method": f"{farm_acres:,.0f} acres × ${_FISCAL_FARM_PROP_TAX_PER_ACRE:.0f}/acre ($38.35 bare land + $36.65 buildings)",
            "confidence": "Very High",
            "io_vector_used": "N/A (MPAC FPCTRP acreage-based)",
            "scalar_applied": _FISCAL_FARM_PROP_TAX_PER_ACRE,
        })

    fiscal_df = pd.DataFrame(rows)
    return fiscal_df


# ==============================================================================
# COUNTY-LEVEL IMPACT ESTIMATION (Ontario Only)
# ==============================================================================
# Methodology: Apportion provincial IO impacts by county Farm Cash Receipt (FCR)
# share. FCR is used as a proxy for Output at the sub-provincial level.
# Source: OMAFRA Ontario Farm Cash Receipts by County & Commodity
# ==============================================================================

_COUNTY_FCR_FP = BASE_DIR / "data" / "latest" / "omafra_county_fcr.csv"

@lru_cache(maxsize=1)
def load_county_fcr() -> pd.DataFrame:
    """Load processed OMAFRA county-level farm cash receipt data."""
    if not _COUNTY_FCR_FP.exists():
        return pd.DataFrame()
    df = smart_read(_COUNTY_FCR_FP)
    return df


def available_county_years() -> list[int]:
    """Return years with county FCR data."""
    df = load_county_fcr()
    if df.empty:
        return []
    return sorted(df["year"].unique().astype(int).tolist())


def available_counties(year: Optional[int] = None) -> list[dict]:
    """Return list of counties with display names for a given year.
    
    Returns list of dicts: [{"slug": "huron", "display": "Huron County", "region": "Western Ontario"}, ...]
    """
    df = load_county_fcr()
    if df.empty:
        return []
    if year:
        df = df[df["year"] == year]
    totals = df[df["commodity"] == "Total"].drop_duplicates(subset=["county"])
    return [
        {"slug": r["county"], "display": r["county_display"], "region": r["region"]}
        for _, r in totals.sort_values("county_display").iterrows()
    ]


def compute_county_impacts(
    year: int,
    scope: Scope,
    geo_scope: GeoScope = "all_provinces",
) -> pd.DataFrame:
    """Estimate county-level economic impacts for Ontario Primary Agriculture.
    
    Methodology:
    1. Compute provincial Ontario impacts for primary_agriculture basket
    2. Load county FCR shares for the given year
    3. Apportion each provincial metric by county share
    
    Returns DataFrame with columns:
        county, county_display, region, year, metric,
        provincial_impact, county_share, county_impact
    """
    # Step 1: Get the provincial primary agriculture impacts
    baskets = load_baskets_config()
    prim_ag = baskets.get("primary_agriculture", {})
    if not prim_ag:
        return pd.DataFrame()
    
    # Get primary agriculture codes
    basket_df = industries_for_basket("primary_agriculture")
    if basket_df.empty:
        return pd.DataFrame()
    codes = basket_df["join_code"].tolist()
    
    # Build per-code shock dict using actual historical output
    shock_dict = {}
    for c in codes:
        shock_dict[c] = get_actual_output("Ontario", year, [c])
    
    # Compute provincial impacts
    _, prov_sum = compute_impacts(shock_dict, "Ontario", year, scope, codes, geo_scope=geo_scope)
    if prov_sum is None or prov_sum.empty:
        return pd.DataFrame()
    
    # Step 2: Load county FCR shares
    fcr = load_county_fcr()
    if fcr.empty:
        return pd.DataFrame()
    
    county_totals = fcr[(fcr["year"] == year) & (fcr["commodity"] == "Total")]
    if county_totals.empty:
        # Try nearest year
        avail = sorted(fcr["year"].unique())
        if not avail:
            return pd.DataFrame()
        nearest = min(avail, key=lambda y: abs(y - year))
        county_totals = fcr[(fcr["year"] == nearest) & (fcr["commodity"] == "Total")]
    
    # Step 3: Apportion
    rows = []
    for _, county_row in county_totals.iterrows():
        county_share = county_row["share_of_province"]
        for _, metric_row in prov_sum.iterrows():
            rows.append({
                "county": county_row["county"],
                "county_display": county_row["county_display"],
                "region": county_row["region"],
                "year": year,
                "metric": metric_row["metric"],
                "provincial_impact": metric_row["impact_dollars"],
                "county_share": county_share,
                "county_impact": metric_row["impact_dollars"] * county_share,
                "fcr_millions": county_row["value_millions"],
            })
    
    return pd.DataFrame(rows)


def get_county_commodity_breakdown(county_slug: str, year: int) -> pd.DataFrame:
    """Get commodity-level FCR breakdown for a specific county.
    
    Returns DataFrame with columns: commodity, value_millions, share_of_county_total
    """
    fcr = load_county_fcr()
    if fcr.empty:
        return pd.DataFrame()
    
    county_data = fcr[(fcr["county"] == county_slug) & (fcr["year"] == year)]
    if county_data.empty:
        return pd.DataFrame()
    
    # Get total and individual commodities
    total_row = county_data[county_data["commodity"] == "Total"]
    items = county_data[county_data["commodity"] != "Total"].copy()
    
    if total_row.empty or items.empty:
        return pd.DataFrame()
    
    total_val = total_row["value_millions"].values[0]
    items = items[items["value_millions"] > 0].copy()
    items["share_of_county_total"] = items["value_millions"] / total_val if total_val > 0 else 0
    items = items[["commodity", "value_millions", "share_of_county_total"]].sort_values(
        "value_millions", ascending=False
    )
    
    return items
