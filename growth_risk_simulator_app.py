"""
OFA Growth & Risk Simulator — Standalone Decision Support Application
=====================================================================
Root-level Streamlit entry point (`growth_risk_simulator_app.py`) integrating:
1. Unified Macro-Financial & Dynamic Stock-Flow Engine (`scripts/growth_risk_engine.py`)
2. Provincial & Sub-Sector IO Multiplier, Fiscal Tax & BRM Subsidy Engine (`scripts/io_multipliers_engine.py`)
3. Analysis-by-Parts (ABP) Downstream Food & Beverage Processing Engine (StatCan SUT 36-10-0478-01)
4. 2026 U.S. Section 338 Tariff & Canada Counter-Tariff "Double Squeeze" Engine (`data/trade_app/trade_exposure.db`)
5. StatCan 32-10-0136-01 "Average Per Farm" Translator + 26 OMAFA Pub 60 / OAG / OPVG / OGVG / NAICS 1114 Budgets
6. StatCan FIPI (18-10-0258-01) Historical Volatility & Live Commodity Futures Engine (`scripts/input_cost_engine.py`)
7. Short-Term NCI Accounting Bridge (`scripts/forecasting_engine.py`) & Executive Report Generators (`.docx` / `.pptx`)

Strictly classifies Nursery, Floriculture, Sod & Greenhouse operators (NAICS 1114 / HS Ch. 06 & 07)
as Primary Agriculture across all farm-gate totals, tariff simulations, and IO multiplier calculations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE if (_HERE / "scripts").exists() else _HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.smart_read import smart_read
from scripts.calibrated_parameters import (
    ABP_PROCESSOR_TABLE,
    ARMA_MONTE_CARLO_PERSISTENCE,
    BRM_EFFECTIVE_PAYOUT_PARAMETERS,
    DEBT_AMORTIZATION_AND_DISTRESS_PARAMS,
    DEBT_REPRICING_SCHEDULE,
    EMPIRICAL_ICOR_PROFILES,
    FARM_HOUSEHOLD_MPC_MULTIPLIERS,
    HOTELLING_INPUT_SUPPLY_ELASTICITIES,
    JORGENSON_CAPEX_ELASTICITIES,
    MUTUALLY_EXCLUSIVE_LEAF_NAICS,
    PARENT_AGGREGATE_NAICS,
    PRIMARY_AG_DISPOSITION_MATRIX,
    PRIMARY_AG_TARIFF_CONFIG,
    PROVINCIAL_TFP_DECOMPOSITION,
    TARGETED_SECTION338_2026_PACKAGE,
    TARGETED_SECTION338_OPTION_LABEL,
    get_constant_2022_price_deflator,
    get_debt_amortization_params,
    get_debt_repricing_schedule,
    get_economic_depreciation_rate,
    get_hotelling_sector_profile,
    get_mc_covariance_params,
    get_targeted_section338_package_params,
)
from scripts.farm_cost_impact_model import ENTERPRISE_BUDGETS, simulate_enterprise_budget
from scripts.growth_risk_engine import (
    _fmt_ofa_num,
    compute_basket_impact,
    compute_cost_commodity_breakdown,
    compute_downstream_abp_impact,
    compute_effective_tariff_price_shock,
    compute_io_scope_retention_table,
    compute_provincial_fcr_weights,
    compute_scenario_comparison_summary,
    compute_simulator_fiscal_summary,
    compute_simulator_hlrf_summary,
    compute_simulator_subsidy_summary,
    compute_statcan_average_farm_impact,
    generate_simulator_pptx_deck,
    generate_simulator_word_briefing,
    get_commodity_tariff_params,
    get_enterprise_tariff_commodity,
    get_scoped_household_mpc_multipliers,
    load_fipi_input_cost_kpis,
    load_section338_and_counter_tariff_data,
    load_subsector_io_multipliers,
    run_dynamic_projection,
    run_monte_carlo_projections,
    run_short_term_nci_bridge,
)
from scripts.io_multipliers_engine import load_baskets_config

# ------------------------------------------------------------------------------
# OFFICIAL OFA BRAND IDENTITY PALETTE & PRESENTATION CHART THEME (2025-26 GUIDE)
# ------------------------------------------------------------------------------
OFA_COLORS: Dict[str, str] = {
    "crimson": "#D7182A",       # Pantone 1797C — Primary shock / alert / brand red
    "valencia": "#DC4949",      # Pantone Red 032C — Secondary warm red
    "bluewood": "#334B58",      # Pickled Bluewood — Primary structural slate-navy
    "tulip": "#ECA842",         # Tulip Tree — Warm amber / gold accent
    "champagne": "#EDE6DE",     # Champagne — Neutral card / bar fill
    "slate_base": "#64748B",    # Unshocked reference baseline
    "positive": "#2E6F40",      # Muted institutional forest for positive gains
    "grid": "#E2E8F0",          # Subtle horizontal gridline
    "text": "#1E293B",          # High-contrast body/tick text
    "subtext": "#475569",       # Subtitle / source note text
}

SHORT_NAICS_LABELS: Dict[str, str] = {
    "All farm types": "All Primary Farms",
    "Oilseed and grain farming [1111]": "Oilseed & Grain [1111]",
    "Potato farming [111211]": "Potato Farming [111211]",
    "Other vegetable (except potato) and melon farming [111219]": "Field Vegetables [111219]",
    "Fruit and tree nut farming [1113]": "Fruit & Tree Nut [1113]",
    "Greenhouse, nursery and floriculture production [1114]": "Greenhouse, Nursery & Floriculture [1114]",
    "Other crop farming [1119]": "Other Crop Farming [1119]",
    "Beef cattle ranching and farming, including feedlots [112110]": "Beef Cattle & Feedlots [112110]",
    "Beef cattle ranching and farming, including feedlots [112111]": "Beef Cattle & Feedlots [112111]",
    "Dairy cattle and milk production [112120]": "Dairy & Milk Production [112120]",
    "Dairy cattle and milk production [11212]": "Dairy & Milk Production [11212]",
    "Hog and pig farming [1122]": "Hog & Pig Farming [1122]",
    "Poultry and egg production [1123]": "Poultry & Egg [1123]",
    "Sheep and goat farming [1124]": "Sheep & Goat [1124]",
    "Other animal production [1129]": "Other Animal Production [1129]",
    "Other animal production [112A]": "Other Animal Production [112A]",
    "Crop production [111]": "Crop Production [111]",
    "Animal production [112]": "Animal Production [112]",
}


def shorten_naics_label(raw_label: str) -> str:
    return SHORT_NAICS_LABELS.get(str(raw_label), str(raw_label))


def ofa_plotly_config(filename_slug: str) -> Dict[str, Any]:
    """Configure 300-DPI presentation-ready PNG camera export (1600x900 @ scale=3 = 4800x2700 px)."""
    clean_slug = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in filename_slug).strip("_")
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": clean_slug or "OFA_Simulator_Chart",
            "width": 1600,
            "height": 900,
            "scale": 3,
        },
    }


def apply_ofa_chart_theme(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    source_note: str = "Source: Statistics Canada Input-Output & Farm Financial Accounts; OFA Modeling.",
    height: int = 430,
    legend_y: float = 1.03,
    top_margin: int = 112,
    bottom_margin: int = 78,
    left_margin: int = 24,
    right_margin: int = 48,
    is_horizontal_bar: bool = False,
) -> go.Figure:
    """
    Apply consistent OFA Brand Identity (2025-26) styling to any Plotly chart:
      - Cross-platform Arial/Roboto sans-serif typography (never clips or switches fonts)
      - Top-left bold Pickled Bluewood headline + crisp slate subtitle
      - Horizontal legend docked ABOVE the plot frame (eliminates X-axis collisions)
      - Fixed-pixel bottom-left institutional source footer (`y=0, yshift=-54`)
    """
    title_html = f"<b>{title}</b>"
    if subtitle:
        title_html += f"<br><span style='font-size:12.5px;color:#475569;font-weight:normal'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["text"]),
        title=dict(
            text=title_html,
            x=0.0,
            xanchor="left",
            xref="container",
            y=0.96,
            yanchor="top",
            yref="container",
            pad=dict(l=12, r=12),
            font=dict(family="Arial, Roboto, sans-serif", size=15.5, color=OFA_COLORS["bluewood"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["text"]),
            title_text="",
        ),
        margin=dict(t=top_margin, b=bottom_margin, l=left_margin, r=right_margin),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=OFA_COLORS["bluewood"],
            font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["text"]),
        ),
    )

    fig.update_xaxes(
        automargin=True,
        showgrid=is_horizontal_bar,
        gridcolor=OFA_COLORS["grid"],
        zeroline=is_horizontal_bar,
        zerolinecolor="#94A3B8",
        zerolinewidth=1.3,
        linecolor="#CBD5E1",
        tickfont=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["text"]),
        title_font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["bluewood"]),
    )
    fig.update_yaxes(
        automargin=True,
        showgrid=(not is_horizontal_bar),
        gridcolor=OFA_COLORS["grid"],
        zeroline=(not is_horizontal_bar),
        zerolinecolor="#CBD5E1",
        linecolor="#CBD5E1",
        tickfont=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["text"]),
        title_font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["bluewood"]),
    )

    if source_note:
        fig.add_annotation(
            text=source_note,
            xref="paper",
            yref="paper",
            x=0.0,
            y=0.0,
            yshift=-54,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(family="Arial, Roboto, sans-serif", size=11, color=OFA_COLORS["subtext"]),
        )
    return fig


# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & OFFICIAL OFA INSTITUTIONAL STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="OFA Growth & Risk Simulator | Macro-to-Farm-Gate Decision System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

if _HERE.name == "pages":
    try:
        from app.utils import inject_standalone_mode, show_data_freshness
        inject_standalone_mode()
        show_data_freshness()
    except Exception:
        pass

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Roboto:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Montserrat', Arial, sans-serif !important;
        color: #334B58;
    }
    .main-header {
        background: linear-gradient(135deg, #243640 0%, #334B58 72%, #3E5A69 100%);
        color: #ffffff;
        padding: 1.45rem 1.85rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 7px solid #D7182A;
        box-shadow: 0 4px 14px rgba(51,75,88,0.16);
    }
    .main-header .ofa-kicker {
        font-family: 'Montserrat', Arial, sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ECA842;
        margin-bottom: 0.25rem;
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Montserrat', Arial, sans-serif !important;
        font-size: 1.82rem;
        font-weight: 800;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 0.42rem 0 0 0;
        font-family: 'Roboto', Arial, sans-serif;
        font-size: 0.95rem;
        color: #EDE6DE;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #334B58;
        border-radius: 10px;
        padding: 0.92rem 1.05rem;
        box-shadow: 0 2px 6px rgba(51,75,88,0.06);
        height: 100%;
    }
    .kpi-card-alert {
        border-top: 4px solid #D7182A;
    }
    .kpi-label {
        font-family: 'Montserrat', Arial, sans-serif;
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #334B58;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-family: 'Montserrat', Arial, sans-serif;
        font-size: 1.42rem;
        font-weight: 800;
        color: #1E293B;
    }
    .kpi-sub {
        font-family: 'Roboto', Arial, sans-serif;
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.22rem;
        line-height: 1.35;
    }
    .badge-primary-ag {
        display: inline-block;
        background: #EDE6DE;
        color: #334B58;
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 700;
        font-size: 0.73rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #ECA842;
    }
    .badge-io-scope {
        display: inline-block;
        background: rgba(236, 168, 66, 0.18);
        color: #FFFFFF;
        font-family: 'Montserrat', Arial, sans-serif;
        font-weight: 700;
        font-size: 0.73rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #ECA842;
    }
    .ofa-footer-bar {
        margin-top: 2.2rem;
        padding-top: 0.85rem;
        font-family: 'Roboto', Arial, sans-serif;
        font-size: 0.86rem;
        color: #334B58;
    }
    .ofa-keyline {
        display: flex;
        width: 100%;
        height: 2px;
        margin-bottom: 0.65rem;
    }
    .ofa-keyline-red {
        width: 65%;
        background: #D7182A;
    }
    .ofa-keyline-slate {
        width: 35%;
        background: #334B58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# CACHED DATA LOADERS
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_core_datasets() -> Dict[str, Any]:
    io_path = PROJECT_ROOT / "data" / "latest" / "io_multipliers_standardized.parquet"
    try:
        io_df = pd.read_parquet(io_path, filters=[("YEAR", ">=", 2021)])
        if io_df.empty:
            io_df = pd.read_parquet(io_path)
    except Exception:
        io_df = pd.read_parquet(io_path)
    baskets_cfg = load_baskets_config()

    with open(PROJECT_ROOT / "data" / "derived" / "farm_expense_profiles.json", "r", encoding="utf-8") as f:
        expense_profiles = json.load(f)

    df_capex = pd.read_csv(PROJECT_ROOT / "data" / "derived" / "basket_capex.csv")
    df_health = pd.read_csv(PROJECT_ROOT / "data" / "derived" / "farm_financial_health.csv")
    df_fcr = smart_read(PROJECT_ROOT / "data" / "latest" / "32-10-0045-01.csv")
    df_136 = smart_read(PROJECT_ROOT / "data" / "latest" / "32-10-0136-01_dashboard.csv")

    mkt_path = PROJECT_ROOT / "data" / "latest" / "market_signals.csv"
    df_mkt = (
        smart_read(mkt_path)
        if (mkt_path.exists() or mkt_path.with_suffix(".parquet").exists())
        else pd.DataFrame()
    )

    return {
        "io_df": io_df,
        "baskets_cfg": baskets_cfg,
        "expense_profiles": expense_profiles,
        "df_capex": df_capex,
        "df_health": df_health,
        "df_fcr": df_fcr,
        "df_136": df_136,
        "df_mkt": df_mkt,
    }


data_bundle = load_core_datasets()
io_df = data_bundle["io_df"]
baskets_cfg = data_bundle["baskets_cfg"]
expense_profiles = data_bundle["expense_profiles"]
df_capex = data_bundle["df_capex"]
df_health = data_bundle["df_health"]
df_fcr = data_bundle["df_fcr"]
df_136 = data_bundle["df_136"]
df_mkt = data_bundle["df_mkt"]


@st.cache_data(show_spinner=False)
def cached_fcr_weights(region_name: str) -> Dict[str, Any]:
    return compute_provincial_fcr_weights(df_fcr, region=region_name)


@st.cache_data(show_spinner=False)
def cached_hlrf_summary(
    region_name: str, target_yr: int, geo_scope_val: str = "all_provinces"
) -> List[Dict[str, Any]]:
    return compute_simulator_hlrf_summary(
        region=region_name, year=target_yr, io_df=io_df, geo_scope=geo_scope_val
    )


@st.cache_data(show_spinner=False)
def cached_io_retention_table(region_name: str, target_yr: int) -> List[Dict[str, Any]]:
    return compute_io_scope_retention_table(io_df=io_df, region=region_name, year=target_yr)


@st.cache_data(show_spinner=False)
def cached_monte_carlo(
    n_sims_val: int,
    horizon_val: int,
    base_output_val: float,
    base_gdp_val: float,
    base_cap_stock_val: float,
    base_capex_val: float,
    shock_capex_pct_val: float,
    tfp_growth_pct_val: float,
    depreciation_rate_pct_val: float,
    cost_shock_val: float,
    rev_shock_val: float,
    region_val: str,
    target_year_val: int,
    base_debt_val: float,
    base_interest_val: float,
    base_nci_val: float,
    shock_interest_pct_val: float,
    tariff_params_val: Optional[Dict[str, Any]],
    mc_basis_val: str,
    ramp_schedule_val: Optional[Tuple[float, ...]],
    primary_join_code_val: Optional[str],
    farm_type_val: str,
    input_price_shocks_val: Dict[str, float],
    intermediate_cost_shock_val: float,
    labor_cost_shock_val: float,
    shock_ir_bps_val: float,
    amortization_rate_val: float,
    icor_val: float,
    brm_subsidy_val: float,
    auto_brm_val: bool,
    brm_regime_val: str,
    geo_scope_val: str = "all_provinces",
) -> Dict[int, Dict[str, float]]:
    return run_monte_carlo_projections(
        n_sims=n_sims_val,
        horizon=horizon_val,
        base_output=base_output_val,
        base_gdp=base_gdp_val,
        base_cap_stock=base_cap_stock_val,
        base_capex=base_capex_val,
        shock_capex_pct=shock_capex_pct_val,
        tfp_growth_pct=tfp_growth_pct_val,
        depreciation_rate_pct=depreciation_rate_pct_val,
        cost_shock_dollars=cost_shock_val,
        rev_shock_dollars=rev_shock_val,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region=region_val,
        target_year=target_year_val,
        base_debt=base_debt_val,
        base_interest=base_interest_val,
        base_nci=base_nci_val,
        shock_interest_pct=shock_interest_pct_val,
        tariff_params=tariff_params_val,
        mc_basis=mc_basis_val,
        capex_gestation_ramp=list(ramp_schedule_val) if ramp_schedule_val is not None else None,
        primary_join_code=primary_join_code_val,
        farm_type=farm_type_val,
        input_price_shocks=input_price_shocks_val,
        intermediate_cost_shock_dollars=intermediate_cost_shock_val,
        labor_cost_shock_dollars=labor_cost_shock_val,
        shock_ir_bps=shock_ir_bps_val,
        amortization_rate=amortization_rate_val,
        icor_value=icor_val,
        brm_subsidy_dollars=brm_subsidy_val,
        auto_brm_stabilization=auto_brm_val,
        brm_regime=brm_regime_val,
        geo_scope=geo_scope_val,
    )


@st.cache_data(show_spinner=False)
def cached_word_briefing_bytes(
    scenario_sig: Tuple[Any, ...],
    _df_dyn: pd.DataFrame,
    _df_base_dyn: pd.DataFrame,
    _mc_summary: Dict[int, Dict[str, float]],
    _abp_res: Dict[str, Any],
    _avg_farm_res: Dict[str, Any],
    _ent_sim: Dict[str, Any],
) -> Optional[bytes]:
    (
        reg, f_type, p_name, base_out, base_gdp,
        comb_out_m, comb_gdp_m, comb_jobs, t_pct, t_comm,
        _ent_key, _ent_ac, _inc_rent, _pt_cust,
    ) = scenario_sig[:14]
    geo_scope_sig = str(scenario_sig[-1]) if scenario_sig[-1] in ("all_provinces", "within_province") else "all_provinces"
    buf = generate_simulator_word_briefing(
        region=reg,
        farm_type=f_type,
        preset_name=p_name,
        df_dyn=_df_dyn,
        mc_summary=_mc_summary,
        abp_res=_abp_res,
        avg_farm_res=_avg_farm_res,
        ent_sim=_ent_sim,
        base_output_dollars=base_out,
        base_gdp_dollars=base_gdp,
        combined_yr1_output_impact_m=comb_out_m,
        combined_yr1_gdp_impact_m=comb_gdp_m,
        combined_yr1_jobs_impact=comb_jobs,
        tariff_pct=t_pct,
        tariff_commodity=t_comm,
        df_base_dyn=_df_base_dyn,
        geo_scope=geo_scope_sig,
    )
    return buf.getvalue() if buf is not None else None


@st.cache_data(show_spinner=False)
def cached_pptx_deck_bytes(
    scenario_sig: Tuple[Any, ...],
    _df_dyn: pd.DataFrame,
    _df_base_dyn: pd.DataFrame,
    _abp_res: Dict[str, Any],
    _avg_farm_res: Dict[str, Any],
    _ent_sim: Dict[str, Any],
) -> Optional[bytes]:
    (
        reg, f_type, p_name, base_out, base_gdp,
        comb_out_m, comb_gdp_m, comb_jobs, _t_pct, _t_comm,
        _ent_key, _ent_ac, _inc_rent, _pt_cust,
    ) = scenario_sig[:14]
    geo_scope_sig = str(scenario_sig[-1]) if scenario_sig[-1] in ("all_provinces", "within_province") else "all_provinces"
    buf = generate_simulator_pptx_deck(
        region=reg,
        farm_type=f_type,
        preset_name=p_name,
        df_dyn=_df_dyn,
        abp_res=_abp_res,
        avg_farm_res=_avg_farm_res,
        ent_sim=_ent_sim,
        base_output_dollars=base_out,
        base_gdp_dollars=base_gdp,
        combined_yr1_output_impact_m=comb_out_m,
        combined_yr1_gdp_impact_m=comb_gdp_m,
        combined_yr1_jobs_impact=comb_jobs,
        df_base_dyn=_df_base_dyn,
        geo_scope=geo_scope_sig,
    )
    return buf.getvalue() if buf is not None else None


@st.cache_data(show_spinner=False)
def cached_section338_data(region_name: str) -> Dict[str, Any]:
    return load_section338_and_counter_tariff_data(region=region_name)


@st.cache_data(show_spinner=False)
def cached_fipi_kpis(region_name: str) -> List[Dict[str, Any]]:
    return load_fipi_input_cost_kpis(region=region_name)


@st.cache_data(show_spinner=False)
def cached_nci_bridge(
    region_name: str,
    grain_price_shock_pct: float,
    livestock_price_shock_pct: float,
    fuel_shock_pct: float,
    fert_shock_pct: float,
    wage_shock_pct: float,
    general_inflation_pct: float,
    interest_shock_bps: float,
) -> Dict[str, Any]:
    return run_short_term_nci_bridge(
        region=region_name,
        grain_price_shock_pct=grain_price_shock_pct,
        livestock_price_shock_pct=livestock_price_shock_pct,
        fuel_shock_pct=fuel_shock_pct,
        fert_shock_pct=fert_shock_pct,
        wage_shock_pct=wage_shock_pct,
        general_inflation_pct=general_inflation_pct,
        interest_shock_bps=interest_shock_bps,
    )


# ------------------------------------------------------------------------------
# SCENARIO PRESETS (100% Grounded in StatCan FIPI, OMAFA Pub 60, & Sept 8 Counter-Tariffs)
# ------------------------------------------------------------------------------
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom Scenario (Manual Sliders)": {
        "desc": "Configure custom input cost, Section 338 tariff, CapEx, TFP, and BRM policy shocks.",
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": TARGETED_SECTION338_OPTION_LABEL, "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2026 Section 338 Actual Package (+50% Targeted Annexes + Sept 29 Alcohol/Whey Ban + Sept 8 Counter-Tariffs)": {
        "desc": (
            "Empirically calibrated PE-ABP-IO simulation of the actual 2026 U.S. Section 338 proclamations "
            "(Procl. 11046–11048 +50% duty on 18 HS Ch. 06 cut-flower/bulb/cutting subheadings [$175.14M ON; soil-grown "
            "nursery stock 0602.20–0602.40 exempt], Sept 15 Procl. 11064–11065 cheese [$117.49M ON], and Sept 29 Procl. "
            "11061–11062 U.S. import prohibition on packaged alcohol 2203–2208 [$799.56M ON] & whey 0404.10 [$2.4M ON]), "
            "paired with Canada's Sept 8, 2026 counter-tariff schedule (-5.4% Jorgenson CapEx shock; zero surtax on bulk "
            "diesel HS 27 or fertilizer HS 31)."
        ),
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": TARGETED_SECTION338_OPTION_LABEL, "tariff_pct": 50.0, "include_proc": True,
        "capex_pct": -5.4, "tfp_pct": None, "brm_m": 0.0,
    },
    "Hypothetical Broad 25% U.S. Blanket Tariff (All Ag Commodities — CUSMA Stress Test)": {
        "desc": (
            "Hypothetical economy-wide CUSMA breakdown stress test applying a uniform 25% U.S. border tariff across "
            "all 18 primary agriculture commodity sectors (without Ghosh domestic-processor rationing on primary crop "
            "tariffs), paired with Canada's Sept 8, 2026 machinery counter-tariff schedule (-5.4% Jorgenson CapEx shock)."
        ),
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 25.0, "include_proc": True,
        "capex_pct": -5.4, "tfp_pct": None, "brm_m": 0.0,
    },
    "Hypothetical Broad 50% U.S. Blanket Tariff (All Ag Max Stress Test)": {
        "desc": (
            "Hypothetical maximum-severity blanket border stress test applying a 50% U.S. tariff across all 18 primary "
            "agriculture sectors paired with Canada's 50% Sept 8, 2026 surtax tier on 413 steel, aluminum & plastic "
            "lines (HS 39/72/73/76 -> -11% Jorgenson CapEx shock)."
        ),
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 50.0, "include_proc": True,
        "capex_pct": -11.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2025 → 2026 OMAFA Pub 60 Empirical Cost Shift (Exact Row-Crop Benchmark)": {
        "desc": (
            "Exact acreage-weighted 2025→2026 OMAFA Publication 60 cost shift across Ontario Corn, Soybeans & "
            "Winter Wheat: Fuel -7% ($1.20→$1.12/L diesel), Fertilizer +19% (+14.6% Corn / +19% Soy / +33% Wheat), "
            "Pesticides & Custom Spraying +8.5% ($13.75→$18/ac pass), Seed +2.5%, Commercial Feed -2.3% (2025 ON FIPI), "
            "Labour +9%, and Operating Interest -150 bps (6.95%→5.45%)."
        ),
        "fuel": -7.0, "fert": 19.0, "chem": 8.5, "seed": 2.5, "feed": -2.3, "wages": 9.0, "ir_bps": -150.0,
        "tariff_comm": TARGETED_SECTION338_OPTION_LABEL, "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2022 StatCan FIPI Energy, Fertilizer & BoC Rate Hike Replay (Historical Benchmark)": {
        "desc": (
            "Exact historical replay of the 2022 Ontario Farm Input Price Index (StatCan Table 18-10-0258-01) "
            "and Bank of Canada tightening cycle: Machinery Fuel +61.2%, Fertilizer +71%, Commercial Seed +5%, "
            "Commercial Feed +18.9%, Hired Labour +7.5%, and +400 bps interest rate tightening."
        ),
        "fuel": 61.2, "fert": 71.0, "chem": 15.0, "seed": 5.0, "feed": 18.9, "wages": 7.5, "ir_bps": 400.0,
        "tariff_comm": TARGETED_SECTION338_OPTION_LABEL, "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "Leamington & Niagara CEA Greenhouse / Nursery Stress (NAICS 1114 Primary Ag)": {
        "desc": (
            "Calibrated to 2025/2026 OGVG & NAICS 1114 benchmarks: "
            "25% U.S. Section 338 tariff on Greenhouse Vegetables & HS Ch. 06 Nursery/Floriculture, paired with "
            "+25% Enbridge Zone 2 natural gas heating & carbon escalation, +14% water-soluble fertility, "
            "+12% biological IPM predators, +8% grafted seedlings/liners, and +5.5% ESDC SAWP/TFW wage increase."
        ),
        "fuel": 25.0, "fert": 14.0, "chem": 12.0, "seed": 8.0, "feed": 0.0, "wages": 5.5, "ir_bps": 0.0,
        "tariff_comm": "Greenhouse Vegetables", "tariff_pct": 25.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
}


# ------------------------------------------------------------------------------
# SIDEBAR CONTROL DECK
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔮 Simulator Control Deck")
    st.caption("Zero-Heuristic Empirical Calibration (StatCan IO 2022, FCR 2025, Expense Profiles 2024)")

    all_regions = ["Ontario", "Canada", "Alberta", "Saskatchewan", "Manitoba", "Quebec", "British Columbia", "Nova Scotia", "New Brunswick", "Prince Edward Island"]
    region = st.selectbox("Step 1: Jurisdiction / Province", all_regions, index=0)

    latest_exp_yr = 2024
    has_reg_profiles = any(str(k).startswith(f"{region}|") for k in expense_profiles.keys())
    exp_geo = region if has_reg_profiles else "Canada"
    avail_farm_types = [
        "All farm types",
        "Greenhouse, nursery and floriculture production [1114]",
        "Oilseed and grain farming [1111]",
        "Dairy cattle and milk production [112120]",
        "Dairy cattle and milk production [11212]",
        "Beef cattle ranching and farming, including feedlots [112110]",
        "Beef cattle ranching and farming, including feedlots [112111]",
        "Hog and pig farming [1122]",
        "Poultry and egg production [1123]",
        "Fruit and tree nut farming [1113]",
        "Other vegetable (except potato) and melon farming [111219]",
        "Potato farming [111211]",
        "Crop production [111]",
        "Animal production [112]",
        "Sheep and goat farming [1124]",
        "Other crop farming [1119]",
        "Other animal production [112A]",
        "Other animal production [1129]",
    ]
    avail_farm_types = [ft for ft in avail_farm_types if f"{exp_geo}|{ft}" in expense_profiles]
    if not avail_farm_types:
        avail_farm_types = [k.split("|", 1)[1] for k in expense_profiles.keys() if str(k).startswith(f"{exp_geo}|")]
    farm_type = st.selectbox(
        "Step 2: Primary Agriculture NAICS Sector",
        avail_farm_types,
        index=0,
        help="NAICS 1114 (Greenhouse, Nursery, Floriculture & Sod) is classified strictly as Primary Agriculture.",
    )

    io_geo_scope = st.radio(
        "Step 2B: IO Geographical Multiplier Scope",
        options=["all_provinces", "within_province"],
        format_func=lambda x: (
            "All-Provinces Footprint (Includes Inter-Provincial Supply Chains)"
            if x == "all_provinces"
            else "Within-Province Only (Own-Border Provincial Retention)"
        ),
        index=0,
        key="rad_io_geo_scope",
        help=(
            "Selects between 'All-Provinces' (capturing upstream backward supply-chain & induced consumer spillovers across Canada) "
            "and 'Within-Province' (isolating strictly intra-provincial backward supply-chain GDP, employment, wages & tax retention) "
            "in Statistics Canada Table 36-10-0595-01 (5th CODR dimension: Geographical coverage). Forward downstream NAICS 311/312 "
            "food & beverage processing linkages are modeled separately via Analysis-by-Parts (ABP). Farm-gate solvency "
            "(NCI, DSCR, Distress Prob, Statutory BRM) remains strictly invariant to regional IO accounting boundaries."
        ),
    )
    if region == "Canada":
        st.caption("ℹ️ At the `Canada` national level, `All-Provinces` and `Within-Province` coincide (StatCan National SUT).")

    preset_name = st.selectbox("Step 3: Quick-Load Empirical Scenario", list(SCENARIO_PRESETS.keys()), index=0)
    preset = SCENARIO_PRESETS[preset_name]
    if preset_name != "Custom Scenario (Manual Sliders)":
        st.info(preset["desc"])
    if "1114" in preset_name and farm_type == "All farm types":
        st.caption("💡 Tip: Select `Greenhouse, nursery and floriculture production [1114]` above to isolate NAICS 1114.")

    with st.expander("⛽ 1. Input Cost & Debt Rate Shocks", expanded=True):
        shock_fuel = st.slider("Fuel, Heating Gas & Electricity (%)", -30.0, 100.0, float(preset["fuel"]), 0.5, key=f"sl_fuel_{preset_name}")
        shock_fert = st.slider("Fertilizer, Lime & Nutrients (%)", -30.0, 100.0, float(preset["fert"]), 0.5, key=f"sl_fert_{preset_name}")
        shock_chem = st.slider("Pesticides & Biological IPM (%)", -30.0, 100.0, float(preset["chem"]), 0.5, key=f"sl_chem_{preset_name}")
        shock_seed = st.slider("Seed, Plugs, Liners & Media (%)", -30.0, 100.0, float(preset["seed"]), 0.5, key=f"sl_seed_{preset_name}")
        shock_feed = st.slider("Commercial Livestock Feed (%)", -30.0, 100.0, float(preset["feed"]), 0.5, key=f"sl_feed_{preset_name}")
        shock_wages = st.slider("Hired Farm & Packhouse Labour (%)", -20.0, 50.0, float(preset["wages"]), 0.5, key=f"sl_wages_{preset_name}")
        shock_ir_bps = st.slider(
            "Farm Debt Interest Rate Shock (bps)",
            -300.0,
            500.0,
            float(preset["ir_bps"]),
            25.0,
            key=f"sl_ir_{preset_name}",
            help="+100 bps = +1 percentage point added to the effective interest rate on farm debt.",
        )

    with st.expander("🛡️ 2. Section 338 Tariff & Downstream ABP", expanded=True):
        comm_options = [TARGETED_SECTION338_OPTION_LABEL, "All Commodities (Provincial FCR Weighted)"] + list(PRIMARY_AG_TARIFF_CONFIG.keys())
        default_comm_idx = comm_options.index(preset["tariff_comm"]) if preset["tariff_comm"] in comm_options else 0
        tariff_commodity = st.selectbox("Target Primary Ag Commodity / Statutory Scope", comm_options, index=default_comm_idx, key=f"sel_comm_{preset_name}")
        tariff_pct = st.slider("U.S. Section 338 Border Tariff (%)", 0.0, 50.0, float(preset["tariff_pct"]), 1.0, key=f"sl_tariff_{preset_name}")
        direct_rev_shock_pct = st.slider("Additional Non-Tariff Market Revenue Shock (%)", -30.0, 30.0, 0.0, 0.5, key=f"sl_rev_{preset_name}")
        include_processing = st.checkbox(
            "Include Downstream / Co-Targeted Processing (NAICS 311/312 ABP)",
            value=bool(preset["include_proc"]),
            key=f"chk_proc_{preset_name}",
            help="Applies official provincial StatCan IO multipliers for NAICS 3111-3121A with SUT basic-price farm input coefficients netted out to prevent double-counting, without applying Ghosh rationing on export tariffs.",
        )

    with st.expander("🏗️ 3. CapEx, ICOR & Productivity (TFP)", expanded=False):
        icor_profile_name = st.selectbox("Empirical Capital-Output (ICOR) Profile", list(EMPIRICAL_ICOR_PROFILES.keys()), index=0)
        icor_cfg = EMPIRICAL_ICOR_PROFILES[icor_profile_name]
        shock_capex_pct = st.slider("Capital Investment (CapEx) Shock (%)", -50.0, 100.0, float(preset["capex_pct"]), 0.2, key=f"sl_capex_{preset_name}")
        use_gestation_ramp = st.checkbox("Apply Multi-Year Capital Gestation Ramp", value=True)
        default_tfp = 1.85 if region == "Ontario" else 1.55
        tfp_growth_pct = st.slider(
            "Annual TFP / MFP Growth Drift (%)",
            -5.0,
            6.0,
            float(preset["tfp_pct"]) if preset.get("tfp_pct") is not None else default_tfp,
            0.05,
            key=f"sl_tfp_{preset_name}_{region}",
        )
        econ_dep_cfg = get_economic_depreciation_rate(region=region, farm_type=farm_type)
        depreciation_rate_pct = st.slider(
            "Annual Capital Depreciation Rate (%)",
            4.0,
            15.0,
            float(econ_dep_cfg["delta_pct"]),
            0.1,
            key=f"sl_dep_{region}_{farm_type}",
            help=(
                f"Empirically calibrated to StatCan Table 36-10-0096-01 & Jorgenson User-Cost benchmarks "
                f"(δ_mach = 12.0%/yr for farm machinery; δ_struct = 4.0%/yr for farm buildings & greenhouses; "
                f"Composite Primary Ag = 9.5%/yr). Active sector basis: {econ_dep_cfg['basis_label']}."
            ),
        )

    with st.expander("🏛️ 4. Policy Offset (BRM) & Monte Carlo", expanded=False):
        auto_brm_toggle = st.checkbox(
            "Auto-Trigger Statutory AgriStability & Agricorp RMP (β_BRM)",
            value=False,
            key="chk_auto_brm",
            help=(
                "Endogenously computes statutory Business Risk Management (BRM) compensation across Years 1–10 "
                "using empirical payout rates (β_BRM), the 78% allowable expense filter (λ_interest = 0.0 so interest "
                "rate spikes are ineligible), ex-post tax-filing cash timing (w_t in Year t, w_{t+1} in Year t+1), "
                "and the 5-year Olympic Average reference margin ratchet (Deep Research Output #6 Section 4)."
            ),
        )
        brm_regime = st.selectbox(
            "Statutory BRM Program Regime",
            options=["statutory_scap_80pct", "adhoc_2025_90pct"],
            format_func=lambda x: (
                "Sustainable CAP Statutory (80% Comp / 70% Trigger + ON RMP/SDRM)"
                if x == "statutory_scap_80pct"
                else "2025/2026 Ad-Hoc Trade-War Enhancement (90% Comp / $6M Cap + ON RMP/SDRM)"
            ),
            index=0,
            disabled=not auto_brm_toggle,
            key="sel_brm_regime",
        )
        brm_subsidy_m = st.slider(
            "Additional Ad-Hoc BRM / Disaster Offset ($M)",
            0.0,
            2000.0,
            float(preset["brm_m"]),
            25.0,
            key=f"sl_brm_{preset_name}",
            help="Optional manual top-up payment (e.g. AgriRecovery disaster relief or ad-hoc tariff compensation) layered on top of statutory BRM.",
        )
        horizon = st.slider("Dynamic Projection Horizon (Years)", 3, 10, 10, 1)
        mc_basis = st.radio(
            "Monte Carlo Productivity Accounting Basis",
            ["gross_output", "value_added"],
            format_func=lambda x: "Gross-Output TFP (KLEMS / USDA ERS: σ=2.9% ON / 4.2% CAN)"
            if x == "gross_output"
            else "Value-Added MFP (StatCan 36-10-0208-01: σ=4.1% ON / 6.0% CAN)",
            index=0,
        )
        n_sims = st.select_slider("Monte Carlo Simulation Runs", options=[100, 200, 500], value=200)

    with st.expander("⚖️ 5. Side-by-Side Scenario Comparison (A vs. B)", expanded=True):
        compare_preset_options = [
            f"Preset: {k}" for k in SCENARIO_PRESETS.keys() if k != "Custom Scenario (Manual Sliders)"
        ]
        compare_options = [
            "Off (Single Active Scenario)",
            "Counterfactual: Active Shocks WITHOUT BRM Offset (Pre-Policy vs. Post-Policy)",
            "Counterfactual: Opposite IO Scope (All-Provinces vs. Within-Province Spillover Split)",
            "Pinned Custom Snapshot (Scenario A)",
        ] + compare_preset_options

        if "pinned_scenario_a" not in st.session_state:
            st.session_state["pinned_scenario_a"] = None

        default_cmp_idx = 0
        if st.session_state.get("force_pinned_mode") and st.session_state.get("pinned_scenario_a") is not None:
            default_cmp_idx = 3
            st.session_state["force_pinned_mode"] = False

        compare_mode = st.selectbox(
            "Compare Active Scenario (B) Against Benchmark (A)",
            options=compare_options,
            index=default_cmp_idx,
            key="sel_compare_mode",
            help=(
                "Select any reference scenario as Benchmark Scenario A (including a 1-click All-Provinces vs. Within-Province "
                "spillover split or a pinned custom snapshot) to view side-by-side KPI deltas (B − A) and overlaid 10-year curves."
            ),
        )

        scope_tag_pin = "Within-Prov" if io_geo_scope == "within_province" else "All-Prov"
        auto_pin_label = (
            f"Pinned ({preset_name[:18]}... · {scope_tag_pin})"
            if preset_name != "Custom Scenario (Manual Sliders)"
            else f"Pinned (Tariff {tariff_pct:.0f}%, IR {shock_ir_bps:+.0f}bps · {scope_tag_pin})"
        )
        pin_label_input = st.text_input(
            "Custom Label for Pinned Scenario A",
            value=auto_pin_label,
            key="txt_pin_label",
        )
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            pin_current_btn = st.button("📌 Pin Current as A", width="stretch")
        with p_col2:
            clear_pin_btn = st.button(
                "🗑️ Clear Pin",
                width="stretch",
                disabled=(st.session_state.get("pinned_scenario_a") is None),
            )
        if clear_pin_btn:
            st.session_state["pinned_scenario_a"] = None
            st.toast("Cleared pinned Scenario A snapshot.")
        if st.session_state.get("pinned_scenario_a") is not None:
            st.caption(f"📌 **Active Pin:** `{st.session_state['pinned_scenario_a']['label']}`")



# ------------------------------------------------------------------------------
# CORE SIMULATION EXECUTION
# ------------------------------------------------------------------------------
io_years = sorted(io_df["YEAR"].dropna().unique())
target_io_year = int(io_years[-1])

# 1. Provincial FCR Weights & Baselines (32-10-0045-01)
fcr_info = cached_fcr_weights(region)
fcr_year = fcr_info["year"]

# 2. Expense Profile & Baseline Revenues/Expenses (32-10-0136-01)
prof = expense_profiles.get(
    f"{exp_geo}|{farm_type}",
    expense_profiles.get(f"{exp_geo}|All farm types", {}),
)
all_farms_prof = expense_profiles.get(f"{exp_geo}|All farm types", {})
farm_type_exp_share = (
    float(prof.get("total_expenses_raw", 1.0)) / max(float(all_farms_prof.get("total_expenses_raw", 1.0)), 1.0)
)
farm_type_rev_share = farm_type_exp_share

# 3. Balance Sheet & Financial Health Baselines
geo_col_h = "geo" if "geo" in df_health.columns else "GEO"
yr_col_h = "data_year" if "data_year" in df_health.columns else "year"
health_sub = df_health[df_health[geo_col_h] == region].sort_values(yr_col_h)
if health_sub.empty:
    health_sub = df_health[df_health[geo_col_h] == "Canada"].sort_values(yr_col_h)
latest_health = health_sub.iloc[-1]
health_year = int(latest_health[yr_col_h])

base_debt_all = float(latest_health.get("total_debt_M", latest_health.get("total_debt", 35000.0))) * 1e6
base_nci_all = float(latest_health.get("net_cash_income_M", latest_health.get("net_cash_income", 4500.0))) * 1e6
base_cash_rec_all = float(latest_health.get("cash_receipts_M", 23000.0)) * 1e6
base_interest_all = float(all_farms_prof.get("interest_dollars", base_debt_all * 0.048))
debt_amort_cfg = get_debt_amortization_params(region=region)
amortization_rate = float(debt_amort_cfg["gamma_principal"])  # 0.0658 ON / 0.0713 CAN (Deep Research Output #6)
implied_debt_rate = (base_interest_all / base_debt_all) if base_debt_all > 0 else 0.048

# Convert bps shock to relative % shock on implied interest rate
shock_interest_rel_pct = ((shock_ir_bps / 10000.0) / implied_debt_rate) * 100.0 if implied_debt_rate > 0 else 0.0

# Scale financial health baselines to selected farm_type share
base_interest = float(prof.get("interest_dollars", base_interest_all * farm_type_rev_share))
farm_type_int_share = (base_interest / base_interest_all) if base_interest_all > 0 else farm_type_rev_share
base_debt = base_debt_all * farm_type_int_share
base_nci = base_nci_all * farm_type_rev_share

# Module 1 & 2 Calibration Profiles for Selected NAICS Sector (Deep Research Output #5)
theta_schedule = get_debt_repricing_schedule(region=region, farm_type=farm_type)
yr1_theta = float(theta_schedule[0])
hotelling_prof = get_hotelling_sector_profile(farm_type=farm_type)

# 4. CapEx & Capital Stock Baselines (Steady-State K_0 = I_0 / delta so baseline K_t does not artificially decay)
geo_col_c = "geo" if "geo" in df_capex.columns else "GEO"
bsk_col_c = "basket_key" if "basket_key" in df_capex.columns else "basket"
yr_col_c = "data_year" if "data_year" in df_capex.columns else "year"
val_col_c = "total_capex_M" if "total_capex_M" in df_capex.columns else "capex_total"
capex_sub = df_capex[(df_capex[geo_col_c] == region) & (df_capex[bsk_col_c] == "primary_agriculture")].sort_values(yr_col_c)
if capex_sub.empty:
    capex_sub = df_capex[(df_capex[geo_col_c] == "Canada") & (df_capex[bsk_col_c] == "primary_agriculture")].sort_values(yr_col_c)
base_capex_all = float(capex_sub.iloc[-1][val_col_c]) * 1e6 if not capex_sub.empty else 4.5e9
base_capex = base_capex_all * farm_type_rev_share
base_cap_stock = base_capex / (depreciation_rate_pct / 100.0)

# 5. Baseline Output & GDP for Selected Farm Sector
total_exp_dollars = float(prof.get("total_expenses_raw", 0.0))
base_output_dollars = float(
    prof.get(
        "total_revenue_raw",
        (fcr_info["total_fcr_m"] * 1e6 * farm_type_rev_share) if fcr_info.get("total_fcr_m", 0) > 0 else (total_exp_dollars * 1.18),
    )
)
primary_join_code = (
    "1114A"
    if "1114" in farm_type
    else (
        "1125"
        if "1125" in farm_type
        else ("112A" if "112" in farm_type else ("111A" if "111" in farm_type else None))
    )
)

if primary_join_code in ("111A", "1114A", "112A", "1125"):
    sub_mults = load_subsector_io_multipliers(io_df, region=region, year=target_io_year, geo_scope=io_geo_scope)
    fallback_mults = {
        "111A": {"output": 1.6420, "gdp": 0.7780, "jobs": 5.1380, "wages": 0.2530},
        "1114A": {"output": 1.5530, "gdp": 0.7700, "jobs": 10.0850, "wages": 0.4600},
        "112A": {"output": 2.2720, "gdp": 0.7760, "jobs": 7.3550, "wages": 0.3380},
        "1125": {"output": 1.8000, "gdp": 0.4500, "jobs": 4.5000, "wages": 0.2200},
    }
    m_unit = sub_mults.get(primary_join_code, fallback_mults[primary_join_code])
    gdp_coeff = float(m_unit["gdp"])
    out_mult = float(m_unit["output"])
    jobs_mult_per_m = float(m_unit["jobs"])
    wages_coeff = float(m_unit["wages"])
    if io_geo_scope == "within_province" and region != "Canada":
        sub_mults_all = load_subsector_io_multipliers(io_df, region=region, year=target_io_year, geo_scope="all_provinces")
        m_unit_all = sub_mults_all.get(primary_join_code, fallback_mults[primary_join_code])
        on_farm_margin_coeff = float(m_unit_all["gdp"])
    else:
        on_farm_margin_coeff = gdp_coeff
else:
    unit_basket = compute_basket_impact(
        io_df, baskets_cfg, region, target_io_year, "primary_agriculture", 1.0, geo_scope=io_geo_scope
    )
    gdp_coeff = float(unit_basket.get("gdp", 0.52))
    out_mult = float(unit_basket.get("output", 1.65))
    jobs_mult_per_m = float(unit_basket.get("jobs", 6.2e-6)) * 1e6
    wages_coeff = float(unit_basket.get("wages", 0.25))
    if io_geo_scope == "within_province" and region != "Canada":
        unit_basket_all = compute_basket_impact(
            io_df, baskets_cfg, region, target_io_year, "primary_agriculture", 1.0, geo_scope="all_provinces"
        )
        on_farm_margin_coeff = float(unit_basket_all.get("gdp", 0.52))
    else:
        on_farm_margin_coeff = gdp_coeff

gdp_retention_pct = (gdp_coeff / max(on_farm_margin_coeff, 1e-9)) * 100.0
nominal_shock_year = 2025
primary_deflator_2022 = get_constant_2022_price_deflator(
    region=region, sector_key=primary_join_code or "all_ag", nominal_year=nominal_shock_year
)
jobs_mult_eff_per_m = jobs_mult_per_m / primary_deflator_2022

# Dynamic provincial GDP margin calibration (Fixes Gap #8A — replaces hardcoded 0.30)
base_gdp_dollars = base_output_dollars * gdp_coeff

# 6. Compute Input Cost & Interest Rate Shocks ($ for selected farm_type)
gross_ratios = prof.get("ratios", {})
fuel_r = float(gross_ratios.get("fuel", gross_ratios.get("energy", 0.0)))
fert_r = float(gross_ratios.get("fertilizer", 0.0))
if "pesticides" in gross_ratios or "seed" in gross_ratios:
    chem_r = float(gross_ratios.get("pesticides", 0.0))
    seed_r = float(gross_ratios.get("seed", 0.0))
else:
    crop_inp_r = float(gross_ratios.get("crop_inputs", 0.0))
    chem_r = 0.50 * crop_inp_r
    seed_r = 0.50 * crop_inp_r
feed_r = float(gross_ratios.get("feed", 0.0))
wage_r = float(gross_ratios.get("wages", gross_ratios.get("labor", 0.0)))
int_r = float(gross_ratios.get("interest", 0.0))

cost_shock_fuel = total_exp_dollars * fuel_r * (shock_fuel / 100.0)
cost_shock_fert = total_exp_dollars * fert_r * (shock_fert / 100.0)
cost_shock_chem = total_exp_dollars * chem_r * (shock_chem / 100.0)
cost_shock_seed = total_exp_dollars * seed_r * (shock_seed / 100.0)
cost_shock_feed = total_exp_dollars * feed_r * (shock_feed / 100.0)
cost_shock_wages = total_exp_dollars * wage_r * (shock_wages / 100.0)
# Full-portfolio Year-5 interest shock vs. Year-1 Repriced Tranche (theta_yr1)
cost_shock_int_full = base_debt * (shock_ir_bps / 10000.0) if base_debt > 0 else (total_exp_dollars * int_r * (shock_interest_rel_pct / 100.0))
cost_shock_int_yr1 = cost_shock_int_full * yr1_theta
cost_shock_int = cost_shock_int_yr1

intermediate_cost_shock_dollars = cost_shock_fuel + cost_shock_fert + cost_shock_chem + cost_shock_seed + cost_shock_feed
labor_cost_shock_dollars = cost_shock_wages
total_cost_shock_dollars = intermediate_cost_shock_dollars + labor_cost_shock_dollars + cost_shock_int_yr1

input_price_shocks_dict = {
    "fuel": shock_fuel,
    "fertilizer": shock_fert,
    "pesticides": shock_chem,
    "seed": shock_seed,
    "feed": shock_feed,
    "labor": shock_wages,
}

# 7. Compute Tariff & Revenue Shock Parameters (PE-ABP-IO Regime 1 & Targeted 2026 Section 338 Package)
s338_channel_breakdown: List[Dict[str, Any]] = []
proc_export_shocks_yr1: Dict[str, float] = {}
proc_export_shocks_yrN: Dict[str, float] = {}

if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL:
    s338_pkg_sr = get_targeted_section338_package_params(
        region=region, farm_type=farm_type, tariff_pct=tariff_pct, is_long_run=False
    )
    s338_pkg_lr = get_targeted_section338_package_params(
        region=region, farm_type=farm_type, tariff_pct=tariff_pct, is_long_run=True
    )
    s338_channel_breakdown = s338_pkg_sr["channel_breakdown"]
    proc_export_shocks_yr1 = s338_pkg_sr["processor_export_shock_by_naics_dollars"] if tariff_pct > 0 else {}
    proc_export_shocks_yrN = s338_pkg_lr["processor_export_shock_by_naics_dollars"] if tariff_pct > 0 else {}
    active_tariff_params = {
        "mode": "targeted_section338_2026",
        "commodity": TARGETED_SECTION338_OPTION_LABEL,
        "tariff_pct": tariff_pct,
        "yr1_primary_phys_dollars": float(s338_pkg_sr["yr1_primary_phys_dollars"]),
        "lr_primary_phys_dollars": float(s338_pkg_sr["lr_primary_phys_dollars"]),
        "yr1_primary_pure_price_dollars": float(s338_pkg_sr["yr1_primary_pure_price_dollars"]),
        "lr_primary_pure_price_dollars": float(s338_pkg_sr["lr_primary_pure_price_dollars"]),
        "effective_lam": float(s338_pkg_sr["effective_lam"]),
        "weight": 1.0,
    }
    yr1_tariff_shock_dollars = float(s338_pkg_sr["yr1_primary_phys_dollars"]) if tariff_pct > 0 else 0.0
    yr1_tariff_price_margin_dollars = float(s338_pkg_sr["yr1_primary_pure_price_dollars"]) if tariff_pct > 0 else 0.0
    yr1_tariff_pct_output_change = yr1_tariff_shock_dollars / max(base_output_dollars, 1.0)
    price_shock_info = {
        "effective_price_drop_pct": (abs(yr1_tariff_price_margin_dollars) / max(base_output_dollars, 1.0)) * 100.0
    }
    if tariff_pct > 0:
        lam_pkg = max(float(s338_pkg_sr["effective_lam"]), 0.01)
        decay_N = (1.0 - lam_pkg) ** (horizon - 1)
        yrN_tariff_phys_dollars = float(s338_pkg_sr["lr_primary_phys_dollars"]) + (
            float(s338_pkg_sr["yr1_primary_phys_dollars"]) - float(s338_pkg_sr["lr_primary_phys_dollars"])
        ) * decay_N
    else:
        yrN_tariff_phys_dollars = 0.0
elif tariff_commodity == "All Commodities (Provincial FCR Weighted)":
    comp_params = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"])
    active_tariff_params = {
        "commodity": f"Composite ({farm_type})",
        "tariff_pct": tariff_pct,
        "e_sr": comp_params["e_sr"],
        "e_lr": comp_params["e_lr"],
        "lam": comp_params["lam"],
        "pass_through": comp_params["pass_through"],
        "pass_through_lr": comp_params.get("pass_through_lr", comp_params["pass_through"]),
        "us_price_share": comp_params["us_price_share"],
        "us_lop_spillover": comp_params["us_lop_spillover"],
        "weight": 1.0,
    }
    price_shock_info = compute_effective_tariff_price_shock(
        tariff_pct=tariff_pct,
        pass_through=active_tariff_params["pass_through"],
        us_price_share=active_tariff_params["us_price_share"],
        us_lop_spillover=active_tariff_params["us_lop_spillover"],
    )
    yr1_tariff_pct_output_change = active_tariff_params["e_sr"] * (price_shock_info["effective_price_drop_pct"] / 100.0)
    yr1_tariff_shock_dollars = base_output_dollars * active_tariff_params["weight"] * yr1_tariff_pct_output_change
    yr1_tariff_price_margin_dollars = (
        -base_output_dollars
        * active_tariff_params["weight"]
        * max(0.0, 1.0 + yr1_tariff_pct_output_change)
        * (price_shock_info["effective_price_drop_pct"] / 100.0)
        if tariff_pct > 0
        else 0.0
    )
    if tariff_pct > 0:
        lam_std = max(float(active_tariff_params["lam"]), 0.01)
        decay_N = (1.0 - lam_std) ** (horizon - 1)
        e_N = float(active_tariff_params["e_lr"]) + (float(active_tariff_params["e_sr"]) - float(active_tariff_params["e_lr"])) * decay_N
        pt_N = float(active_tariff_params["pass_through_lr"]) + (float(active_tariff_params["pass_through"]) - float(active_tariff_params["pass_through_lr"])) * decay_N
        ps_N = compute_effective_tariff_price_shock(
            tariff_pct=tariff_pct,
            pass_through=pt_N,
            us_price_share=active_tariff_params["us_price_share"],
            us_lop_spillover=active_tariff_params["us_lop_spillover"],
        )
        yrN_tariff_phys_dollars = base_output_dollars * active_tariff_params["weight"] * e_N * (ps_N["effective_price_drop_pct"] / 100.0)
    else:
        yrN_tariff_phys_dollars = 0.0
else:
    c_params = get_commodity_tariff_params(tariff_commodity, region=region)
    comm_fcr_m = fcr_info["commodities"].get(tariff_commodity, {}).get("fcr_m", 0.0)
    sector_fcr_m = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"]).get("fcr_m", fcr_info["total_market_fcr_m"])
    comm_weight = min(1.0, comm_fcr_m / max(sector_fcr_m, 1.0)) if farm_type in ("All farm types", "Crop production [111]", "Animal production [112]", "Oilseed and grain farming [1111]", "Greenhouse, nursery and floriculture production [1114]") else 1.0
    if comm_weight <= 0.0:
        comm_weight = 0.05
    active_tariff_params = {
        "commodity": tariff_commodity,
        "tariff_pct": tariff_pct,
        "e_sr": c_params["e_sr"],
        "e_lr": c_params["e_lr"],
        "lam": c_params["lam"],
        "pass_through": c_params["pass_through"],
        "pass_through_lr": c_params.get("pass_through_lr", c_params["pass_through"]),
        "us_price_share": c_params["us_price_share"],
        "us_lop_spillover": c_params["us_lop_spillover"],
        "e_lr_p10_p90": c_params.get("e_lr_p10_p90"),
        "lam_p10_p90": c_params.get("lam_p10_p90"),
        "pass_through_p10_p90": c_params.get("pass_through_p10_p90"),
        "weight": comm_weight,
    }
    price_shock_info = compute_effective_tariff_price_shock(
        tariff_pct=tariff_pct,
        pass_through=active_tariff_params["pass_through"],
        us_price_share=active_tariff_params["us_price_share"],
        us_lop_spillover=active_tariff_params["us_lop_spillover"],
    )
    yr1_tariff_pct_output_change = active_tariff_params["e_sr"] * (price_shock_info["effective_price_drop_pct"] / 100.0)
    yr1_tariff_shock_dollars = base_output_dollars * active_tariff_params["weight"] * yr1_tariff_pct_output_change
    yr1_tariff_price_margin_dollars = (
        -base_output_dollars
        * active_tariff_params["weight"]
        * max(0.0, 1.0 + yr1_tariff_pct_output_change)
        * (price_shock_info["effective_price_drop_pct"] / 100.0)
        if tariff_pct > 0
        else 0.0
    )
    if tariff_pct > 0:
        lam_std = max(float(active_tariff_params["lam"]), 0.01)
        decay_N = (1.0 - lam_std) ** (horizon - 1)
        e_N = float(active_tariff_params["e_lr"]) + (float(active_tariff_params["e_sr"]) - float(active_tariff_params["e_lr"])) * decay_N
        pt_N = float(active_tariff_params["pass_through_lr"]) + (float(active_tariff_params["pass_through"]) - float(active_tariff_params["pass_through_lr"])) * decay_N
        ps_N = compute_effective_tariff_price_shock(
            tariff_pct=tariff_pct,
            pass_through=pt_N,
            us_price_share=active_tariff_params["us_price_share"],
            us_lop_spillover=active_tariff_params["us_lop_spillover"],
        )
        yrN_tariff_phys_dollars = base_output_dollars * active_tariff_params["weight"] * e_N * (ps_N["effective_price_drop_pct"] / 100.0)
    else:
        yrN_tariff_phys_dollars = 0.0

direct_mkt_shock_dollars = base_output_dollars * (direct_rev_shock_pct / 100.0)
brm_offset_dollars = brm_subsidy_m * 1e6

total_yr1_rev_shock_dollars = yr1_tariff_shock_dollars + direct_mkt_shock_dollars

# 8. Run Unshocked Baseline Trajectory (`df_base_dyn`) AND Shocked Trajectory (`df_dyn`)
ramp_schedule = icor_cfg["ramp_schedule"] if use_gestation_ramp else None

df_base_dyn = run_dynamic_projection(
    horizon=horizon,
    base_output=base_output_dollars,
    base_gdp=base_gdp_dollars,
    base_cap_stock=base_cap_stock,
    base_capex=base_capex,
    shock_capex_pct=0.0,
    tfp_growth_pct=default_tfp,
    depreciation_rate_pct=depreciation_rate_pct,
    cost_shock_dollars=0.0,
    rev_shock_dollars=0.0,
    io_df=io_df,
    baskets_cfg=baskets_cfg,
    region=region,
    target_year=target_io_year,
    base_debt=base_debt,
    base_interest=base_interest,
    base_nci=base_nci,
    shock_interest_pct=0.0,
    tariff_params=None,
    capex_gestation_ramp=ramp_schedule,
    primary_join_code=primary_join_code,
    farm_type=farm_type,
    input_price_shocks=None,
    intermediate_cost_shock_dollars=0.0,
    labor_cost_shock_dollars=0.0,
    shock_ir_bps=0.0,
    amortization_rate=amortization_rate,
    icor_value=icor_cfg["icor"],
    brm_subsidy_dollars=0.0,
    geo_scope=io_geo_scope,
)

df_dyn = run_dynamic_projection(
    horizon=horizon,
    base_output=base_output_dollars,
    base_gdp=base_gdp_dollars,
    base_cap_stock=base_cap_stock,
    base_capex=base_capex,
    shock_capex_pct=shock_capex_pct,
    tfp_growth_pct=tfp_growth_pct,
    depreciation_rate_pct=depreciation_rate_pct,
    cost_shock_dollars=total_cost_shock_dollars,
    rev_shock_dollars=direct_mkt_shock_dollars,
    io_df=io_df,
    baskets_cfg=baskets_cfg,
    region=region,
    target_year=target_io_year,
    base_debt=base_debt,
    base_interest=base_interest,
    base_nci=base_nci,
    shock_interest_pct=shock_interest_rel_pct,
    tariff_params=active_tariff_params if tariff_pct > 0 else None,
    capex_gestation_ramp=ramp_schedule,
    primary_join_code=primary_join_code,
    farm_type=farm_type,
    input_price_shocks=input_price_shocks_dict,
    intermediate_cost_shock_dollars=intermediate_cost_shock_dollars,
    labor_cost_shock_dollars=labor_cost_shock_dollars,
    shock_ir_bps=shock_ir_bps,
    amortization_rate=amortization_rate,
    icor_value=icor_cfg["icor"],
    brm_subsidy_dollars=brm_offset_dollars,
    auto_brm_stabilization=auto_brm_toggle,
    brm_regime=brm_regime,
    geo_scope=io_geo_scope,
)

mc_summary = cached_monte_carlo(
    n_sims_val=n_sims,
    horizon_val=horizon,
    base_output_val=base_output_dollars,
    base_gdp_val=base_gdp_dollars,
    base_cap_stock_val=base_cap_stock,
    base_capex_val=base_capex,
    shock_capex_pct_val=shock_capex_pct,
    tfp_growth_pct_val=tfp_growth_pct,
    depreciation_rate_pct_val=depreciation_rate_pct,
    cost_shock_val=total_cost_shock_dollars,
    rev_shock_val=direct_mkt_shock_dollars,
    region_val=region,
    target_year_val=target_io_year,
    base_debt_val=base_debt,
    base_interest_val=base_interest,
    base_nci_val=base_nci,
    shock_interest_pct_val=shock_interest_rel_pct,
    tariff_params_val=active_tariff_params if tariff_pct > 0 else None,
    mc_basis_val=mc_basis,
    ramp_schedule_val=tuple(ramp_schedule) if ramp_schedule is not None else None,
    primary_join_code_val=primary_join_code,
    farm_type_val=farm_type,
    input_price_shocks_val=input_price_shocks_dict,
    intermediate_cost_shock_val=intermediate_cost_shock_dollars,
    labor_cost_shock_val=labor_cost_shock_dollars,
    shock_ir_bps_val=shock_ir_bps,
    amortization_rate_val=amortization_rate,
    icor_val=icor_cfg["icor"],
    brm_subsidy_val=brm_offset_dollars,
    auto_brm_val=auto_brm_toggle,
    brm_regime_val=brm_regime,
    geo_scope_val=io_geo_scope,
)

# 9. Isolate Pure Shock Impact (Shocked Trajectory minus Unshocked Baseline Trajectory)
#    AND Downstream / Co-Targeted Processing Analysis-by-Parts (ABP) Impact
#    Wohlgenant (1989) Regime 1 Rule: Foreign border tariffs on primary exports NEVER ration
#    domestic processors via Ghosh forward linkages; ONLY non-tariff physical supply shocks
#    (drought, input spikes, capital stock shifts) + co-targeted processor export shocks apply.
base_yr1_row = df_base_dyn.iloc[0]
base_yrN_row = df_base_dyn.iloc[-1]
yr1_row = df_dyn.iloc[0]
yrN_row = df_dyn.iloc[-1]

yr1_primary_delta_output = float(yr1_row["Gross_Output"]) - float(base_yr1_row["Gross_Output"])
yr1_primary_delta_gdp = float(yr1_row["Direct_GDP"]) - float(base_yr1_row["Direct_GDP"])
yr1_nci_delta = float(yr1_row["Net_Cash_Income"]) - float(base_yr1_row["Net_Cash_Income"])
yr1_hh_induced_gdp_m = float(yr1_row["Induced_HH_GDP_Delta_M"]) - float(base_yr1_row["Induced_HH_GDP_Delta_M"])
yr1_hh_induced_jobs = float(yr1_row["Induced_HH_Jobs_Delta"]) - float(base_yr1_row["Induced_HH_Jobs_Delta"])
yr1_primary_delta_jobs = ((yr1_primary_delta_output / 1e6) * jobs_mult_eff_per_m) + yr1_hh_induced_jobs
yr1_primary_delta_wages = yr1_primary_delta_output * wages_coeff

yr1_trend_primary_output_m = (float(base_yr1_row["Gross_Output"]) - base_output_dollars) / 1e6
yr1_trend_primary_gdp_m = (float(base_yr1_row["Direct_GDP"]) - base_gdp_dollars) / 1e6

yr1_non_tariff_supply_dollars = yr1_primary_delta_output - yr1_tariff_shock_dollars

abp_res = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=yr1_non_tariff_supply_dollars,
    primary_cost_shock_dollars=intermediate_cost_shock_dollars,
    processor_export_shock_by_naics=proc_export_shocks_yr1 if include_processing else None,
    geo_scope=io_geo_scope,
)

if include_processing:
    combined_yr1_output_impact_m = (yr1_primary_delta_output / 1e6) + float(abp_res["proc_direct_output_m"])
    combined_yr1_gdp_impact_m = (yr1_primary_delta_gdp / 1e6) + float(abp_res["proc_net_gdp_m"])
    combined_yr1_jobs_impact = yr1_primary_delta_jobs + float(abp_res["proc_net_jobs"])
    combined_yr1_wages_impact_m = (yr1_primary_delta_wages / 1e6) + float(abp_res["proc_net_wages_m"])
    yr1_supply_chain_gross_out_m = ((yr1_primary_delta_output * out_mult) / 1e6) + float(abp_res["proc_net_output_m"])
else:
    combined_yr1_output_impact_m = yr1_primary_delta_output / 1e6
    combined_yr1_gdp_impact_m = yr1_primary_delta_gdp / 1e6
    combined_yr1_jobs_impact = yr1_primary_delta_jobs
    combined_yr1_wages_impact_m = yr1_primary_delta_wages / 1e6
    yr1_supply_chain_gross_out_m = (yr1_primary_delta_output * out_mult) / 1e6

yrN_primary_shock_gdp_m = (float(yrN_row["Direct_GDP"]) - float(base_yrN_row["Direct_GDP"])) / 1e6
yrN_primary_shock_out_m = (float(yrN_row["Gross_Output"]) - float(base_yrN_row["Gross_Output"])) / 1e6
cum_10yr_shock_gdp_m = float((df_dyn["Direct_GDP"] - df_base_dyn["Direct_GDP"]).sum()) / 1e6

yrN_non_tariff_supply_dollars = (yrN_primary_shock_out_m * 1e6) - yrN_tariff_phys_dollars

abp_res_yrN = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=yrN_non_tariff_supply_dollars,
    primary_cost_shock_dollars=intermediate_cost_shock_dollars,
    processor_export_shock_by_naics=proc_export_shocks_yrN if include_processing else None,
    is_long_run=True,
    geo_scope=io_geo_scope,
)
if include_processing:
    combined_yrN_gdp_impact_m = yrN_primary_shock_gdp_m + float(abp_res_yrN["proc_net_gdp_m"])
    combined_yrN_out_impact_m = yrN_primary_shock_out_m + float(abp_res_yrN["proc_direct_output_m"])
else:
    combined_yrN_gdp_impact_m = yrN_primary_shock_gdp_m
    combined_yrN_out_impact_m = yrN_primary_shock_out_m

# 9B. Side-by-Side Scenario Comparison (Scenario A Benchmark vs. Active Scenario B)
if pin_current_btn:
    st.session_state["pinned_scenario_a"] = {
        "label": pin_label_input.strip() or "Pinned Scenario A",
        "fuel": shock_fuel,
        "fert": shock_fert,
        "chem": shock_chem,
        "seed": shock_seed,
        "feed": shock_feed,
        "wages": shock_wages,
        "ir_bps": shock_ir_bps,
        "tariff_comm": tariff_commodity,
        "tariff_pct": tariff_pct,
        "direct_rev_pct": direct_rev_shock_pct,
        "include_proc": include_processing,
        "capex_pct": shock_capex_pct,
        "tfp_pct": tfp_growth_pct,
        "brm_m": brm_subsidy_m,
        "auto_brm": auto_brm_toggle,
        "brm_regime": brm_regime,
        "geo_scope": io_geo_scope,
    }
    st.toast(f"📌 Pinned current state as Scenario A: {st.session_state['pinned_scenario_a']['label']}")

scen_a_cfg: Optional[Dict[str, Any]] = None
if compare_mode == "Counterfactual: Active Shocks WITHOUT BRM Offset (Pre-Policy vs. Post-Policy)":
    scen_a_cfg = {
        "label": "Pre-Policy (No BRM Offset)",
        "fuel": shock_fuel,
        "fert": shock_fert,
        "chem": shock_chem,
        "seed": shock_seed,
        "feed": shock_feed,
        "wages": shock_wages,
        "ir_bps": shock_ir_bps,
        "tariff_comm": tariff_commodity,
        "tariff_pct": tariff_pct,
        "direct_rev_pct": direct_rev_shock_pct,
        "include_proc": include_processing,
        "capex_pct": shock_capex_pct,
        "tfp_pct": tfp_growth_pct,
        "brm_m": 0.0,
        "auto_brm": False,
        "brm_regime": brm_regime,
        "geo_scope": io_geo_scope,
    }
elif compare_mode == "Counterfactual: Opposite IO Scope (All-Provinces vs. Within-Province Spillover Split)":
    opp_scope = "all_provinces" if io_geo_scope == "within_province" else "within_province"
    opp_lbl = "All-Provinces Footprint" if opp_scope == "all_provinces" else "Within-Province Only"
    scen_a_cfg = {
        "label": f"Same Shocks [{opp_lbl}]",
        "fuel": shock_fuel,
        "fert": shock_fert,
        "chem": shock_chem,
        "seed": shock_seed,
        "feed": shock_feed,
        "wages": shock_wages,
        "ir_bps": shock_ir_bps,
        "tariff_comm": tariff_commodity,
        "tariff_pct": tariff_pct,
        "direct_rev_pct": direct_rev_shock_pct,
        "include_proc": include_processing,
        "capex_pct": shock_capex_pct,
        "tfp_pct": tfp_growth_pct,
        "brm_m": brm_subsidy_m,
        "auto_brm": auto_brm_toggle,
        "brm_regime": brm_regime,
        "geo_scope": opp_scope,
    }
elif compare_mode == "Pinned Custom Snapshot (Scenario A)" or pin_current_btn:
    if st.session_state.get("pinned_scenario_a") is not None:
        scen_a_cfg = st.session_state["pinned_scenario_a"]
elif compare_mode.startswith("Preset: "):
    preset_key_a = compare_mode.replace("Preset: ", "", 1)
    if preset_key_a in SCENARIO_PRESETS:
        pr_a = SCENARIO_PRESETS[preset_key_a]
        scen_a_cfg = {
            "label": preset_key_a.split("(")[0].strip(),
            "fuel": float(pr_a["fuel"]),
            "fert": float(pr_a["fert"]),
            "chem": float(pr_a["chem"]),
            "seed": float(pr_a["seed"]),
            "feed": float(pr_a["feed"]),
            "wages": float(pr_a["wages"]),
            "ir_bps": float(pr_a["ir_bps"]),
            "tariff_comm": str(pr_a["tariff_comm"]),
            "tariff_pct": float(pr_a["tariff_pct"]),
            "direct_rev_pct": 0.0,
            "include_proc": bool(pr_a["include_proc"]),
            "capex_pct": float(pr_a["capex_pct"]),
            "tfp_pct": float(pr_a["tfp_pct"]) if pr_a.get("tfp_pct") is not None else default_tfp,
            "brm_m": float(pr_a["brm_m"]),
            "auto_brm": False,
            "brm_regime": "statutory_scap_80pct",
            "geo_scope": io_geo_scope,
        }

df_dyn_a: Optional[pd.DataFrame] = None
comparison_summary: Optional[Dict[str, Any]] = None

if scen_a_cfg is not None:
    scen_a_geo_scope = str(scen_a_cfg.get("geo_scope", io_geo_scope))
    if scen_a_geo_scope == io_geo_scope:
        base_gdp_dollars_a = base_gdp_dollars
        df_base_dyn_a = df_base_dyn
        jobs_mult_eff_per_m_a = jobs_mult_eff_per_m
    else:
        if primary_join_code in ("111A", "1114A", "112A", "1125"):
            sub_mults_a = load_subsector_io_multipliers(io_df, region=region, year=target_io_year, geo_scope=scen_a_geo_scope)
            m_unit_a = sub_mults_a.get(primary_join_code, fallback_mults[primary_join_code])
            gdp_coeff_a = float(m_unit_a["gdp"])
            jobs_mult_per_m_a = float(m_unit_a["jobs"])
        else:
            unit_basket_a = compute_basket_impact(
                io_df, baskets_cfg, region, target_io_year, "primary_agriculture", 1.0, geo_scope=scen_a_geo_scope
            )
            gdp_coeff_a = float(unit_basket_a.get("gdp", 0.52))
            jobs_mult_per_m_a = float(unit_basket_a.get("jobs", 6.2e-6)) * 1e6
        jobs_mult_eff_per_m_a = jobs_mult_per_m_a / primary_deflator_2022
        base_gdp_dollars_a = base_output_dollars * gdp_coeff_a
        df_base_dyn_a = run_dynamic_projection(
            horizon=horizon,
            base_output=base_output_dollars,
            base_gdp=base_gdp_dollars_a,
            base_cap_stock=base_cap_stock,
            base_capex=base_capex,
            shock_capex_pct=0.0,
            tfp_growth_pct=default_tfp,
            depreciation_rate_pct=depreciation_rate_pct,
            cost_shock_dollars=0.0,
            rev_shock_dollars=0.0,
            io_df=io_df,
            baskets_cfg=baskets_cfg,
            region=region,
            target_year=target_io_year,
            base_debt=base_debt,
            base_interest=base_interest,
            base_nci=base_nci,
            shock_interest_pct=0.0,
            tariff_params=None,
            capex_gestation_ramp=ramp_schedule,
            primary_join_code=primary_join_code,
            farm_type=farm_type,
            input_price_shocks=None,
            intermediate_cost_shock_dollars=0.0,
            labor_cost_shock_dollars=0.0,
            shock_ir_bps=0.0,
            amortization_rate=amortization_rate,
            icor_value=icor_cfg["icor"],
            brm_subsidy_dollars=0.0,
            geo_scope=scen_a_geo_scope,
        )

    cs_fuel_a = total_exp_dollars * fuel_r * (float(scen_a_cfg["fuel"]) / 100.0)
    cs_fert_a = total_exp_dollars * fert_r * (float(scen_a_cfg["fert"]) / 100.0)
    cs_chem_a = total_exp_dollars * chem_r * (float(scen_a_cfg["chem"]) / 100.0)
    cs_seed_a = total_exp_dollars * seed_r * (float(scen_a_cfg["seed"]) / 100.0)
    cs_feed_a = total_exp_dollars * feed_r * (float(scen_a_cfg["feed"]) / 100.0)
    cs_wages_a = total_exp_dollars * wage_r * (float(scen_a_cfg["wages"]) / 100.0)
    ir_bps_a = float(scen_a_cfg["ir_bps"])
    ir_rel_pct_a = ((ir_bps_a / 10000.0) / implied_debt_rate) * 100.0 if implied_debt_rate > 0 else 0.0
    cs_int_yr1_a = (base_debt * (ir_bps_a / 10000.0) if base_debt > 0 else (total_exp_dollars * int_r * (ir_rel_pct_a / 100.0))) * yr1_theta

    int_cost_a = cs_fuel_a + cs_fert_a + cs_chem_a + cs_seed_a + cs_feed_a
    tot_cost_a = int_cost_a + cs_wages_a + cs_int_yr1_a
    inp_shocks_a = {
        "fuel": float(scen_a_cfg["fuel"]),
        "fertilizer": float(scen_a_cfg["fert"]),
        "pesticides": float(scen_a_cfg["chem"]),
        "seed": float(scen_a_cfg["seed"]),
        "feed": float(scen_a_cfg["feed"]),
        "labor": float(scen_a_cfg["wages"]),
    }

    t_comm_a = str(scen_a_cfg["tariff_comm"])
    t_pct_a = float(scen_a_cfg["tariff_pct"])
    proc_shocks_a_yr1: Dict[str, float] = {}
    proc_shocks_a_yrN: Dict[str, float] = {}
    inc_proc_a = bool(scen_a_cfg.get("include_proc", True))

    if t_comm_a == TARGETED_SECTION338_OPTION_LABEL:
        pkg_a_sr = get_targeted_section338_package_params(region=region, farm_type=farm_type, tariff_pct=t_pct_a, is_long_run=False)
        pkg_a_lr = get_targeted_section338_package_params(region=region, farm_type=farm_type, tariff_pct=t_pct_a, is_long_run=True)
        proc_shocks_a_yr1 = pkg_a_sr["processor_export_shock_by_naics_dollars"] if t_pct_a > 0 else {}
        proc_shocks_a_yrN = pkg_a_lr["processor_export_shock_by_naics_dollars"] if t_pct_a > 0 else {}
        t_params_a = {
            "mode": "targeted_section338_2026",
            "commodity": TARGETED_SECTION338_OPTION_LABEL,
            "tariff_pct": t_pct_a,
            "yr1_primary_phys_dollars": float(pkg_a_sr["yr1_primary_phys_dollars"]),
            "lr_primary_phys_dollars": float(pkg_a_sr["lr_primary_phys_dollars"]),
            "yr1_primary_pure_price_dollars": float(pkg_a_sr["yr1_primary_pure_price_dollars"]),
            "lr_primary_pure_price_dollars": float(pkg_a_sr["lr_primary_pure_price_dollars"]),
            "effective_lam": float(pkg_a_sr["effective_lam"]),
            "weight": 1.0,
        }
        yr1_t_phys_a = float(pkg_a_sr["yr1_primary_phys_dollars"]) if t_pct_a > 0 else 0.0
        lam_a = max(float(pkg_a_sr["effective_lam"]), 0.01)
        decay_N_a = (1.0 - lam_a) ** (horizon - 1)
        yrN_t_phys_a = (
            float(pkg_a_sr["lr_primary_phys_dollars"])
            + (float(pkg_a_sr["yr1_primary_phys_dollars"]) - float(pkg_a_sr["lr_primary_phys_dollars"])) * decay_N_a
            if t_pct_a > 0
            else 0.0
        )
    elif t_comm_a == "All Commodities (Provincial FCR Weighted)":
        comp_p_a = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"])
        t_params_a = {
            "commodity": f"Composite ({farm_type})",
            "tariff_pct": t_pct_a,
            "e_sr": comp_p_a["e_sr"],
            "e_lr": comp_p_a["e_lr"],
            "lam": comp_p_a["lam"],
            "pass_through": comp_p_a["pass_through"],
            "pass_through_lr": comp_p_a.get("pass_through_lr", comp_p_a["pass_through"]),
            "us_price_share": comp_p_a["us_price_share"],
            "us_lop_spillover": comp_p_a["us_lop_spillover"],
            "weight": 1.0,
        }
        ps_a1 = compute_effective_tariff_price_shock(
            tariff_pct=t_pct_a, pass_through=t_params_a["pass_through"],
            us_price_share=t_params_a["us_price_share"], us_lop_spillover=t_params_a["us_lop_spillover"],
        )
        yr1_t_phys_a = base_output_dollars * t_params_a["e_sr"] * (ps_a1["effective_price_drop_pct"] / 100.0) if t_pct_a > 0 else 0.0
        decay_N_a = (1.0 - max(float(t_params_a["lam"]), 0.01)) ** (horizon - 1)
        e_N_a = t_params_a["e_lr"] + (t_params_a["e_sr"] - t_params_a["e_lr"]) * decay_N_a
        pt_N_a = t_params_a["pass_through_lr"] + (t_params_a["pass_through"] - t_params_a["pass_through_lr"]) * decay_N_a
        ps_aN = compute_effective_tariff_price_shock(
            tariff_pct=t_pct_a, pass_through=pt_N_a,
            us_price_share=t_params_a["us_price_share"], us_lop_spillover=t_params_a["us_lop_spillover"],
        )
        yrN_t_phys_a = base_output_dollars * e_N_a * (ps_aN["effective_price_drop_pct"] / 100.0) if t_pct_a > 0 else 0.0
    else:
        cp_a = get_commodity_tariff_params(t_comm_a, region=region)
        c_fcr_a = fcr_info["commodities"].get(t_comm_a, {}).get("fcr_m", 0.0)
        s_fcr_a = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"]).get("fcr_m", fcr_info["total_market_fcr_m"])
        c_wt_a = min(1.0, c_fcr_a / max(s_fcr_a, 1.0)) if farm_type in ("All farm types", "Crop production [111]", "Animal production [112]", "Oilseed and grain farming [1111]", "Greenhouse, nursery and floriculture production [1114]") else 1.0
        if c_wt_a <= 0.0:
            c_wt_a = 0.05
        t_params_a = {
            "commodity": t_comm_a,
            "tariff_pct": t_pct_a,
            "e_sr": cp_a["e_sr"],
            "e_lr": cp_a["e_lr"],
            "lam": cp_a["lam"],
            "pass_through": cp_a["pass_through"],
            "pass_through_lr": cp_a.get("pass_through_lr", cp_a["pass_through"]),
            "us_price_share": cp_a["us_price_share"],
            "us_lop_spillover": cp_a["us_lop_spillover"],
            "weight": c_wt_a,
        }
        ps_a1 = compute_effective_tariff_price_shock(
            tariff_pct=t_pct_a, pass_through=t_params_a["pass_through"],
            us_price_share=t_params_a["us_price_share"], us_lop_spillover=t_params_a["us_lop_spillover"],
        )
        yr1_t_phys_a = base_output_dollars * c_wt_a * t_params_a["e_sr"] * (ps_a1["effective_price_drop_pct"] / 100.0) if t_pct_a > 0 else 0.0
        decay_N_a = (1.0 - max(float(t_params_a["lam"]), 0.01)) ** (horizon - 1)
        e_N_a = t_params_a["e_lr"] + (t_params_a["e_sr"] - t_params_a["e_lr"]) * decay_N_a
        pt_N_a = t_params_a["pass_through_lr"] + (t_params_a["pass_through"] - t_params_a["pass_through_lr"]) * decay_N_a
        ps_aN = compute_effective_tariff_price_shock(
            tariff_pct=t_pct_a, pass_through=pt_N_a,
            us_price_share=t_params_a["us_price_share"], us_lop_spillover=t_params_a["us_lop_spillover"],
        )
        yrN_t_phys_a = base_output_dollars * c_wt_a * e_N_a * (ps_aN["effective_price_drop_pct"] / 100.0) if t_pct_a > 0 else 0.0

    dir_mkt_a = base_output_dollars * (float(scen_a_cfg.get("direct_rev_pct", 0.0)) / 100.0)
    brm_off_a = float(scen_a_cfg.get("brm_m", 0.0)) * 1e6

    df_dyn_a = run_dynamic_projection(
        horizon=horizon,
        base_output=base_output_dollars,
        base_gdp=base_gdp_dollars_a,
        base_cap_stock=base_cap_stock,
        base_capex=base_capex,
        shock_capex_pct=float(scen_a_cfg.get("capex_pct", 0.0)),
        tfp_growth_pct=float(scen_a_cfg.get("tfp_pct", default_tfp)),
        depreciation_rate_pct=depreciation_rate_pct,
        cost_shock_dollars=tot_cost_a,
        rev_shock_dollars=dir_mkt_a,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region=region,
        target_year=target_io_year,
        base_debt=base_debt,
        base_interest=base_interest,
        base_nci=base_nci,
        shock_interest_pct=ir_rel_pct_a,
        tariff_params=t_params_a if t_pct_a > 0 else None,
        capex_gestation_ramp=ramp_schedule,
        primary_join_code=primary_join_code,
        farm_type=farm_type,
        input_price_shocks=inp_shocks_a,
        intermediate_cost_shock_dollars=int_cost_a,
        labor_cost_shock_dollars=cs_wages_a,
        shock_ir_bps=ir_bps_a,
        amortization_rate=amortization_rate,
        icor_value=icor_cfg["icor"],
        brm_subsidy_dollars=brm_off_a,
        auto_brm_stabilization=bool(scen_a_cfg.get("auto_brm", False)),
        brm_regime=str(scen_a_cfg.get("brm_regime", "statutory_scap_80pct")),
        geo_scope=scen_a_geo_scope,
    )

    yr1_prim_out_a = float(df_dyn_a.iloc[0]["Gross_Output"]) - float(df_base_dyn_a.iloc[0]["Gross_Output"])
    yrN_prim_out_a = float(df_dyn_a.iloc[-1]["Gross_Output"]) - float(df_base_dyn_a.iloc[-1]["Gross_Output"])
    abp_yr1_a = compute_downstream_abp_impact(
        io_df=io_df,
        region=region,
        year=target_io_year,
        farm_type=farm_type,
        primary_supply_shock_dollars=yr1_prim_out_a - yr1_t_phys_a,
        primary_cost_shock_dollars=int_cost_a,
        processor_export_shock_by_naics=proc_shocks_a_yr1 if inc_proc_a else None,
        is_long_run=False,
        geo_scope=scen_a_geo_scope,
    )
    abp_yrN_a = compute_downstream_abp_impact(
        io_df=io_df,
        region=region,
        year=target_io_year,
        farm_type=farm_type,
        primary_supply_shock_dollars=yrN_prim_out_a - yrN_t_phys_a,
        primary_cost_shock_dollars=int_cost_a,
        processor_export_shock_by_naics=proc_shocks_a_yrN if inc_proc_a else None,
        is_long_run=True,
        geo_scope=scen_a_geo_scope,
    )
    active_b_label = (
        preset_name.split("(")[0].strip()
        if preset_name != "Custom Scenario (Manual Sliders)"
        else "Active Custom Scenario"
    )
    if auto_brm_toggle:
        active_b_label += " (+ Statutory BRM)"
    elif brm_subsidy_m > 0:
        active_b_label += f" (+ ${brm_subsidy_m:,.0f}M BRM)"
    if scen_a_geo_scope != io_geo_scope:
        b_scope_lbl = "Within-Province Only" if io_geo_scope == "within_province" else "All-Provinces Footprint"
        active_b_label += f" [{b_scope_lbl}]"

    comparison_summary = compute_scenario_comparison_summary(
        label_a=str(scen_a_cfg["label"]),
        label_b=active_b_label,
        df_dyn_a=df_dyn_a,
        df_dyn_b=df_dyn,
        df_base_dyn=df_base_dyn,
        abp_yr1_a=abp_yr1_a,
        abp_yrN_a=abp_yrN_a,
        abp_yr1_b=abp_res,
        abp_yrN_b=abp_res_yrN,
        jobs_mult_per_m=jobs_mult_eff_per_m,
        include_proc_a=inc_proc_a,
        include_proc_b=include_processing,
        brm_adhoc_m_a=float(scen_a_cfg.get("brm_m", 0.0)),
        brm_adhoc_m_b=brm_subsidy_m,
        df_base_dyn_a=df_base_dyn_a,
        jobs_mult_per_m_a=jobs_mult_eff_per_m_a,
    )

# Factor-by-Factor Attribution using CSNA HFCE Bridge, Source/Regime MPCs, Duality, ABP & Repricing Equations
hh_scoped = get_scoped_household_mpc_multipliers(
    io_df=io_df, region=region, year=target_io_year, geo_scope=io_geo_scope
)
yr1_auto_brm_cash_m = float(yr1_row.get("Auto_BRM_Cash_M", 0.0))
yr1_auto_brm_ent_m = float(yr1_row.get("Auto_BRM_Entitlement_M", 0.0))
yr1_auto_brm_lag_m = max(0.0, yr1_auto_brm_ent_m - yr1_auto_brm_cash_m)
yr1_eff_beta_brm_pct = float(yr1_row.get("Effective_Beta_BRM_Pct", 0.0))
yr1_brm_tier = str(yr1_row.get("BRM_Severity_Tier", "none"))
total_yr1_brm_cash_m = brm_subsidy_m + yr1_auto_brm_cash_m

yr1_mkt_nci_delta = yr1_nci_delta - (yr1_auto_brm_cash_m * 1e6) - brm_offset_dollars
is_yr1_mkt_distress = (
    float(yr1_row["DSCR"]) < float(debt_amort_cfg["dscr_distress_trigger"])
    and yr1_mkt_nci_delta < 0.0
)
mpc_mkt_yr1 = float(
    hh_scoped["mpc_mkt_distress_sr"] if is_yr1_mkt_distress else hh_scoped["mpc_mkt_normal_sr"]
)
mpc_stat_yr1 = float(hh_scoped["mpc_stat_brm_sr"])
mpc_adhoc_yr1 = float(hh_scoped["mpc_adhoc_brm_sr"])
hh_gdp_mult_yr1 = float(hh_scoped["gdp_basic_prices_mult"])
hfce_deflator_2022 = get_constant_2022_price_deflator(
    region=region, sector_key="hfce", nominal_year=nominal_shock_year
)
hh_jobs_mult_eff_yr1 = float(hh_scoped["jobs_per_million_consumption_2022"]) / hfce_deflator_2022

eff_hh_gdp_mkt = mpc_mkt_yr1 * hh_gdp_mult_yr1
eff_hh_jobs_mkt = mpc_mkt_yr1 * hh_jobs_mult_eff_yr1
eff_hh_gdp_stat = mpc_stat_yr1 * hh_gdp_mult_yr1
eff_hh_jobs_stat = mpc_stat_yr1 * hh_jobs_mult_eff_yr1
eff_hh_gdp_adhoc = mpc_adhoc_yr1 * hh_gdp_mult_yr1
eff_hh_jobs_adhoc = mpc_adhoc_yr1 * hh_jobs_mult_eff_yr1

inp_elasts = hotelling_prof["inputs"]
sector_yr1_margin_share = float(hotelling_prof["yr1_margin_share"])

# Exact linear ABP response coefficients from `compute_downstream_abp_impact`
abp_unit_supply = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=1e6,
    primary_cost_shock_dollars=0.0,
    geo_scope=io_geo_scope,
)
abp_unit_cost = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=0.0,
    primary_cost_shock_dollars=1e6,
    geo_scope=io_geo_scope,
)
proc_out_per_m_supply = float(abp_unit_supply["proc_direct_output_m"]) if include_processing else 0.0
proc_gdp_per_m_supply = float(abp_unit_supply["proc_net_gdp_m"]) if include_processing else 0.0
proc_jobs_per_m_supply = float(abp_unit_supply["proc_net_jobs"]) if include_processing else 0.0
proc_out_per_m_cost = float(abp_unit_cost["proc_direct_output_m"]) if include_processing else 0.0
proc_gdp_per_m_cost = float(abp_unit_cost["proc_net_gdp_m"]) if include_processing else 0.0
proc_jobs_per_m_cost = float(abp_unit_cost["proc_net_jobs"]) if include_processing else 0.0


def _build_input_factor_row(
    label: str,
    setting: str,
    inp_key: str,
    shock_pct: float,
    cost_dollars: float,
    is_labor: bool = False,
) -> Optional[Dict[str, Any]]:
    if abs(shock_pct) < 1e-4 and abs(cost_dollars) < 1e3:
        return None
    icfg = inp_elasts[inp_key]
    eps_sr = float(icfg["eps_Y_w_sr"])
    m_share = sector_yr1_margin_share
    phys_out_d = base_output_dollars * eps_sr * (shock_pct / 100.0)
    margin_comp_d = cost_dollars * m_share
    nci_delta_d = (phys_out_d * on_farm_margin_coeff) - margin_comp_d
    va_gdp_d = (phys_out_d * gdp_coeff) - (0.0 if is_labor else margin_comp_d) + (nci_delta_d * eff_hh_gdp_mkt)
    int_cost_for_abp_m = 0.0 if is_labor else (cost_dollars / 1e6)
    proc_out_m = ((phys_out_d / 1e6) * proc_out_per_m_supply) + (int_cost_for_abp_m * proc_out_per_m_cost)
    proc_gdp_m = ((phys_out_d / 1e6) * proc_gdp_per_m_supply) + (int_cost_for_abp_m * proc_gdp_per_m_cost)
    proc_jobs = ((phys_out_d / 1e6) * proc_jobs_per_m_supply) + (int_cost_for_abp_m * proc_jobs_per_m_cost)
    jobs_d = ((phys_out_d / 1e6) * jobs_mult_eff_per_m) + proc_jobs + ((nci_delta_d / 1e6) * eff_hh_jobs_mkt)
    return {
        "Shock Factor": label,
        "Slider Setting": setting,
        "Yr 1 Farm Margin / NCI ($M)": nci_delta_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": phys_out_d / 1e6,
        "Downstream Proc Direct Output ($M)": proc_out_m,
        "Yr 1 Direct Output ($M)": (phys_out_d / 1e6) + proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (va_gdp_d / 1e6) + proc_gdp_m,
        "Employment Impact (Jobs)": jobs_d,
        "Economic Transmission Mechanism": (
            f"Hotelling Duality ({m_share*100:.1f}% Yr-1 Margin Squeeze / {(1-m_share)*100:.1f}% Physical Output)"
        ),
    }


factor_attribution_rows: List[Dict[str, Any]] = []
for row_cand in [
    _build_input_factor_row("Fuel, Heating Gas & Electricity", f"{_fmt_ofa_num(shock_fuel, 1, True)}%", "fuel", shock_fuel, cost_shock_fuel),
    _build_input_factor_row("Fertilizer, Lime & Nutrients", f"{_fmt_ofa_num(shock_fert, 1, True)}%", "fertilizer", shock_fert, cost_shock_fert),
    _build_input_factor_row("Pesticides & Biological IPM", f"{_fmt_ofa_num(shock_chem, 1, True)}%", "pesticides", shock_chem, cost_shock_chem),
    _build_input_factor_row("Seed, Plugs, Liners & Media", f"{_fmt_ofa_num(shock_seed, 1, True)}%", "seed", shock_seed, cost_shock_seed),
    _build_input_factor_row("Commercial Livestock Feed", f"{_fmt_ofa_num(shock_feed, 1, True)}%", "feed", shock_feed, cost_shock_feed),
    _build_input_factor_row("Hired Farm & Packhouse Labour", f"{_fmt_ofa_num(shock_wages, 1, True)}%", "labor", shock_wages, cost_shock_wages, is_labor=True),
]:
    if row_cand is not None:
        factor_attribution_rows.append(row_cand)

# Interest Rate Factor Row
if abs(shock_ir_bps) > 1e-4:
    eff_yr1_bps = shock_ir_bps * yr1_theta
    ir_phys_out_d = base_output_dollars * (-0.0015) * (eff_yr1_bps / 100.0)
    ir_nci_d = (ir_phys_out_d * on_farm_margin_coeff) - cost_shock_int_yr1
    ir_hh_gdp_d = ir_nci_d * eff_hh_gdp_mkt
    ir_va_gdp_d = (ir_phys_out_d * gdp_coeff) + ir_hh_gdp_d
    ir_proc_out_m = (ir_phys_out_d / 1e6) * proc_out_per_m_supply
    ir_proc_gdp_m = (ir_phys_out_d / 1e6) * proc_gdp_per_m_supply
    ir_proc_jobs = (ir_phys_out_d / 1e6) * proc_jobs_per_m_supply
    ir_jobs_d = ((ir_phys_out_d / 1e6) * jobs_mult_eff_per_m) + ir_proc_jobs + ((ir_nci_d / 1e6) * eff_hh_jobs_mkt)
    factor_attribution_rows.append({
        "Shock Factor": "Farm Debt Carrying Cost (Interest Rate)",
        "Slider Setting": f"{_fmt_ofa_num(shock_ir_bps, 0, True)} bps (Yr 1 Eff: {_fmt_ofa_num(eff_yr1_bps, 1, True)} bps on {yr1_theta*100:.0f}% Repriced)",
        "Yr 1 Farm Margin / NCI ($M)": ir_nci_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": ir_phys_out_d / 1e6,
        "Downstream Proc Direct Output ($M)": ir_proc_out_m,
        "Yr 1 Direct Output ($M)": (ir_phys_out_d / 1e6) + ir_proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (ir_va_gdp_d / 1e6) + ir_proc_gdp_m,
        "Employment Impact (Jobs)": ir_jobs_d,
        "Economic Transmission Mechanism": (
            f"{yr1_theta*100:.0f}% Debt Repriced in Yr 1 ({_fmt_ofa_num(-cost_shock_int_yr1/1e6, 1, True)}M vs. {_fmt_ofa_num(-cost_shock_int_full/1e6, 1, True)}M Yr 5) + Yr 2–10 CapEx Crowd-Out"
        ),
    })

# Tariff (Primary Agriculture + Co-Targeted Processor Channels), Market Revenue, TFP & BRM Factor Rows
yr1_tfp_dev_out_d = base_output_dollars * ((tfp_growth_pct - default_tfp) / 100.0) if abs(tfp_growth_pct - default_tfp) > 1e-4 else 0.0

if tariff_pct > 0:
    if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL:
        t_lbl_prim = "U.S. Section 338 Tariff (Primary Agriculture)"
        t_set_prim = f"{_fmt_ofa_num(tariff_pct, 0)}% Procl. 11048/11064 Annexes"
        t_mech_prim = "Targeted HS Ch. 06 Cut Flowers/Bulbs + P5 Dairy Blend Dilution + Unpooled Goat/Honey + Distillery Corn Basis"
    else:
        short_comm = "All Ag Commodities" if tariff_commodity == "All Commodities (Provincial FCR Weighted)" else tariff_commodity
        t_lbl_prim = f"U.S. Section 338 Tariff ({short_comm})"
        t_set_prim = f"{_fmt_ofa_num(tariff_pct, 1)}% tariff (-{price_shock_info['effective_price_drop_pct']:.2f}% eff. price)"
        t_mech_prim = (
            f"Farm-Gate Price Squeeze (-{price_shock_info['effective_price_drop_pct']:.2f}%) + Nerlove Supply Response (Zero Ghosh Processor Rationing)"
        )

    if abs(yr1_tariff_shock_dollars) >= 1e3 or abs(yr1_tariff_price_margin_dollars) >= 1e3:
        t_nci_d = (yr1_tariff_shock_dollars * on_farm_margin_coeff) + yr1_tariff_price_margin_dollars
        t_va_gdp_d = (yr1_tariff_shock_dollars * gdp_coeff) + (t_nci_d * eff_hh_gdp_mkt)
        t_jobs_d = ((yr1_tariff_shock_dollars / 1e6) * jobs_mult_eff_per_m) + ((t_nci_d / 1e6) * eff_hh_jobs_mkt)
        factor_attribution_rows.append({
            "Shock Factor": t_lbl_prim,
            "Slider Setting": t_set_prim,
            "Yr 1 Farm Margin / NCI ($M)": t_nci_d / 1e6,
            "Yr 1 Physical Ag Output ($M)": yr1_tariff_shock_dollars / 1e6,
            "Downstream Proc Direct Output ($M)": 0.0,
            "Yr 1 Direct Output ($M)": yr1_tariff_shock_dollars / 1e6,
            "Yr 1 Value-Added GDP ($M)": t_va_gdp_d / 1e6,
            "Employment Impact (Jobs)": t_jobs_d,
            "Economic Transmission Mechanism": t_mech_prim,
        })

    if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL and include_processing and proc_export_shocks_yr1:
        abp_proc_only = compute_downstream_abp_impact(
            io_df=io_df,
            region=region,
            year=target_io_year,
            farm_type=farm_type,
            primary_supply_shock_dollars=0.0,
            primary_cost_shock_dollars=0.0,
            processor_export_shock_by_naics=proc_export_shocks_yr1,
            geo_scope=io_geo_scope,
        )
        if abs(float(abp_proc_only["proc_direct_output_m"])) > 1e-3:
            factor_attribution_rows.append({
                "Shock Factor": "U.S. Section 338 (Co-Targeted Processors)",
                "Slider Setting": "Sept 15 Cheese + Sept 29 Alcohol/Whey Ban",
                "Yr 1 Farm Margin / NCI ($M)": 0.0,
                "Yr 1 Physical Ag Output ($M)": 0.0,
                "Downstream Proc Direct Output ($M)": float(abp_proc_only["proc_direct_output_m"]),
                "Yr 1 Direct Output ($M)": float(abp_proc_only["proc_direct_output_m"]),
                "Yr 1 Value-Added GDP ($M)": float(abp_proc_only["proc_net_gdp_m"]),
                "Employment Impact (Jobs)": float(abp_proc_only["proc_net_jobs"]),
                "Economic Transmission Mechanism": (
                    "Co-Targeted NAICS 3121A/31212/3115/3118 Direct Export Shock (Upstream Farm Inputs Netted via ABP)"
                ),
            })

for f_lbl, f_set, f_out_d, f_extra_nci_d, f_hh_gdp_mult_i, f_hh_jobs_mult_i, f_mech in [
    (
        "Non-Tariff Market Revenue Shock",
        f"{_fmt_ofa_num(direct_rev_shock_pct, 1, True)}%",
        direct_mkt_shock_dollars,
        0.0,
        eff_hh_gdp_mkt,
        eff_hh_jobs_mkt,
        "Direct farm-gate revenue & supply shock",
    ),
    (
        "TFP Productivity Drift Deviation",
        f"{tfp_growth_pct - default_tfp:+.2f}% vs. {default_tfp:.2f}% base",
        yr1_tfp_dev_out_d,
        0.0,
        eff_hh_gdp_mkt,
        eff_hh_jobs_mkt,
        "KLEMS Hicks-neutral productivity shift",
    ),
    (
        "Ad-Hoc BRM / Disaster Direct Offset",
        f"+${_fmt_ofa_num(brm_subsidy_m, 0)}M payment",
        0.0,
        brm_offset_dollars,
        eff_hh_gdp_adhoc,
        eff_hh_jobs_adhoc,
        "CSNA Direct Cash Transfer to NCI (100% Margin Relief + Induced Rural Household Spending)",
    ),
]:
    if abs(f_out_d) < 1e3 and abs(f_extra_nci_d) < 1e3:
        continue
    f_nci_d = (f_out_d * on_farm_margin_coeff) + f_extra_nci_d
    f_va_gdp_d = (f_out_d * gdp_coeff) + (f_nci_d * f_hh_gdp_mult_i)
    f_proc_out_m = (f_out_d / 1e6) * proc_out_per_m_supply
    f_proc_gdp_m = (f_out_d / 1e6) * proc_gdp_per_m_supply
    f_proc_jobs = (f_out_d / 1e6) * proc_jobs_per_m_supply
    f_jobs_d = ((f_out_d / 1e6) * jobs_mult_eff_per_m) + f_proc_jobs + ((f_nci_d / 1e6) * f_hh_jobs_mult_i)
    factor_attribution_rows.append({
        "Shock Factor": f_lbl,
        "Slider Setting": f_set,
        "Yr 1 Farm Margin / NCI ($M)": f_nci_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": f_out_d / 1e6,
        "Downstream Proc Direct Output ($M)": f_proc_out_m,
        "Yr 1 Direct Output ($M)": (f_out_d / 1e6) + f_proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (f_va_gdp_d / 1e6) + f_proc_gdp_m,
        "Employment Impact (Jobs)": f_jobs_d,
        "Economic Transmission Mechanism": f_mech,
    })

if auto_brm_toggle:
    regime_short = "80% SCAP + RMP" if brm_regime == "statutory_scap_80pct" else "90% Enhanced + RMP"
    if yr1_auto_brm_cash_m > 1e-3:
        auto_brm_d = yr1_auto_brm_cash_m * 1e6
        auto_brm_gdp_d = auto_brm_d * eff_hh_gdp_stat
        auto_brm_jobs_d = yr1_auto_brm_cash_m * eff_hh_jobs_stat
        factor_attribution_rows.append({
            "Shock Factor": "Statutory BRM Auto-Stabilizer (AgriStability + ON RMP)",
            "Slider Setting": f"{regime_short} ({_fmt_ofa_num(yr1_eff_beta_brm_pct, 1)}% Eff. Payout)",
            "Yr 1 Farm Margin / NCI ($M)": yr1_auto_brm_cash_m,
            "Yr 1 Physical Ag Output ($M)": 0.0,
            "Downstream Proc Direct Output ($M)": 0.0,
            "Yr 1 Direct Output ($M)": 0.0,
            "Yr 1 Value-Added GDP ($M)": auto_brm_gdp_d / 1e6,
            "Employment Impact (Jobs)": auto_brm_jobs_d,
            "Economic Transmission Mechanism": (
                f"Yr 1 Entitlement +${_fmt_ofa_num(yr1_auto_brm_ent_m, 1)}M → +${_fmt_ofa_num(yr1_auto_brm_cash_m, 1)}M Yr 1 Cash + "
                f"+${_fmt_ofa_num(yr1_auto_brm_lag_m, 1)}M Yr 2 Settlement"
            ),
        })
    elif abs(shock_ir_bps) > 1e-4:
        factor_attribution_rows.append({
            "Shock Factor": "Statutory BRM Auto-Stabilizer (AgriStability + ON RMP)",
            "Slider Setting": f"{regime_short} (Interest Ineligible)",
            "Yr 1 Farm Margin / NCI ($M)": 0.0,
            "Yr 1 Physical Ag Output ($M)": 0.0,
            "Downstream Proc Direct Output ($M)": 0.0,
            "Yr 1 Direct Output ($M)": 0.0,
            "Yr 1 Value-Added GDP ($M)": 0.0,
            "Employment Impact (Jobs)": 0.0,
            "Economic Transmission Mechanism": (
                "Statutory Rule: Debt interest expenses are 100% non-allowable under AgriStability → $0M payout"
            ),
        })

capex_yr2_k_delta_m = 0.0
if abs(shock_capex_pct) > 1e-4:
    capex_yr2_k_delta_m = (base_capex * (shock_capex_pct / 100.0) * (ramp_schedule[0] if ramp_schedule else 1.0)) / 1e6
    factor_attribution_rows.append({
        "Shock Factor": "Capital Investment (CapEx / Machinery Surtax)",
        "Slider Setting": f"{_fmt_ofa_num(shock_capex_pct, 1, True)}% CapEx ({_fmt_ofa_num(capex_yr2_k_delta_m, 1, True)}M Yr-1 Investment)",
        "Yr 1 Farm Margin / NCI ($M)": 0.0,
        "Yr 1 Physical Ag Output ($M)": 0.0,
        "Downstream Proc Direct Output ($M)": 0.0,
        "Yr 1 Direct Output ($M)": 0.0,
        "Yr 1 Value-Added GDP ($M)": 0.0,
        "Employment Impact (Jobs)": 0.0,
        "Economic Transmission Mechanism": (
            f"Jorgenson Capital Gestation: Yr-1 CapEx shock ({_fmt_ofa_num(capex_yr2_k_delta_m, 1, True)}M) enters Years 2–{horizon} Capital Stock (ICOR={icor_cfg['icor']})"
        ),
    })


# ------------------------------------------------------------------------------
# HERO BANNER & EXECUTIVE KPI RIBBON (OFA BRAND IDENTITY 2025–26)
# ------------------------------------------------------------------------------
hero_io_scope_lbl = (
    "National SUT"
    if region == "Canada"
    else ("Within-Province Only" if io_geo_scope == "within_province" else "All-Provinces Footprint")
)
st.markdown(
    f"""
    <div class="main-header">
        <div class="ofa-kicker">Ontario Federation of Agriculture &nbsp;//&nbsp; Economic Research &amp; Policy Decision System</div>
        <h1>OFA Growth &amp; Risk Simulator</h1>
        <p>
            <strong>Jurisdiction:</strong> {region} &nbsp;|&nbsp;
            <strong>Primary Ag Sector:</strong> {shorten_naics_label(farm_type)} &nbsp;|&nbsp;
            <strong>Baselines:</strong> StatCan IO ({target_io_year}) · FCR ({fcr_year}) · Expense Profiles ({latest_exp_yr}) · Balance Sheet ({health_year})
            &nbsp;&nbsp;<span class="badge-primary-ag">NAICS 1114 / HS Ch. 06 = Primary Agriculture</span>
            &nbsp;&nbsp;<span class="badge-io-scope">IO Scope: {hero_io_scope_lbl}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

mc_yr1 = mc_summary[1]
mc_yrN = mc_summary[horizon]
dscr_yr1_delta = float(yr1_row["DSCR"]) - float(base_yr1_row["DSCR"])
lrf_yr1_delta = float(yr1_row["LRF"]) - float(base_yr1_row["LRF"])
fsf_yr1_delta = float(yr1_row["FSF"]) - float(base_yr1_row["FSF"])
yr1_proc_direct_m = float(abp_res["proc_direct_output_m"]) if include_processing else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    c1_alert = "kpi-card-alert" if combined_yr1_output_impact_m < -0.05 else ""
    c1_col = OFA_COLORS["slate_base"] if abs(combined_yr1_output_impact_m) < 0.05 else (OFA_COLORS["positive"] if combined_yr1_output_impact_m > 0 else OFA_COLORS["crimson"])
    st.markdown(
        f"""
        <div class="kpi-card {c1_alert}">
            <div class="kpi-label">Year 1 Direct Output Shock</div>
            <div class="kpi-val" style="color: {c1_col};">
                {_fmt_ofa_num(combined_yr1_output_impact_m, 1, True)} $M
            </div>
            <div class="kpi-sub">Primary: {_fmt_ofa_num(yr1_primary_delta_output/1e6, 1, True)}M | Proc Direct: {_fmt_ofa_num(yr1_proc_direct_m, 1, True)}M | Supply-Chain: {_fmt_ofa_num(yr1_supply_chain_gross_out_m, 1, True)}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    dom_gdp_or_nci = combined_yr1_gdp_impact_m if abs(combined_yr1_gdp_impact_m) >= abs(yr1_nci_delta / 1e6) else (yr1_nci_delta / 1e6)
    c2_alert = "kpi-card-alert" if dom_gdp_or_nci < -0.05 else ""
    c2_col = OFA_COLORS["slate_base"] if abs(combined_yr1_gdp_impact_m) < 0.05 and abs(yr1_nci_delta / 1e6) < 0.05 else (OFA_COLORS["positive"] if dom_gdp_or_nci > 0 else OFA_COLORS["crimson"])
    st.markdown(
        f"""
        <div class="kpi-card {c2_alert}">
            <div class="kpi-label">Year 1 GDP &amp; Margin Shock</div>
            <div class="kpi-val" style="color: {c2_col};">
                {_fmt_ofa_num(combined_yr1_gdp_impact_m, 1, True)} $M GDP
            </div>
            <div class="kpi-sub">Yr 1 Farm NCI Margin: {_fmt_ofa_num(yr1_nci_delta/1e6, 1, True)} $M | HH Induced: {_fmt_ofa_num(yr1_hh_induced_gdp_m, 1, True)}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    c3_alert = "kpi-card-alert" if combined_yr1_jobs_impact < -0.5 else ""
    c3_col = OFA_COLORS["slate_base"] if abs(combined_yr1_jobs_impact) < 0.5 else (OFA_COLORS["positive"] if combined_yr1_jobs_impact > 0 else OFA_COLORS["crimson"])
    st.markdown(
        f"""
        <div class="kpi-card {c3_alert}">
            <div class="kpi-label">Year 1 Employment Shock</div>
            <div class="kpi-val" style="color: {c3_col};">
                {combined_yr1_jobs_impact:+,.0f} Jobs
            </div>
            <div class="kpi-sub">Supply Chain + HH Induced ({yr1_hh_induced_jobs:+,.0f} Jobs, 2022$ deflated)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    c4_alert = "kpi-card-alert" if combined_yrN_gdp_impact_m < -0.05 else ""
    c4_col = OFA_COLORS["slate_base"] if abs(combined_yrN_gdp_impact_m) < 0.05 else (OFA_COLORS["positive"] if combined_yrN_gdp_impact_m > 0 else OFA_COLORS["crimson"])
    st.markdown(
        f"""
        <div class="kpi-card {c4_alert}">
            <div class="kpi-label">Year {horizon} GDP Shock (Multi-Yr)</div>
            <div class="kpi-val" style="color: {c4_col};">
                {_fmt_ofa_num(combined_yrN_gdp_impact_m, 1, True)} $M
            </div>
            <div class="kpi-sub">Primary: {_fmt_ofa_num(yrN_primary_shock_gdp_m, 1, True)}M | Proc ABP (LR): {_fmt_ofa_num(abp_res_yrN['proc_net_gdp_m'] if include_processing else 0.0, 1, True)}M | {horizon}-Yr Cum: {cum_10yr_shock_gdp_m:+,.0f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k5:
    c5_col = OFA_COLORS["positive"] if yr1_row["DSCR"] >= 1.5 else (OFA_COLORS["tulip"] if yr1_row["DSCR"] >= 1.15 else OFA_COLORS["crimson"])
    c5_alert = "kpi-card-alert" if yr1_row["DSCR"] < 1.15 else ""
    st.markdown(
        f"""
        <div class="kpi-card {c5_alert}">
            <div class="kpi-label">Debt Service Coverage (DSCR)</div>
            <div class="kpi-val" style="color: {c5_col};">
                {yr1_row['DSCR']:.2f}x ({dscr_yr1_delta:+.2f}x)
            </div>
            <div class="kpi-sub">Yr 1 Repriced ({yr1_theta*100:.0f}%): {implied_debt_rate*100:.2f}% → {yr1_row['Effective_Debt_Rate_Pct']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k6:
    distress_p = float(yr1_row.get("Distress_Prob_Pct", 8.3))
    base_distress_p = float(base_yr1_row.get("Distress_Prob_Pct", 8.3))
    distress_delta = distress_p - base_distress_p
    c6_col = OFA_COLORS["positive"] if yr1_row["LRF"] >= 1.0 else (OFA_COLORS["tulip"] if yr1_row["LRF"] >= 0.85 else OFA_COLORS["crimson"])
    c6_alert = "kpi-card-alert" if yr1_row["LRF"] < 0.85 else ""
    st.markdown(
        f"""
        <div class="kpi-card {c6_alert}">
            <div class="kpi-label">Liquidity Resilience (LRF)</div>
            <div class="kpi-val" style="color: {c6_col};">
                {yr1_row['LRF']:.2f} ({lrf_yr1_delta:+.2f})
            </div>
            <div class="kpi-sub">FSF: {yr1_row['FSF']:.2f}% ({fsf_yr1_delta:+.2f} pp) | Distress Risk: {_fmt_ofa_num(distress_p, 1)}% ({_fmt_ofa_num(distress_delta, 1, True)} pp)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Side-by-Side Scenario Comparison Panel (Scenario A Benchmark vs. Active Scenario B)
if comparison_summary is not None:
    lbl_a = comparison_summary["label_a"]
    lbl_b = comparison_summary["label_b"]
    m_a = comparison_summary["metrics_a"]
    m_b = comparison_summary["metrics_b"]
    d_ba = comparison_summary["deltas_b_minus_a"]
    with st.expander(
        f"⚖️ Side-by-Side Scenario Comparison — Scenario A [{lbl_a}] vs. Scenario B [{lbl_b}]",
        expanded=True,
    ):
        df_cmp_tbl = pd.DataFrame(comparison_summary["comparison_rows"])
        st.dataframe(df_cmp_tbl, width="stretch", hide_index=True)

        cmp_vals_all = [
            m_a["yr1_output_m"], m_b["yr1_output_m"],
            m_a["yr1_gdp_m"], m_b["yr1_gdp_m"],
            m_a["yr1_nci_m"], m_b["yr1_nci_m"],
            m_a["yrN_gdp_m"], m_b["yrN_gdp_m"],
        ]
        cmp_min, cmp_max = min(cmp_vals_all + [0.0]), max(cmp_vals_all + [0.0])
        cmp_span = max(abs(cmp_max - cmp_min), 10.0)

        cmp_metrics_order = [
            f"Year {horizon} Multi-Yr GDP ($M)",
            "Year 1 Farm Cash Margin ($M)",
            "Year 1 Value-Added GDP ($M)",
            "Year 1 Direct Output ($M)",
        ]
        vals_a_plot = [m_a["yrN_gdp_m"], m_a["yr1_nci_m"], m_a["yr1_gdp_m"], m_a["yr1_output_m"]]
        vals_b_plot = [m_b["yrN_gdp_m"], m_b["yr1_nci_m"], m_b["yr1_gdp_m"], m_b["yr1_output_m"]]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            y=cmp_metrics_order,
            x=vals_a_plot,
            name=f"Scenario A: {lbl_a}",
            orientation="h",
            marker_color=OFA_COLORS["tulip"],
            text=[f"{_fmt_ofa_num(v, 1, True)}M" for v in vals_a_plot],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_cmp.add_trace(go.Bar(
            y=cmp_metrics_order,
            x=vals_b_plot,
            name=f"Scenario B: {lbl_b}",
            orientation="h",
            marker_color=OFA_COLORS["bluewood"],
            text=[f"{_fmt_ofa_num(v, 1, True)}M" for v in vals_b_plot],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_cmp.update_layout(barmode="group", bargap=0.26)
        apply_ofa_chart_theme(
            fig_cmp,
            title=f"Scenario A vs. Scenario B Economic Impact Comparison — {region}",
            subtitle=f"Benchmark A ({lbl_a}) vs. Active Scenario B ({lbl_b}) | IO Scope: {hero_io_scope_lbl}",
            source_note="Source: Statistics Canada Tables 36-10-0595-01, 32-10-0045-01 & 32-10-0136-01; OFA PE-ABP-IO Modeling.",
            height=390,
            is_horizontal_bar=True,
        )
        fig_cmp.update_xaxes(
            title_text="Shock Impact vs. Unshocked Baseline ($M)",
            range=[cmp_min - 0.24 * cmp_span, cmp_max + 0.24 * cmp_span],
        )
        fig_cmp.update_yaxes(title_text="")
        st.plotly_chart(fig_cmp, width="stretch", config=ofa_plotly_config(f"OFA_Scenario_Comparison_{region}"))

        cmp_dl_c1, cmp_dl_c2 = st.columns([1.0, 2.2])
        with cmp_dl_c1:
            st.download_button(
                "📥 Download Side-by-Side Comparison (CSV)",
                data=df_cmp_tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"OFA_Scenario_Comparison_{region}.csv",
                mime="text/csv",
                key="btn_dl_cmp_csv",
            )
        with cmp_dl_c2:
            st.caption(
                f"**Net Policy / Scenario Delta (B − A):** Year 1 Farm Cash Margin **`{_fmt_ofa_num(d_ba['yr1_nci_m'], 1, True)} $M`** | "
                f"Year 1 DSCR **`{d_ba['yr1_dscr']:+.2f}x`** | Year 1 Distress Risk **`{_fmt_ofa_num(d_ba['yr1_distress_pct'], 2, True)} pp`** | "
                f"{horizon}-Yr Cumulative Primary GDP **`{d_ba['cum_10yr_gdp_m']:+,.0f} $M`**."
            )

# Active Shock Factor Attribution Panel (Full-Width Table + Full-Width Presentation-Ready Chart)
if factor_attribution_rows:
    with st.expander(f"🎯 Active Shock Factor Attribution — Exact Margin vs. Value-Added GDP vs. Direct Output Split ({len(factor_attribution_rows)} Active Channel{'s' if len(factor_attribution_rows) != 1 else ''})", expanded=True):
        df_attr_chart = pd.DataFrame(factor_attribution_rows)
        df_attr_table = df_attr_chart.copy()
        if len(factor_attribution_rows) > 1:
            tot_row = pd.DataFrame([{
                "Shock Factor": "Total Combined Active Shocks (Reconciled to KPI Banner)",
                "Slider Setting": f"{len(factor_attribution_rows)} active channels",
                "Yr 1 Farm Margin / NCI ($M)": float(df_attr_chart["Yr 1 Farm Margin / NCI ($M)"].sum()),
                "Yr 1 Physical Ag Output ($M)": float(df_attr_chart["Yr 1 Physical Ag Output ($M)"].sum()),
                "Downstream Proc Direct Output ($M)": float(df_attr_chart["Downstream Proc Direct Output ($M)"].sum()),
                "Yr 1 Direct Output ($M)": float(df_attr_chart["Yr 1 Direct Output ($M)"].sum()),
                "Yr 1 Value-Added GDP ($M)": float(df_attr_chart["Yr 1 Value-Added GDP ($M)"].sum()),
                "Employment Impact (Jobs)": float(df_attr_chart["Employment Impact (Jobs)"].sum()),
                "Economic Transmission Mechanism": (
                    f"Combined Direct Output = {_fmt_ofa_num(float(df_attr_chart['Yr 1 Direct Output ($M)'].sum()), 1, True)} $M"
                ),
            }])
            df_attr_table = pd.concat([df_attr_chart, tot_row], ignore_index=True)

        st.dataframe(
            df_attr_table.style.format({
                "Yr 1 Farm Margin / NCI ($M)": "{:+,.2f} $M",
                "Yr 1 Physical Ag Output ($M)": "{:+,.2f} $M",
                "Downstream Proc Direct Output ($M)": "{:+,.2f} $M",
                "Yr 1 Direct Output ($M)": "{:+,.2f} $M",
                "Yr 1 Value-Added GDP ($M)": "{:+,.2f} $M",
                "Employment Impact (Jobs)": "{:+,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

        # Filter out pure multi-year gestation rows ($0 across all three Year-1 bars) so chart never renders empty rows
        yr1_nonzero_mask = (
            df_attr_chart["Yr 1 Farm Margin / NCI ($M)"].abs()
            + df_attr_chart["Yr 1 Value-Added GDP ($M)"].abs()
            + df_attr_chart["Yr 1 Direct Output ($M)"].abs()
        ) > 0.01
        df_attr_plot = df_attr_chart[yr1_nonzero_mask].copy()

        if not df_attr_plot.empty:
            # Sort so largest absolute shock is at the top of horizontal grouped chart
            df_attr_plot["_abs_max"] = df_attr_plot[
                ["Yr 1 Farm Margin / NCI ($M)", "Yr 1 Value-Added GDP ($M)", "Yr 1 Direct Output ($M)"]
            ].abs().max(axis=1)
            df_attr_plot = df_attr_plot.sort_values("_abs_max", ascending=True)

            attr_vals = (
                df_attr_plot["Yr 1 Farm Margin / NCI ($M)"].tolist()
                + df_attr_plot["Yr 1 Value-Added GDP ($M)"].tolist()
                + df_attr_plot["Yr 1 Direct Output ($M)"].tolist()
            )
            a_min, a_max = min(attr_vals + [0.0]), max(attr_vals + [0.0])
            a_span = max(abs(a_max - a_min), 10.0)

            capex_note = (
                f" | Note: Yr-1 CapEx shock ({_fmt_ofa_num(capex_yr2_k_delta_m, 1, True)} $M) enters Years 2–{horizon} productive capital stock."
                if abs(shock_capex_pct) > 1e-4
                else ""
            )
            fig_attr = go.Figure()
            for col_name, bar_lbl, bar_color in [
                ("Yr 1 Direct Output ($M)", "Year 1 Direct Output ($M)", OFA_COLORS["bluewood"]),
                ("Yr 1 Value-Added GDP ($M)", "Year 1 Value-Added GDP ($M)", OFA_COLORS["tulip"]),
                ("Yr 1 Farm Margin / NCI ($M)", "Year 1 Farm Cash Margin / NCI ($M)", OFA_COLORS["crimson"]),
            ]:
                series_vals = df_attr_plot[col_name].tolist()
                fig_attr.add_trace(go.Bar(
                    y=df_attr_plot["Shock Factor"].tolist(),
                    x=series_vals,
                    name=bar_lbl,
                    orientation="h",
                    marker_color=bar_color,
                    text=[f"{_fmt_ofa_num(v, 1, True)}M" if abs(v) >= 0.05 else "" for v in series_vals],
                    textposition="outside",
                    cliponaxis=False,
                ))
            fig_attr.update_layout(barmode="group", bargap=0.24, bargroupgap=0.08)
            apply_ofa_chart_theme(
                fig_attr,
                title=f"Year 1 Shock Attribution: Farm Cash Margin vs. Value-Added GDP vs. Direct Output ({region})",
                subtitle=f"Sector: {shorten_naics_label(farm_type)} | IO Scope: {hero_io_scope_lbl}{capex_note}",
                source_note="Source: Statistics Canada Tables 36-10-0595-01, 32-10-0045-01 & 32-10-0136-01; OFA PE-ABP-IO Modeling.",
                height=max(360, 88 * len(df_attr_plot) + 175),
                is_horizontal_bar=True,
            )
            fig_attr.update_xaxes(
                title_text="Year 1 Economic Impact vs. Baseline ($M)",
                range=[a_min - 0.25 * a_span, a_max + 0.25 * a_span],
            )
            fig_attr.update_yaxes(title_text="")
            st.plotly_chart(fig_attr, width="stretch", config=ofa_plotly_config(f"OFA_Year1_Factor_Attribution_{region}"))

        st.caption(
            f"**Accounting Note:** Debt interest shocks reprice across the 5-year maturity ladder (`{yr1_theta*100:.0f}%` in Year 1 → `100%` by Year 5), compressing **Farm Net Cash Income (`NCI`)** and **Multi-Year CapEx** while leaving Year-1 physical farm output virtually intact. Intermediate input spikes compress **Year-1 Value-Added GDP & Margins (`{_fmt_ofa_num(hotelling_prof['yr1_margin_share']*100, 1)}%`)** while reducing Year-1 physical output by **`{_fmt_ofa_num(hotelling_prof['yr1_output_share']*100, 1)}%`**."
        )
else:
    st.info(
        f"**No Active Shock Sliders (`0 $M` Shock Impact):** All input cost, interest rate, tariff, CapEx, and BRM sliders are currently at `0`. "
        f"Move any slider in the left sidebar or choose a Quick-Load Scenario to inspect isolated Farm Cash Margin (`NCI`), Value-Added GDP, Direct Output, and Multi-Year CapEx impacts (baseline trend growth of `+{_fmt_ofa_num(default_tfp, 2)}%/yr` TFP = `+{_fmt_ofa_num(yr1_trend_primary_output_m, 1)} $M` Year 1 Primary Output is held as the reference baseline)."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# FIVE INTEGRATED ANALYTICAL TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Macro, Value-Chain (ABP) & Fiscal Impact",
    "📈 2. 10-Year Dynamic Trajectory & Monte Carlo Risk",
    "🛡️ 3. 2026 Section 338 Tariff & Counter-Tariff Squeeze",
    "🚜 4. Farm-Gate ('Average Per Farm' & 26 Pub 60 / CEA Budgets)",
    "📚 5. Econometric Calibration Matrices & Executive Briefings",
])


# ==============================================================================
# TAB 1: MACRO, VALUE-CHAIN (ABP) & FISCAL IMPACT
# ==============================================================================
with tab1:
    st.subheader(f"Macroeconomic, Value-Chain Analysis-by-Parts (ABP) & Fiscal Impact — {region}")
    st.caption(
        f"🌐 **Active StatCan `36-10-0595-01` IO Geographical Scope:** **{hero_io_scope_lbl}** | "
        f"Primary Output Multiplier = **`{out_mult:.3f}`** | Value-Added GDP Multiplier = **`{gdp_coeff:.3f}`** | "
        f"Employment = **`{jobs_mult_per_m:.2f}` Jobs / \\$1M (`2022$`)** (`{jobs_mult_eff_per_m:.2f}` Jobs / \\$1M nominal) | "
        f"Own-Border GDP Retention = **`{_fmt_ofa_num(gdp_retention_pct, 1)}%`** of All-Provinces footprint."
    )

    col_left, col_right = st.columns([1.22, 1.0])

    with col_left:
        st.markdown("#### Mutually Exclusive NAICS Sub-Sector Input Cost Breakdown")
        tier_mode = st.radio(
            "NAICS Hierarchy Tier",
            ["leaf", "parent"],
            format_func=lambda x: "4-/5-/6-Digit Leaf Sub-Sectors (Mutually Exclusive — No Double Counting)"
            if x == "leaf"
            else "3-Digit Macro Aggregates (Crop Production [111] vs. Animal Production [112])",
            horizontal=True,
        )
        cost_breakdown_list = compute_cost_commodity_breakdown(
            expense_profiles,
            shock_fuel=shock_fuel,
            shock_fert=shock_fert,
            shock_chem=shock_chem,
            shock_seed=shock_seed,
            shock_feed=shock_feed,
            shock_wages=shock_wages,
            shock_interest_rel_pct=shock_interest_rel_pct,
            tier=tier_mode,
            region=exp_geo,
            apply_yr1_repricing_lag=True,
        )
        df_cost_bd = pd.DataFrame(cost_breakdown_list)
        if not df_cost_bd.empty and df_cost_bd["Shock ($M)"].abs().sum() > 0:
            df_cost_plot = df_cost_bd.copy()
            df_cost_plot["Subsector"] = df_cost_plot["Commodity"].map(shorten_naics_label)
            df_cost_plot = df_cost_plot.sort_values("Shock ($M)", ascending=True)

            cost_cols_cfg = [
                ("Fuel ($M)", "Fuel & Energy", OFA_COLORS["crimson"]),
                ("Fertilizer ($M)", "Fertilizer & Lime", OFA_COLORS["bluewood"]),
                ("Pesticides ($M)", "Pesticides & IPM", OFA_COLORS["tulip"]),
                ("Seed & Plants ($M)", "Seed & Plants", OFA_COLORS["positive"]),
                ("Feed ($M)", "Commercial Feed", "#5B7C8D"),
                ("Labour ($M)", "Hired Labour", OFA_COLORS["valencia"]),
                ("Interest ($M)", "Repriced Debt Interest", "#94A3B8"),
            ]
            fig_cost = go.Figure()
            for col_c, leg_lbl, col_hex in cost_cols_cfg:
                if col_c in df_cost_plot.columns and df_cost_plot[col_c].abs().sum() > 0.005:
                    fig_cost.add_trace(go.Bar(
                        y=df_cost_plot["Subsector"].tolist(),
                        x=df_cost_plot[col_c].tolist(),
                        name=leg_lbl,
                        orientation="h",
                        marker_color=col_hex,
                    ))

            pos_totals = df_cost_plot[[c for c, _, _ in cost_cols_cfg if c in df_cost_plot.columns]].clip(lower=0).sum(axis=1)
            neg_totals = df_cost_plot[[c for c, _, _ in cost_cols_cfg if c in df_cost_plot.columns]].clip(upper=0).sum(axis=1)
            c_min = float(min(neg_totals.min(), 0.0))
            c_max = float(max(pos_totals.max(), 0.0))
            c_span = max(abs(c_max - c_min), 5.0)

            net_totals = df_cost_plot["Shock ($M)"].tolist()
            label_x = [
                float(pos_totals.iloc[i]) if net_totals[i] >= 0 else float(neg_totals.iloc[i])
                for i in range(len(net_totals))
            ]
            fig_cost.add_trace(go.Scatter(
                y=df_cost_plot["Subsector"].tolist(),
                x=label_x,
                mode="text",
                text=[f"  {_fmt_ofa_num(v, 1, True)}M  " if abs(v) >= 0.1 else "" for v in net_totals],
                textposition=["middle right" if v >= 0 else "middle left" for v in net_totals],
                textfont=dict(family="Arial, Roboto, sans-serif", size=11, color=OFA_COLORS["bluewood"]),
                showlegend=False,
                cliponaxis=False,
                hoverinfo="skip",
            ))

            fig_cost.update_layout(barmode="relative", bargap=0.22)
            apply_ofa_chart_theme(
                fig_cost,
                title=f"Year 1 Direct Cost Shock by NAICS Sub-Sector — {region}",
                subtitle="Operating Input & Year-1 Repriced Debt Carrying Cost Shocks ($M)",
                source_note="Source: Statistics Canada Table 32-10-0136-01 & FCC Debt Maturity Profiles; OFA Modeling.",
                height=max(440, 34 * len(df_cost_plot) + 155),
                top_margin=124,
                is_horizontal_bar=True,
            )
            fig_cost.update_xaxes(
                title_text="Year 1 Direct Cost Change ($M)",
                range=[c_min - 0.22 * c_span, c_max + 0.22 * c_span],
            )
            fig_cost.update_yaxes(title_text="")
            st.plotly_chart(fig_cost, width="stretch", config=ofa_plotly_config(f"OFA_Subsector_Cost_Breakdown_{region}"))
        else:
            st.info("Adjust any Input Cost or Debt Interest Rate slider in the sidebar to inspect sub-sector cost exposure.")
            if not df_cost_bd.empty:
                df_cost_disp = df_cost_bd[["Commodity", "Total Revenue ($M)", "Total Expenses ($M)"]].copy()
                df_cost_disp["Commodity"] = df_cost_disp["Commodity"].map(shorten_naics_label)
                st.dataframe(df_cost_disp, width="stretch", hide_index=True)

    with col_right:
        st.markdown("#### Primary Agriculture Output Disposition & Downstream ABP Netting")
        disp = abp_res["disposition_shares"]
        disp_labels = [
            "Domestic Processors (NAICS 311/312)",
            "Direct Primary Exports & Inter-Provincial",
            "Direct Fresh Household & Foodservice",
            "Intra-Ag Feed, Seed & Landscaping",
        ]
        disp_vals = [
            disp["s_proc"] * 100.0,
            disp["s_exp"] * 100.0,
            disp["s_fresh"] * 100.0,
            disp["s_intra_ag"] * 100.0,
        ]
        disp_colors = [
            OFA_COLORS["bluewood"],
            OFA_COLORS["crimson"],
            OFA_COLORS["tulip"],
            "#8FA3B0",
        ]
        fig_disp = go.Figure(data=[go.Pie(
            labels=disp_labels,
            values=disp_vals,
            hole=0.52,
            sort=False,
            marker=dict(colors=disp_colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(family="Arial, Roboto, sans-serif", size=12, color="#FFFFFF"),
            hovertemplate="<b>%{label}</b><br>Share of Farm Output: %{value:.1f}%<extra></extra>",
        )])
        apply_ofa_chart_theme(
            fig_disp,
            title=f"Primary Farm Output Disposition — {region}",
            subtitle=f"Sector: {shorten_naics_label(farm_type)} (StatCan SUT 36-10-0478-01)",
            source_note="Source: Statistics Canada Supply and Use Table 36-10-0478-01; OFA Calibration.",
            height=400,
            top_margin=126,
            legend_y=1.02,
        )
        st.plotly_chart(fig_disp, width="stretch", config=ofa_plotly_config(f"OFA_Output_Disposition_{region}"))

        st.markdown(
            f"""
            **Analysis-by-Parts (ABP) Non-Double-Counting Summary ({'ACTIVE' if include_processing else 'INACTIVE'}):**
            - **Domestic Supply Shock Forwarded to Processors (`{_fmt_ofa_num(abp_res['s_proc']*100, 1)}%` SUT Share):** `${_fmt_ofa_num(abp_res['farm_to_proc_flow_m'], 1, True)}M`
            - **Direct Processor Output Contraction (`NAICS 311/312`):** `${_fmt_ofa_num(abp_res['proc_direct_output_m'], 1, True)}M`
            - **ABP Netted Downstream GDP Impact (`NAICS 311/312`):** `${_fmt_ofa_num(abp_res['proc_net_gdp_m'], 1, True)}M` (`{abp_res['proc_net_jobs']:+,.0f}` Jobs)
            - **Upstream Farm GDP Netted Out (Zero Double-Counting):** `${_fmt_ofa_num(abp_res['double_counted_farm_gdp_removed_m'], 1, True)}M`
            """
        )

    st.markdown("#### Downstream & Co-Targeted Food & Beverage Manufacturing (`NAICS 311/312`) ABP Table")
    df_abp_sub = pd.DataFrame(abp_res["subsectors"])
    if not df_abp_sub.empty:
        st.dataframe(
            df_abp_sub.style.format({
                "Allocation Share (%)": "{:.1f}%",
                "a_farm_dom": "{:.4f}",
                "Gross Output Mult": "{:.3f}",
                "ABP Net Output Mult": "{:.3f}",
                "Gross GDP Mult": "{:.3f}",
                "ABP Net GDP Mult": "{:.3f}",
                "ABP Net Jobs (/ $1M)": "{:.2f}",
                "Direct Processor Shock ($M)": "{:+,.2f}",
                "ABP Net Output Impact ($M)": "{:+,.2f}",
                "ABP Net GDP Impact ($M)": "{:+,.2f}",
                "ABP Net Jobs": "{:+,.0f}",
                "ABP Net Wages ($M)": "{:+,.2f}",
                "Upstream Farm GDP Netted Out ($M)": "{:+,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

    # Sister Engine 1B & 5: Fiscal Tax Impact + BRM Subsidy Impact + OAG Horticulture Value Chain
    st.markdown("---")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("#### 🏛️ Government Fiscal Tax Revenue Impact vs. BRM Stabilization Cost")
        fiscal_imp = compute_simulator_fiscal_summary(
            gdp_impact=combined_yr1_gdp_impact_m * 1e6,
            wages_impact=combined_yr1_wages_impact_m * 1e6,
            output_impact=combined_yr1_output_impact_m * 1e6,
            stage="growing",
        )
        fisc_df = pd.DataFrame([
            {"Order of Government": "Federal Government (CIT, PIT, GST)", "Fiscal Revenue Impact ($M)": fiscal_imp["federal"] / 1e6},
            {"Order of Government": "Provincial Government (CIT, PIT, PST/HST)", "Fiscal Revenue Impact ($M)": fiscal_imp["provincial"] / 1e6},
            {"Order of Government": "Municipal Governments (Property & Local Fees)", "Fiscal Revenue Impact ($M)": fiscal_imp["municipal"] / 1e6},
            {"Order of Government": "Total Consolidated Tax Impact", "Fiscal Revenue Impact ($M)": fiscal_imp["total"] / 1e6},
        ])
        st.dataframe(fisc_df.style.format({"Fiscal Revenue Impact ($M)": "{:+,.2f} $M"}), width="stretch", hide_index=True)
        if total_yr1_brm_cash_m > 0:
            sub_imp = compute_simulator_subsidy_summary(
                total_yr1_brm_cash_m * 1e6,
                region=region,
                year=target_io_year,
                io_df=io_df,
                geo_scope=io_geo_scope,
            )
            st.success(
                f"**Active BRM Cash Stabilization (`+${_fmt_ofa_num(total_yr1_brm_cash_m, 1)}M` Yr 1 Cash Inflow = "
                f"`+${_fmt_ofa_num(yr1_auto_brm_cash_m, 1)}M` Statutory + `+${_fmt_ofa_num(brm_subsidy_m, 1)}M` Ad-Hoc; "
                f"`+${_fmt_ofa_num(yr1_auto_brm_lag_m, 1)}M` Yr 2 Tax-Filing Settlement Carryover):** "
                f"Generates **`${_fmt_ofa_num(sub_imp['gdp']/1e6, 1, True)}M`** in induced/indirect GDP and preserves **`{sub_imp['jobs']:+,.0f}`** jobs across rural {region} supply chains."
            )
        elif auto_brm_toggle and abs(shock_ir_bps) > 1e-4:
            st.info(
                "**Statutory BRM Auto-Stabilizer Active (`$0M` Entitlement):** Under the Sustainable CAP AgriStability Handbook, "
                "farm debt interest expenses are 100% non-allowable. Isolated interest rate shocks do not trigger statutory BRM payouts."
            )

    with f_col2:
        st.markdown("#### 🍎 Sister Engine 5: OAG Horticulture Multi-Stage Value-Chain (HLRF)")
        try:
            hlrf_rows = cached_hlrf_summary(
                region_name=region, target_yr=target_io_year, geo_scope_val=io_geo_scope
            )
            if hlrf_rows:
                st.dataframe(
                    pd.DataFrame(hlrf_rows).style.format({
                        "Direct Spend ($M)": "${:,.2f}M",
                        "Output Impact ($M)": "${:,.2f}M",
                        "GDP Impact ($M)": "${:,.2f}M",
                        "Jobs (FTEs)": "{:,.0f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
        except Exception as exc:
            st.caption(f"OAG Horticulture HLRF module note: {exc}")


# ==============================================================================
# TAB 2: 10-YEAR DYNAMIC TRAJECTORY & MONTE CARLO RISK
# ==============================================================================
with tab2:
    st.subheader(f"10-Year Dynamic Stock-Flow Trajectory & {n_sims}-Run Correlated Monte Carlo Fan Charts")
    mc_info = get_mc_covariance_params(region=region, basis=mc_basis)
    st.caption(
        f"**Monte Carlo Calibration Source:** {mc_info['source']} | "
        f"Mean TFP Drift $\\mu_{{\\text{{TFP}}}} = {mc_info['mu_tfp']:.2f}\\%$ | "
        f"Annual TFP Volatility $\\sigma_{{\\text{{TFP}}}} = {mc_info['sigma_tfp']:.2f}\\%$ | "
        f"Baseline StatCan IO Year: **{target_io_year}**"
    )

    years_x = df_dyn["Year"].tolist()
    t_vals = df_dyn["t"].tolist()
    tfp_lbl = _fmt_ofa_num(default_tfp, 2)

    c_fan1, c_fan2 = st.columns(2)

    with c_fan1:
        # 1. Gross Output Fan Chart
        out_base = [v / 1e9 for v in df_base_dyn["Gross_Output"]]
        out_det = [v / 1e9 for v in df_dyn["Gross_Output"]]
        out_p10 = [mc_summary[t]["Gross_Output_p10"] / 1e9 for t in t_vals]
        out_p90 = [mc_summary[t]["Gross_Output_p90"] / 1e9 for t in t_vals]

        fig_out = go.Figure()
        fig_out.add_trace(go.Scatter(
            x=years_x + years_x[::-1],
            y=out_p90 + out_p10[::-1],
            fill="toself",
            fillcolor="rgba(51, 75, 88, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Monte Carlo Band",
        ))
        fig_out.add_trace(go.Scatter(
            x=years_x,
            y=out_base,
            mode="lines",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
            name=f"Unshocked Baseline (+{tfp_lbl}% TFP)",
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_out.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Gross_Output"]],
                mode="lines+markers",
                line=dict(color=OFA_COLORS["tulip"], dash="dashdot", width=2.6),
                name=f"Scenario A: {comparison_summary['label_a']}",
            ))
        fig_out.add_trace(go.Scatter(
            x=years_x,
            y=out_det,
            mode="lines+markers",
            line=dict(color=OFA_COLORS["bluewood"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["bluewood"]),
            name="Active Scenario B ($B)",
        ))
        apply_ofa_chart_theme(
            fig_out,
            title=f"Primary Farm Gross Output Trajectory — {region}",
            subtitle=f"Deterministic Path & {n_sims}-Run P10–P90 Confidence Band ($B)",
            source_note="Source: Statistics Canada Tables 32-10-0045-01 & 36-10-0208-01; OFA Dynamic Stock-Flow Engine.",
            height=430,
            top_margin=120,
        )
        fig_out.update_yaxes(title_text="Gross Output ($B)")
        fig_out.update_xaxes(title_text="Projection Year", dtick=1)
        st.plotly_chart(fig_out, width="stretch", config=ofa_plotly_config(f"OFA_Gross_Output_Trajectory_{region}"))

    with c_fan2:
        # 2. Direct Value-Added GDP Fan Chart
        gdp_base = [v / 1e9 for v in df_base_dyn["Direct_GDP"]]
        gdp_det = [v / 1e9 for v in df_dyn["Direct_GDP"]]
        gdp_p10 = [mc_summary[t]["Direct_GDP_p10"] / 1e9 for t in t_vals]
        gdp_p90 = [mc_summary[t]["Direct_GDP_p90"] / 1e9 for t in t_vals]

        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(
            x=years_x + years_x[::-1],
            y=gdp_p90 + gdp_p10[::-1],
            fill="toself",
            fillcolor="rgba(215, 24, 42, 0.11)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Monte Carlo Band",
        ))
        fig_gdp.add_trace(go.Scatter(
            x=years_x,
            y=gdp_base,
            mode="lines",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
            name=f"Unshocked Baseline (+{tfp_lbl}% TFP)",
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            if scen_a_cfg is not None and str(scen_a_cfg.get("geo_scope", io_geo_scope)) != io_geo_scope:
                fig_gdp.add_trace(go.Scatter(
                    x=years_x,
                    y=[v / 1e9 for v in df_base_dyn_a["Direct_GDP"]],
                    mode="lines",
                    line=dict(color="#94A3B8", dash="dot", width=1.8),
                    name=f"Unshocked Baseline A ({comparison_summary['label_a']})",
                ))
            fig_gdp.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Direct_GDP"]],
                mode="lines+markers",
                line=dict(color=OFA_COLORS["tulip"], dash="dashdot", width=2.6),
                name=f"Scenario A: {comparison_summary['label_a']}",
            ))
        fig_gdp.add_trace(go.Scatter(
            x=years_x,
            y=gdp_det,
            mode="lines+markers",
            line=dict(color=OFA_COLORS["crimson"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["crimson"]),
            name="Active Scenario B GDP ($B)",
        ))
        apply_ofa_chart_theme(
            fig_gdp,
            title=f"Value-Added GDP Trajectory — {region}",
            subtitle=f"IO Scope: {hero_io_scope_lbl} | P10–P90 Stochastic Band ($B)",
            source_note="Source: Statistics Canada Tables 36-10-0595-01 & 36-10-0208-01; OFA Dynamic Stock-Flow Engine.",
            height=430,
            top_margin=120,
        )
        fig_gdp.update_yaxes(title_text="Value-Added GDP ($B)")
        fig_gdp.update_xaxes(title_text="Projection Year", dtick=1)
        st.plotly_chart(fig_gdp, width="stretch", config=ofa_plotly_config(f"OFA_Value_Added_GDP_Trajectory_{region}"))

    c_fan3, c_fan4 = st.columns(2)
    with c_fan3:
        # 3. DSCR & LRF Solvency Trajectory
        fig_fin = go.Figure()
        fig_fin.add_trace(go.Scatter(
            x=years_x,
            y=df_base_dyn["DSCR"],
            mode="lines",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
            name="Unshocked Baseline DSCR (x)",
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_fin.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["DSCR"],
                mode="lines+markers",
                line=dict(color=OFA_COLORS["tulip"], dash="dashdot", width=2.6),
                name=f"Scenario A DSCR: {comparison_summary['label_a']}",
            ))
        fig_fin.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["DSCR"],
            mode="lines+markers",
            line=dict(color=OFA_COLORS["bluewood"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["bluewood"]),
            name="Scenario B DSCR (x)",
        ))
        fig_fin.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["LRF"],
            mode="lines+markers",
            line=dict(color=OFA_COLORS["crimson"], width=2.8),
            marker=dict(size=6, symbol="diamond", color=OFA_COLORS["crimson"]),
            name="Scenario B Liquidity Resilience (LRF)",
        ))
        fig_fin.add_hline(
            y=1.15,
            line_dash="dot",
            line_color=OFA_COLORS["valencia"],
            line_width=1.8,
            annotation_text="1.15x Credit Covenant",
            annotation_position="top right",
            annotation_font=dict(family="Arial, Roboto, sans-serif", size=11, color=OFA_COLORS["crimson"]),
        )
        apply_ofa_chart_theme(
            fig_fin,
            title=f"Solvency & Liquidity Trajectory — {region}",
            subtitle="Debt Service Coverage Ratio (DSCR) & Liquidity Resilience Factor (LRF)",
            source_note="Source: Statistics Canada Tables 32-10-0052-01 & 32-10-0102-01; FFSC Solvency Benchmarks.",
            height=420,
            top_margin=120,
        )
        fig_fin.update_yaxes(title_text="Coverage Ratio / Resilience Index")
        fig_fin.update_xaxes(title_text="Projection Year", dtick=1)
        st.plotly_chart(fig_fin, width="stretch", config=ofa_plotly_config(f"OFA_Solvency_Liquidity_Trajectory_{region}"))

    with c_fan4:
        # 4. Productive Capital Stock & Net Cash Income (Dual-Axis)
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(
            x=years_x,
            y=[v / 1e9 for v in df_dyn["Capital_Stock"]],
            name="Productive Capital Stock ($B)",
            marker=dict(color=OFA_COLORS["champagne"], line=dict(color="#CBD5E1", width=1)),
        ))
        fig_cap.add_trace(go.Scatter(
            x=years_x,
            y=[v / 1e9 for v in df_base_dyn["Net_Cash_Income"]],
            mode="lines",
            name="Unshocked Baseline NCI ($B)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_cap.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Net_Cash_Income"]],
                mode="lines+markers",
                name=f"Scenario A NCI: {comparison_summary['label_a']}",
                yaxis="y2",
                line=dict(color=OFA_COLORS["tulip"], dash="dashdot", width=2.6),
            ))
        fig_cap.add_trace(go.Scatter(
            x=years_x,
            y=[v / 1e9 for v in df_dyn["Net_Cash_Income"]],
            mode="lines+markers",
            name="Scenario B Net Cash Income ($B)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["crimson"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["crimson"]),
        ))
        apply_ofa_chart_theme(
            fig_cap,
            title=f"Productive Capital Stock & Net Cash Income — {region}",
            subtitle=f"Capital Stock Accumulation ($B, Left) vs. Farm Net Cash Income ($B, Right | ICOR={icor_cfg['icor']})",
            source_note="Source: Statistics Canada Tables 36-10-0096-01 & 32-10-0052-01; OFA Modeling.",
            height=420,
            top_margin=120,
            right_margin=76,
        )
        fig_cap.update_yaxes(title_text="Productive Capital Stock ($B)")
        fig_cap.update_xaxes(title_text="Projection Year", dtick=1)
        fig_cap.update_layout(
            yaxis2=dict(
                title=dict(text="Net Cash Income ($B)", font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["crimson"])),
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickfont=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["crimson"]),
            )
        )
        st.plotly_chart(fig_cap, width="stretch", config=ofa_plotly_config(f"OFA_Capital_Stock_NCI_Trajectory_{region}"))

    c_fan5, c_fan6 = st.columns(2)
    with c_fan5:
        # 5. 5-Year Contractual Debt Rollover Staircase & Effective Portfolio Rate (%)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Bar(
            x=years_x,
            y=df_dyn["Debt_Repricing_Share_Pct"],
            name="Cumulative Debt Repriced (%)",
            marker=dict(color="rgba(51, 75, 88, 0.20)", line=dict(color=OFA_COLORS["bluewood"], width=1)),
        ))
        fig_roll.add_trace(go.Scatter(
            x=years_x,
            y=df_base_dyn["Effective_Debt_Rate_Pct"],
            mode="lines",
            name="Baseline Effective Rate (%)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_roll.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["Effective_Debt_Rate_Pct"],
                mode="lines+markers",
                name=f"Scenario A Rate: {comparison_summary['label_a']}",
                yaxis="y2",
                line=dict(color=OFA_COLORS["tulip"], dash="dashdot", width=2.6),
            ))
        fig_roll.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["Effective_Debt_Rate_Pct"],
            mode="lines+markers",
            name="Scenario B Effective Rate (%)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["crimson"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["crimson"]),
        ))
        apply_ofa_chart_theme(
            fig_roll,
            title="5-Year Contractual Debt Rollover & Effective Interest Rate",
            subtitle="Share of Farm Debt Portfolio Repriced (%, Left) vs. Effective Interest Rate (%, Right)",
            source_note="Source: Statistics Canada Tables 32-10-0051-01 & 32-10-0052-01; FCC Term Maturity Ladder.",
            height=410,
            top_margin=120,
            right_margin=76,
        )
        fig_roll.update_yaxes(title_text="Cumulative Debt Repriced (%)", range=[0, 112])
        fig_roll.update_xaxes(title_text="Projection Year", dtick=1)
        fig_roll.update_layout(
            yaxis2=dict(
                title=dict(text="Effective Interest Rate (%)", font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["crimson"])),
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickfont=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["crimson"]),
            )
        )
        st.plotly_chart(fig_roll, width="stretch", config=ofa_plotly_config(f"OFA_Debt_Rollover_Trajectory_{region}"))

    with c_fan6:
        # 6. Endogenous CapEx Shift (%) & Financial Distress Probability (%)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=years_x,
            y=df_dyn["CapEx_CrowdOut_Pct"],
            name="Endogenous CapEx Shift (%)",
            marker=dict(color=OFA_COLORS["tulip"], line=dict(color="#D99226", width=1)),
        ))
        fig_dist.add_trace(go.Scatter(
            x=years_x,
            y=df_base_dyn["Distress_Prob_Pct"],
            mode="lines",
            name="Baseline Distress Risk (%)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["slate_base"], dash="dash", width=2.2),
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_dist.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["Distress_Prob_Pct"],
                mode="lines+markers",
                name=f"Scenario A Distress: {comparison_summary['label_a']}",
                yaxis="y2",
                line=dict(color=OFA_COLORS["bluewood"], dash="dashdot", width=2.6),
            ))
        fig_dist.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["Distress_Prob_Pct"],
            mode="lines+markers",
            name="Scenario B Distress Risk (%)",
            yaxis="y2",
            line=dict(color=OFA_COLORS["crimson"], width=3.2),
            marker=dict(size=6.5, color=OFA_COLORS["crimson"]),
        ))
        apply_ofa_chart_theme(
            fig_dist,
            title="Endogenous Farm CapEx Shift & Financial Distress Risk",
            subtitle="Jorgenson / Pecking-Order CapEx Shift (%, Left) vs. Solvency Arrears Risk (%, Right)",
            source_note="Source: Statistics Canada Table 32-10-0102-01, FCC Credit Arrears & Jorgenson User-Cost Model.",
            height=410,
            top_margin=120,
            right_margin=76,
        )
        fig_dist.update_yaxes(title_text="Endogenous CapEx Shift (%)")
        fig_dist.update_xaxes(title_text="Projection Year", dtick=1)
        fig_dist.update_layout(
            yaxis2=dict(
                title=dict(text="Financial Distress Risk (%)", font=dict(family="Arial, Roboto, sans-serif", size=12.5, color=OFA_COLORS["crimson"])),
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                tickfont=dict(family="Arial, Roboto, sans-serif", size=12, color=OFA_COLORS["crimson"]),
            )
        )
        st.plotly_chart(fig_dist, width="stretch", config=ofa_plotly_config(f"OFA_CapEx_Distress_Trajectory_{region}"))

    # Complete Year-by-Year Trajectory Table
    st.markdown("#### Complete Year-by-Year Dynamic Trajectory Table (with Isolated Shock vs. Baseline & P10–P90 Bands)")
    table_rows = []
    for idx_r, r in df_dyn.iterrows():
        t_i = int(r["t"])
        mc_t = mc_summary[t_i]
        b_r = df_base_dyn.iloc[idx_r]
        isolated_gdp_shock_m = (float(r["Direct_GDP"]) - float(b_r["Direct_GDP"])) / 1e6
        isolated_out_shock_m = (float(r["Gross_Output"]) - float(b_r["Gross_Output"])) / 1e6
        isolated_nci_shock_m = (float(r["Net_Cash_Income"]) - float(b_r["Net_Cash_Income"])) / 1e6
        auto_brm_cash_r = float(r.get("Auto_BRM_Cash_M", 0.0))
        auto_brm_beta_r = float(r.get("Effective_Beta_BRM_Pct", 0.0))
        olympic_ref_r = float(r.get("Olympic_Ref_Ratio_Pct", 100.0))
        row_dict: Dict[str, Any] = {
            "Year": r["Year"],
            "Output ($B) [P10–P90]": f"${r['Gross_Output']/1e9:.2f}B [${mc_t['Gross_Output_p10']/1e9:.2f}B–${mc_t['Gross_Output_p90']/1e9:.2f}B]",
            "Isolated Output Shock ($M)": f"{_fmt_ofa_num(isolated_out_shock_m, 1, True)}M",
            "GDP ($B) [P10–P90]": f"${r['Direct_GDP']/1e9:.2f}B [${mc_t['Direct_GDP_p10']/1e9:.2f}B–${mc_t['Direct_GDP_p90']/1e9:.2f}B]",
            "Isolated GDP Shock ($M)": f"{_fmt_ofa_num(isolated_gdp_shock_m, 1, True)}M",
            "Isolated NCI Margin Shock ($M)": f"{_fmt_ofa_num(isolated_nci_shock_m, 1, True)}M",
        }
        if df_dyn_a is not None:
            r_a = df_dyn_a.iloc[idx_r]
            delta_gdp_ba_m = (float(r["Direct_GDP"]) - float(r_a["Direct_GDP"])) / 1e6
            delta_nci_ba_m = (float(r["Net_Cash_Income"]) - float(r_a["Net_Cash_Income"])) / 1e6
            delta_dscr_ba = float(r["DSCR"]) - float(r_a["DSCR"])
            row_dict["Δ vs. Scen A (GDP / NCI / DSCR)"] = f"{_fmt_ofa_num(delta_gdp_ba_m, 1, True)}M GDP | {_fmt_ofa_num(delta_nci_ba_m, 1, True)}M NCI | {delta_dscr_ba:+.2f}x"
        row_dict.update({
            "Statutory BRM Cash ($M)": (
                f"+${_fmt_ofa_num(auto_brm_cash_r, 1)}M ({_fmt_ofa_num(auto_brm_beta_r, 1)}% Eff, Ref={olympic_ref_r:.0f}%)"
                if auto_brm_toggle
                else "Off"
            ),
            "Debt Repriced (%) & Eff. Rate": f"{r['Debt_Repricing_Share_Pct']:.0f}% ({r['Effective_Debt_Rate_Pct']:.2f}%)",
            "CapEx Shift (%)": f"{r['CapEx_CrowdOut_Pct']:+.2f}%",
            "Capital Stock ($B)": f"${r['Capital_Stock']/1e9:.2f}B",
            "DSCR [P10–P90]": f"{r['DSCR']:.2f}x (Δ {r['DSCR']-b_r['DSCR']:+.2f}x) [{mc_t['DSCR_p10']:.2f}x–{mc_t['DSCR_p90']:.2f}x]",
            "FSF (%)": f"{r['FSF']:.2f}% (Δ {r['FSF']-b_r['FSF']:+.2f} pp)",
            "LRF [P10–P90]": f"{r['LRF']:.2f} [{mc_t['LRF_p10']:.2f}–{mc_t['LRF_p90']:.2f}]",
            "Financial Distress Risk (%)": f"{_fmt_ofa_num(float(r.get('Distress_Prob_Pct', 8.3)), 1)}%",
            "Nerlove Convergence (%)": f"{_fmt_ofa_num(float(r['Nerlove_Omega_Pct']), 1)}%" if tariff_pct > 0 else "—",
        })
        table_rows.append(row_dict)
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    # Sister Engine 7A: Short-Term Net Cash Income (NCI) & Realized Net Income (RNI) Bridge
    total_yr1_farm_rev_shock_pct = (
        (
            (-price_shock_info["effective_price_drop_pct"] * max(0.0, 1.0 + yr1_tariff_pct_output_change))
            + (yr1_tariff_pct_output_change * 100.0)
        )
        * active_tariff_params["weight"]
        + direct_rev_shock_pct
    )
    with st.expander("🔗 Sister Engine 7A: Short-Term Net Cash Income (NCI) & AgriStability / Agricorp RMP Bridge (`scripts/forecasting_engine.py`)", expanded=False):
        nci_bridge = cached_nci_bridge(
            region_name=region,
            grain_price_shock_pct=round(total_yr1_farm_rev_shock_pct, 3),
            livestock_price_shock_pct=round(total_yr1_farm_rev_shock_pct, 3),
            fuel_shock_pct=round(shock_fuel, 3),
            fert_shock_pct=round(shock_fert, 3),
            wage_shock_pct=round(shock_wages, 3),
            general_inflation_pct=round((shock_chem + shock_seed + shock_feed) / 3.0, 3),
            interest_shock_bps=round(shock_ir_bps, 1),
        )
        if nci_bridge:
            nb1, nb2, nb3, nb4 = st.columns(4)
            nb1.metric("Trend Baseline NCI ($M)", f"${nci_bridge['base_nci_m']:,.0f}M")
            nb2.metric("Shocked Short-Term NCI ($M)", f"${nci_bridge['det_nci_m']:,.0f}M", delta=f"{nci_bridge['det_nci_m']-nci_bridge['base_nci_m']:+,.0f}M")
            nb3.metric("Shocked Realized Net Income ($M)", f"${nci_bridge['det_rni_m']:,.0f}M", delta=f"{nci_bridge['det_rni_m']-nci_bridge['base_rni_m']:+,.0f}M")
            nb4.metric(
                "Empirical BRM Entitlement ($M)",
                f"${_fmt_ofa_num(nci_bridge['brm_auto_payout_m'], 1, True)}M",
                delta=f"Yr 1 Cash: ${_fmt_ofa_num(nci_bridge.get('brm_yr1_cash_payout_m', 0.0), 1, True)}M | Yr 2 Lag: ${_fmt_ofa_num(nci_bridge.get('brm_yr2_lagged_payout_m', 0.0), 1, True)}M",
                help=(
                    f"Effective Marginal BRM Payout Rate = {_fmt_ofa_num(nci_bridge.get('beta_brm_effective', 0.0)*100, 1)}% "
                    f"(Tier: {nci_bridge.get('brm_severity_tier', 'none')}). Reflects AgriStability allowable expense rules "
                    f"(debt interest spikes are non-allowable) plus Ontario Agricorp RMP & SDRM."
                ),
            )


# ==============================================================================
# TAB 3: 2026 SECTION 338 TARIFF & COUNTER-TARIFF "DOUBLE SQUEEZE"
# ==============================================================================
with tab3:
    st.subheader(f"2026 U.S. Section 338 Tariff & Canada Counter-Tariff 'Double Squeeze' — {region}")
    st.markdown(
        f"""
        **Provincial Farm Cash Receipts (`StatCan 32-10-0045-01`, {fcr_year}) Coverage:**
        Mapped Primary Agriculture commodities cover **`${_fmt_ofa_num(fcr_info['mapped_sum_m'], 1)}M`** out of **`${_fmt_ofa_num(fcr_info['total_market_fcr_m'], 1)}M`**
        in total {region} market receipts (**`{_fmt_ofa_num(fcr_info['mapped_coverage_pct'], 1)}%` empirical coverage**).
        **Nursery, Floriculture & Sod (`HS Ch. 06`)** and **Greenhouse Vegetables (`NAICS 1114`)** are modeled strictly as **Primary Agriculture**.
        """
    )

    default_eval_rate = 50.0 if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL else 25.0
    eval_tariff_pct = tariff_pct if tariff_pct > 0 else default_eval_rate

    # Build 18-sector dynamic tariff vulnerability table for the active tariff rate
    tariff_rows = []
    for comm_name, comm_data in fcr_info["commodities"].items():
        p = comm_data["params"]
        fcr_val_m = comm_data["fcr_m"]
        sh_pct = comm_data["share_of_market_fcr"] * 100.0
        ps_comm = compute_effective_tariff_price_shock(
            tariff_pct=eval_tariff_pct,
            pass_through=p["pass_through"],
            us_price_share=p["us_price_share"],
            us_lop_spillover=p["us_lop_spillover"],
        )
        eff_p_drop = ps_comm["effective_price_drop_pct"] / 100.0
        e_sr = float(p["e_sr"])
        e_lr = float(p["e_lr"])
        lam = float(p["lam"])

        dq_yr1_pct = e_sr * eff_p_drop * 100.0
        e_yr5 = e_lr + (e_sr - e_lr) * ((1.0 - lam) ** 4)
        e_yr10 = e_lr + (e_sr - e_lr) * ((1.0 - lam) ** 9)
        dq_yr5_pct = e_yr5 * eff_p_drop * 100.0
        dq_yr10_pct = e_yr10 * eff_p_drop * 100.0
        yr1_tot_rev_pct = (-eff_p_drop * 100.0 * max(0.0, 1.0 + dq_yr1_pct / 100.0)) + dq_yr1_pct
        yr10_tot_rev_pct = (-eff_p_drop * 100.0 * max(0.0, 1.0 + dq_yr10_pct / 100.0)) + dq_yr10_pct

        tariff_rows.append({
            "Primary Ag Commodity": comm_name,
            "Category": comm_data["category"],
            f"{fcr_year} {region} FCR ($M)": fcr_val_m,
            "Share of Market FCR (%)": sh_pct,
            "Direct U.S. Share": p["us_price_share"],
            "LOP Spillover": p["us_lop_spillover"],
            "Pass-Through": p["pass_through"],
            "Eff. Farm Price Drop (%)": -ps_comm["effective_price_drop_pct"],
            "Year 1 Output Δ (%)": dq_yr1_pct,
            "Year 5 Output Δ (%)": dq_yr5_pct,
            "Year 10 Output Δ (%)": dq_yr10_pct,
            "Year 1 Physical Output Δ ($M)": fcr_val_m * (dq_yr1_pct / 100.0),
            "Year 1 Total FCR / Margin Loss ($M)": fcr_val_m * (yr1_tot_rev_pct / 100.0),
            "Year 10 Total FCR / Output Loss ($M)": fcr_val_m * (yr10_tot_rev_pct / 100.0),
        })
    df_tariff_18 = pd.DataFrame(tariff_rows).sort_values("Year 10 Total FCR / Output Loss ($M)", ascending=True)

    # Executive Chart #11: Targeted Section 338 Statutory Channel Breakdown OR Top 10 Broad Commodity Exposure
    if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL:
        eval_pkg_tab3 = get_targeted_section338_package_params(
            region=region, farm_type=farm_type, tariff_pct=eval_tariff_pct, is_long_run=False
        )
        df_s338_ch = pd.DataFrame(eval_pkg_tab3["channel_breakdown"])
        if not df_s338_ch.empty:
            st.markdown(
                f"#### Statutory 2026 U.S. Section 338 Channel Breakdown (Procl. 11046–11065 Evaluated at `{_fmt_ofa_num(eval_tariff_pct, 0)}%` in {region})"
            )
            df_s338_plot = df_s338_ch.copy()
            df_s338_plot["_abs_sort"] = df_s338_plot[["yr1_phys_output_m", "yr1_total_margin_m"]].abs().max(axis=1)
            df_s338_plot = df_s338_plot.sort_values("_abs_sort", ascending=True)

            s_vals = df_s338_plot["yr1_phys_output_m"].tolist() + df_s338_plot["yr1_total_margin_m"].tolist()
            s_min, s_max = min(s_vals + [0.0]), max(s_vals + [0.0])
            s_span = max(abs(s_max - s_min), 10.0)

            fig_s338_rank = go.Figure()
            fig_s338_rank.add_trace(go.Bar(
                y=df_s338_plot["channel"].tolist(),
                x=df_s338_plot["yr1_phys_output_m"].tolist(),
                name="Year 1 Direct Output Contraction ($M)",
                orientation="h",
                marker_color=OFA_COLORS["bluewood"],
                text=[f"{_fmt_ofa_num(v, 1, True)}M" if abs(v) >= 0.05 else "$0M (Pooled)" for v in df_s338_plot["yr1_phys_output_m"]],
                textposition="outside",
                cliponaxis=False,
            ))
            fig_s338_rank.add_trace(go.Bar(
                y=df_s338_plot["channel"].tolist(),
                x=df_s338_plot["yr1_total_margin_m"].tolist(),
                name="Year 1 Cash Margin / EBITDA Squeeze ($M)",
                orientation="h",
                marker_color=OFA_COLORS["crimson"],
                text=[f"{_fmt_ofa_num(v, 1, True)}M" for v in df_s338_plot["yr1_total_margin_m"]],
                textposition="outside",
                cliponaxis=False,
            ))
            fig_s338_rank.update_layout(barmode="group", bargap=0.25)
            apply_ofa_chart_theme(
                fig_s338_rank,
                title=f"2026 U.S. Section 338 Statutory Exposure by Channel — {region} ({_fmt_ofa_num(eval_tariff_pct, 0)}% Duty + Sept 29 Bans)",
                subtitle="Primary Ag (8-Digit HTSUS Filtered) & Co-Targeted Food/Beverage Processors: Output vs. Margin Squeeze ($M)",
                source_note="Source: Statistics Canada CIMT 2025, U.S. Proclamations 11046–11065 (9903.03.12–.18) & OFA PE-ABP-IO Model.",
                height=max(410, 56 * len(df_s338_plot) + 150),
                top_margin=120,
                is_horizontal_bar=True,
            )
            fig_s338_rank.update_xaxes(
                title_text="Year 1 Impact vs. Baseline ($M)",
                range=[s_min - 0.25 * s_span, s_max + 0.12 * s_span],
            )
            fig_s338_rank.update_yaxes(title_text="")
            st.plotly_chart(fig_s338_rank, width="stretch", config=ofa_plotly_config(f"OFA_Section338_Channel_Ranking_{region}"))

            df_s338_disp = df_s338_ch.rename(columns={
                "channel": "Statutory Section 338 Channel",
                "sector_type": "Classification",
                "targeted_trade_base_m": "Targeted U.S. Trade Base ($M)",
                "yr1_phys_output_m": "Yr 1 Direct Output Δ ($M)",
                "lr_phys_output_m": "Long-Run Output Δ ($M)",
                "yr1_pure_price_m": "Yr 1 Pure Price / Blend Squeeze ($M)",
                "yr1_total_margin_m": "Yr 1 Total Cash Margin Δ ($M)",
            })
            st.dataframe(
                df_s338_disp.style.format({
                    "Targeted U.S. Trade Base ($M)": "${:,.1f}M",
                    "Yr 1 Direct Output Δ ($M)": "${:+,.1f}M",
                    "Long-Run Output Δ ($M)": "${:+,.1f}M",
                    "Yr 1 Pure Price / Blend Squeeze ($M)": "${:+,.1f}M",
                    "Yr 1 Total Cash Margin Δ ($M)": "${:+,.1f}M",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "**8-Digit HTSUS Scope Note:** Soil-grown live nursery stock & potted plants (`0602.20–0602.40`, `0602.90.30/.60`), retail yogurt (`0403.20`), and butter (`0405.10`) are outside Proclamations `11046–11065` (`$0` direct duty). Packaged beverage alcohol (`2203–2208`) and whey/molasses (`0404.10`, `1703`) reflect Proclamations `11061–11062` (Sept 29, 2026 U.S. import prohibition)."
            )
            st.markdown("---")
    else:
        df_top10_t = df_tariff_18.nsmallest(10, "Year 1 Total FCR / Margin Loss ($M)").sort_values(
            "Year 1 Total FCR / Margin Loss ($M)", ascending=False
        )
        t_vals_10 = (
            df_top10_t["Year 1 Physical Output Δ ($M)"].tolist()
            + df_top10_t["Year 1 Total FCR / Margin Loss ($M)"].tolist()
        )
        t_min, t_max = min(t_vals_10 + [0.0]), max(t_vals_10 + [0.0])
        t_span = max(abs(t_max - t_min), 10.0)

        fig_s338_rank = go.Figure()
        fig_s338_rank.add_trace(go.Bar(
            y=df_top10_t["Primary Ag Commodity"].tolist(),
            x=df_top10_t["Year 1 Physical Output Δ ($M)"].tolist(),
            name="Year 1 Physical Output Contraction ($M)",
            orientation="h",
            marker_color=OFA_COLORS["bluewood"],
            text=[f"{_fmt_ofa_num(v, 1, True)}M" for v in df_top10_t["Year 1 Physical Output Δ ($M)"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_s338_rank.add_trace(go.Bar(
            y=df_top10_t["Primary Ag Commodity"].tolist(),
            x=df_top10_t["Year 1 Total FCR / Margin Loss ($M)"].tolist(),
            name="Year 1 Total FCR / Margin Squeeze ($M)",
            orientation="h",
            marker_color=OFA_COLORS["crimson"],
            text=[f"{_fmt_ofa_num(v, 1, True)}M" for v in df_top10_t["Year 1 Total FCR / Margin Loss ($M)"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_s338_rank.update_layout(barmode="group", bargap=0.25)
        apply_ofa_chart_theme(
            fig_s338_rank,
            title=f"Top 10 Exposed Primary Agriculture Commodities under Blanket {_fmt_ofa_num(eval_tariff_pct, 0)}% Tariff — {region}",
            subtitle="Year 1 Physical Supply Contraction vs. Total Farm Cash Receipt / Margin Loss ($M)",
            source_note="Source: Statistics Canada Table 32-10-0045-01 & Nerlove-Armington Pass-Through Calibration.",
            height=470,
            top_margin=120,
            is_horizontal_bar=True,
        )
        fig_s338_rank.update_xaxes(
            title_text="Year 1 Impact vs. Baseline ($M)",
            range=[t_min - 0.25 * t_span, t_max + 0.12 * t_span],
        )
        fig_s338_rank.update_yaxes(title_text="")
        st.plotly_chart(fig_s338_rank, width="stretch", config=ofa_plotly_config(f"OFA_Blanket_Tariff_Top10_{region}"))

    st.markdown(f"#### Reference: 18-Sector Primary Agriculture Blanket Tariff Vulnerability Matrix (Evaluated at `{_fmt_ofa_num(eval_tariff_pct, 0)}%` U.S. Tariff in {region})")
    st.dataframe(
        df_tariff_18.style.format({
            f"{fcr_year} {region} FCR ($M)": "${:,.1f}M",
            "Share of Market FCR (%)": "{:.2f}%",
            "Direct U.S. Share": "{:.2f}",
            "LOP Spillover": "{:.2f}",
            "Pass-Through": "{:.2f}",
            "Eff. Farm Price Drop (%)": "{:+.1f}%",
            "Year 1 Output Δ (%)": "{:+.1f}%",
            "Year 5 Output Δ (%)": "{:+.1f}%",
            "Year 10 Output Δ (%)": "{:+.1f}%",
            "Year 1 Physical Output Δ ($M)": "${:+,.1f}M",
            "Year 1 Total FCR / Margin Loss ($M)": "${:+,.1f}M",
            "Year 10 Total FCR / Output Loss ($M)": "${:+,.1f}M",
        }),
        width="stretch",
        hide_index=True,
    )

    # Sister Engine 2: Live SQLite DB (`trade_exposure.db`) & Counter-Tariffs (`canada_counter_tariffs_sept8_2026.csv`)
    st.markdown("---")
    trade_db = cached_section338_data(region_name=region)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### 🔎 U.S. Section 338 Statutory Tariff & Prohibition Lines (`trade_exposure.db`)")
        if trade_db["has_db"] and not trade_db["section338_df"].empty:
            st.dataframe(trade_db["section338_df"], width="stretch", hide_index=True, height=320)
        else:
            st.info("Section 338 database table not found.")

    with t_col2:
        st.markdown("#### ⚠️ Canada's Sept 8, 2026 Counter-Tariff Schedule (Input & Agri-Food Surtaxes)")
        if not trade_db["counter_tariffs_df"].empty:
            ct_df = trade_db["counter_tariffs_df"]
            ct_filter = st.radio(
                "Filter Counter-Tariff Schedule",
                ["All Lines", "Farm Machinery & Equipment Parts (e.g. HS 8433.90)", "Agri-Food & Inputs"],
                horizontal=True,
            )
            if ct_filter == "Farm Machinery & Equipment Parts (e.g. HS 8433.90)":
                ct_df = ct_df[ct_df["is_machinery_tools"] == True]
            elif ct_filter == "Agri-Food & Inputs":
                ct_df = ct_df[ct_df["is_agri_food"] == True]
            st.dataframe(ct_df, width="stretch", hide_index=True, height=280)


# ==============================================================================
# TAB 4: FARM-GATE TRANSLATOR ("AVERAGE PER FARM" + 26 OMAFA PUB 60 / CEA BUDGETS)
# ==============================================================================
with tab4:
    st.subheader("Farm-Gate Impact Translator: StatCan 'Average Per Farm' & 26 Ontario Enterprise Budgets")

    fg_col1, fg_col2 = st.columns([1.05, 1.15])

    with fg_col1:
        st.markdown(f"#### 1. StatCan `32-10-0136-01` Official 'Average Per Farm' Impact ({region})")
        avg_farm_res = compute_statcan_average_farm_impact(
            df_136=df_136,
            region=region,
            farm_type=farm_type,
            shock_fuel=shock_fuel,
            shock_fert=shock_fert,
            shock_chem=shock_chem,
            shock_seed=shock_seed,
            shock_feed=shock_feed,
            shock_wages=shock_wages,
            shock_interest_rel_pct=shock_interest_rel_pct * yr1_theta,
            shock_revenue_pct=total_yr1_farm_rev_shock_pct,
            brm_subsidy_dollars=total_yr1_brm_cash_m * 1e6,
            expense_profiles=expense_profiles,
        )
        if avg_farm_res:
            st.markdown(
                f"""
                - **NAICS Farm Type:** `{shorten_naics_label(avg_farm_res['farm_type'])}` ({avg_farm_res['year']} StatCan Benchmark)
                - **Farms Reporting in {avg_farm_res['region']}:** `{avg_farm_res['farms_reporting']:,}` farms
                - **Baseline Operating Revenue / Farm:** `${avg_farm_res['base_rev_per_farm']:,.0f}` → **Shocked:** `${avg_farm_res['new_rev_per_farm']:,.0f}` (`${avg_farm_res['delta_rev_per_farm']:+,.0f}/farm`)
                - **Baseline Operating Expenses / Farm:** `${avg_farm_res['base_exp_per_farm']:,.0f}` → **Shocked:** `${avg_farm_res['new_exp_per_farm']:,.0f}` (`${avg_farm_res['delta_exp_per_farm']:+,.0f}/farm`)
                - **Net Operating Income (NOI) / Farm:** `${avg_farm_res['base_noi_per_farm']:,.0f}` → **`${avg_farm_res['new_noi_per_farm']:,.0f}` (`${avg_farm_res['delta_noi_per_farm']:+,.0f}/farm`)**
                """
            )

            # Executive Chart #12: Representative Per-Farm Operating Bridge ($/Farm)
            bridge_cats = ["Operating Revenue", "Operating Expenses", "Net Operating Income"]
            b_vals = [
                float(avg_farm_res["base_rev_per_farm"]),
                float(avg_farm_res["base_exp_per_farm"]),
                float(avg_farm_res["base_noi_per_farm"]),
            ]
            s_vals_f = [
                float(avg_farm_res["new_rev_per_farm"]),
                float(avg_farm_res["new_exp_per_farm"]),
                float(avg_farm_res["new_noi_per_farm"]),
            ]
            f_min = min(b_vals + s_vals_f + [0.0])
            f_max = max(b_vals + s_vals_f + [0.0])
            f_span = max(abs(f_max - f_min), 1000.0)

            fig_farm_bridge = go.Figure()
            fig_farm_bridge.add_trace(go.Bar(
                x=bridge_cats,
                y=b_vals,
                name=f"Baseline ({avg_farm_res['year']} StatCan)",
                marker_color=OFA_COLORS["bluewood"],
                text=[f"${v:,.0f}" for v in b_vals],
                textposition="outside",
                cliponaxis=False,
            ))
            fig_farm_bridge.add_trace(go.Bar(
                x=bridge_cats,
                y=s_vals_f,
                name="Shocked Scenario",
                marker_color=OFA_COLORS["crimson"],
                text=[f"${v:,.0f}" for v in s_vals_f],
                textposition="outside",
                cliponaxis=False,
            ))
            fig_farm_bridge.update_layout(barmode="group", bargap=0.28)
            apply_ofa_chart_theme(
                fig_farm_bridge,
                title=f"Average Per-Farm Operating Bridge — {region}",
                subtitle=f"Sector: {shorten_naics_label(avg_farm_res['farm_type'])} ({avg_farm_res['farms_reporting']:,} Farms Reporting | $/Farm)",
                source_note="Source: Statistics Canada Table 32-10-0136-01; OFA Farm-Gate Translator.",
                height=390,
                top_margin=120,
            )
            fig_farm_bridge.update_yaxes(
                title_text="Dollars per Farm ($/Farm)",
                tickprefix="$",
                range=[min(0.0, f_min - 0.18 * f_span), f_max + 0.20 * f_span],
            )
            fig_farm_bridge.update_xaxes(title_text="")
            st.plotly_chart(fig_farm_bridge, width="stretch", config=ofa_plotly_config(f"OFA_Per_Farm_Operating_Bridge_{region}"))

            st.dataframe(
                pd.DataFrame(avg_farm_res["line_items"]).style.format({
                    "Baseline ($/farm)": "${:,.0f}",
                    "Shocked ($/farm)": "${:,.0f}",
                    "Delta ($/farm)": "${:+,.0f}",
                }),
                width="stretch",
                hide_index=True,
            )

    with fg_col2:
        st.markdown("#### 2. OMAFA Publication 60 (2025/2026), OAG, OPVG, OPB & NAICS 1114 Per-Acre Calculator")
        ent_keys = list(ENTERPRISE_BUDGETS.keys())
        default_ent_key = "soybeans_rr_notill_2026"
        if "1114" in farm_type:
            default_ent_key = "gh_floriculture_1114_2025" if "gh_floriculture_1114_2025" in ent_keys else "gh_tomatoes_acre_2025"
        elif "1113" in farm_type:
            default_ent_key = "apple_hd_mature_2025"
        elif "111211" in farm_type:
            default_ent_key = "field_potatoes_chip_2025"
        elif "111219" in farm_type:
            default_ent_key = "field_tomatoes_proc_2025"

        sel_ent_key = st.selectbox(
            "Select Verified 2025/2026 Enterprise Benchmark Budget",
            ent_keys,
            index=ent_keys.index(default_ent_key),
            format_func=lambda k: ENTERPRISE_BUDGETS[k]["label"],
        )
        ent_meta = ENTERPRISE_BUDGETS[sel_ent_key]
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            sel_acres = st.number_input(f"Enterprise Scale ({ent_meta['unit_name']}s)", min_value=1.0, max_value=10000.0, value=float(ent_meta["default_acres"]), step=10.0)
        with ec2:
            inc_cash_rent = st.checkbox("Include Market Cash Rent ($/ac)", value=False, help="OMAFA Pub 60 leaves Land Rent at $0 on printed worksheets; check this to include verified Ontario market cash rent ($160-$450/ac).")
        with ec3:
            pt_custom = st.checkbox("Pass Fuel/Labour through Custom Work", value=True, help="Passes Fuel (22%), Repairs (18%), and Labour (25%) shocks into custom spraying/fertilizing rates.")

        # Enterprise-specific tariff price pass-through
        ent_tariff_comm = get_enterprise_tariff_commodity(sel_ent_key)
        ent_tariff_note = ""
        if tariff_pct > 0:
            if tariff_commodity == TARGETED_SECTION338_OPTION_LABEL:
                wedge_ent = float(np.log(1.0 + tariff_pct / 100.0) / np.log(1.50))
                k_low = sel_ent_key.lower()
                if "flori" in k_low or ("1114" in k_low and "nursery" not in k_low and "sod" not in k_low):
                    # Targeted HS Ch. 06 cut flowers/bulbs/cuttings (Procl. 11048)
                    d_ln_p_fl = -0.72 * float(np.log(1.0 + tariff_pct / 100.0)) * 0.55
                    ent_tariff_price_drop_pct = (1.0 - float(np.exp(d_ln_p_fl))) * 100.0
                    ent_tariff_note = f"Targeted Procl. 11048 Floriculture Exposure: -{ent_tariff_price_drop_pct:.1f}% blended farm price squeeze."
                elif "nursery" in k_low or "sod" in k_low:
                    ent_tariff_price_drop_pct = 0.0
                    ent_tariff_note = "Procl. 11048 Scope Rule: Soil-grown woody nursery stock & sod (0602.20–0602.40) are untariffed (0% direct duty)."
                elif "Corn" in ent_tariff_comm:
                    ent_tariff_price_drop_pct = 0.62 * wedge_ent
                    ent_tariff_note = f"Sept 29 Spirit Ban Backward Spillover: -{ent_tariff_price_drop_pct:.2f}% local Ontario corn basis dip."
                elif "Dairy" in ent_tariff_comm:
                    ent_tariff_price_drop_pct = 1.05 * wedge_ent
                    ent_tariff_note = f"Sept 15 Cheese & Sept 29 Whey Ban: -{ent_tariff_price_drop_pct:.2f}% P5 pooled milk blend dilution."
                elif "grape" in k_low or "wine" in k_low:
                    ent_tariff_price_drop_pct = 4.20 * wedge_ent
                    ent_tariff_note = f"Sept 29 Wine Export Prohibition: -{ent_tariff_price_drop_pct:.1f}% specialty/Icewine grape price squeeze."
                else:
                    ent_tariff_price_drop_pct = 0.0
                    ent_tariff_note = "Procl. 11046–11065 Scope Rule: Selected crop/livestock enterprise is outside 2026 Section 338 tariff annexes (0% direct border duty)."
            elif tariff_commodity == "All Commodities (Provincial FCR Weighted)" or tariff_commodity == ent_tariff_comm:
                ent_t_cfg = get_commodity_tariff_params(ent_tariff_comm, region=region)
                ent_t_shock = compute_effective_tariff_price_shock(
                    tariff_pct=tariff_pct,
                    pass_through=ent_t_cfg["pass_through"],
                    us_price_share=ent_t_cfg["us_price_share"],
                    us_lop_spillover=ent_t_cfg["us_lop_spillover"],
                )
                ent_tariff_price_drop_pct = float(ent_t_shock["effective_price_drop_pct"])
            else:
                ent_tariff_price_drop_pct = 0.0
        else:
            ent_tariff_price_drop_pct = 0.0

        ent_sim = simulate_enterprise_budget(
            enterprise_key=sel_ent_key,
            acres=sel_acres,
            shock_seed_pct=shock_seed,
            shock_fertilizer_pct=shock_fert,
            shock_pesticides_pct=shock_chem,
            shock_fuel_pct=shock_fuel,
            shock_repairs_pct=shock_fuel * 0.5,
            shock_labour_pct=shock_wages,
            shock_interest_bps=shock_ir_bps,
            shock_output_price_pct=(-ent_tariff_price_drop_pct) + direct_rev_shock_pct,
            include_cash_rent=inc_cash_rent,
            pass_through_custom_work=pt_custom,
        )

        st.markdown(
            f"""
            **{ent_sim['label']} (`{ent_sim['basis']}` | Target Yield: `{ent_sim['yield_qty']:,.2f} {ent_sim['yield_unit']}` @ `{ent_sim['price_unit']}`):**
            - **Operating Cost / Acre:** `${ent_sim['base_op_ac']:,.2f}` → **`${ent_sim['new_op_ac']:,.2f}/ac`** (`${ent_sim['delta_op_ac']:+,.2f}/ac` / `{_fmt_ofa_num(ent_sim['pct_op_increase'], 1, True)}%`)
            - **Net Margin / Acre (After Overhead):** `${ent_sim['base_net_margin_ac']:,.2f}` → **`${ent_sim['new_net_margin_ac']:,.2f}/ac`** (`${ent_sim['delta_net_margin_ac']:+,.2f}/ac`)
            - **Whole-Enterprise ({ent_sim['acres']:,.0f} {ent_sim['unit_name']}s) Net Margin Impact:** **`${ent_sim['farm_delta_net_margin']:+,.0f}`** (Baseline: `${ent_sim['farm_base_net_margin']:,.0f}` → New: `${ent_sim['farm_new_net_margin']:,.0f}`)
            """
        )
        if ent_tariff_note:
            st.caption(f"🛡️ **Statutory Tariff Treatment:** {ent_tariff_note}")

        st.dataframe(
            pd.DataFrame(ent_sim["line_items"]).style.format({
                "Baseline ($/ac)": "${:,.2f}",
                "Shocked ($/ac)": "${:,.2f}",
                "Delta ($/ac)": "${:+,.2f}",
                "Whole Farm Delta ($)": "${:+,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

    # Sister Engine 3: StatCan FIPI (18-10-0258-01) & Live Commodity Futures Signals
    st.markdown("---")
    st.markdown("#### 📡 Sister Engine 3: StatCan FIPI (`18-10-0258-01`) Input Price Index & Live Commodity Futures")
    fipi_kpis = cached_fipi_kpis(region_name=region)
    if fipi_kpis:
        fipi_cols = st.columns(len(fipi_kpis))
        for idx, fk in enumerate(fipi_kpis):
            yoy_str = f"{_fmt_ofa_num(fk['yoy_change'], 1, True)}% YoY" if fk.get("yoy_change") is not None else "N/A"
            fipi_cols[idx].metric(
                f"{fk['category']} ({fk['quarter_label']})",
                f"{_fmt_ofa_num(fk['value'], 1)}",
                delta=yoy_str,
                delta_color="inverse",
            )
    if not df_mkt.empty:
        st.dataframe(df_mkt, width="stretch", hide_index=True)


# ==============================================================================
# TAB 5: ECONOMETRIC CALIBRATION MATRICES & EXECUTIVE BRIEFINGS
# ==============================================================================
with tab5:
    st.subheader("Verified Econometric Calibration Matrices & Executive Report Exports")
    st.markdown(
        """
        Every parameter in this simulator is grounded in official Statistics Canada tables, OMAFA Publication 60
        worksheets, and peer-reviewed agricultural economics literature—with **zero contrived heuristics**.

        - **Active IOIC 2022 / NAICS 2022 Multiplier Vintage Lock:** Statistics Canada Tables `36-10-0594-01` (Canada) &
          `36-10-0595-01` (Provincial/Territorial, 5th CODR dimension `Geographical coverage`: *All provinces* vs. *Within province*),
          released **November 19, 2025** (reference year **2022**; next release **2023** scheduled **November 20, 2026**), paired
          with Detail Supply and Use Table `36-10-0478-01` (**2022**) and Symmetric IO Table `36-10-0001-01`.
        - **PE-ABP-IO Export Tariff Protocol (Wohlgenant 1989 & 8-Digit HTSUS Scope):** Foreign export tariffs on primary crops
          never trigger Ghosh physical supply rationing on domestic food processors; instead, co-targeted food & beverage
          manufacturing shocks (`NAICS 3115`, `3118`, `31212`, `3121A`) enter directly with upstream farm coefficients netted out
          via Analysis-by-Parts (`M_p^gross - sum_f a_{f,p} M_f^gross`).
        - **Employment Unit & Constant-2022$ Deflation:** All `36-10-0594-01`, `36-10-0595-01`, and `PEC*` HFCE employment metrics
          report **total headcount jobs** (Table `36-10-0480-01`), deflated to constant `2022$` via official annual FPPI (`32-10-0099-01`)
          and annual average CPI (`18-10-0005-01`).
        """
    )

    with st.expander("1. Master 18-Sector Nerlove-Armington & Tariff Pass-Through Calibration Table", expanded=True):
        cal_rows = []
        for k, v in PRIMARY_AG_TARIFF_CONFIG.items():
            cal_rows.append({
                "Sector": k,
                "Category": v["category"],
                "NAICS": v["naics"],
                "IO Join Code": v["io_join_code"],
                "SR Supply Elasticity": v["e_sr"],
                "SR Range (P10–P90)": f"{v['e_sr_p10_p90'][0]:.2f} to {v['e_sr_p10_p90'][1]:.2f}",
                "LR Supply Elasticity": v["e_lr"],
                "LR Range (P10–P90)": f"{v['e_lr_p10_p90'][0]:.2f} to {v['e_lr_p10_p90'][1]:.2f}",
                "Nerlove Speed": v["lam"],
                "Farm Incidence (SR)": v["pass_through"],
                "U.S. Share CAN": v["us_price_share"],
                "U.S. Share ON": v["us_price_share_on"],
                "LOP Spillover": v["us_lop_spillover"],
                "Citations": v["citations"],
            })
        st.dataframe(pd.DataFrame(cal_rows), width="stretch", hide_index=True)

    with st.expander("2. Empirical 5x5 Stochastic Variance-Covariance Matrix & Provincial TFP Decomposition", expanded=False):
        mc_c1, mc_c2 = st.columns(2)
        labels_5 = ["1. FIPI (% YoY)", "2. TFP (% YoY)", "3. Interest Rate (Δ pp)", "4. Net Cash Income (% YoY)", "5. Farm CapEx (% YoY)"]
        with mc_c1:
            st.markdown(f"**Correlation Matrix $\\mathbf{{R}}$ ({mc_info['region']} — {mc_info['basis']}):**")
            st.dataframe(pd.DataFrame(mc_info["corr"], index=labels_5, columns=labels_5).style.format("{:.2f}"), width="stretch")
        with mc_c2:
            st.markdown(f"**Variance-Covariance Matrix $\\boldsymbol{{\\Sigma}}$ ({mc_info['region']} — {mc_info['basis']}):**")
            st.dataframe(pd.DataFrame(mc_info["cov"], index=labels_5, columns=labels_5).style.format("{:.3f}"), width="stretch")

        st.markdown("**Darku, Malla & Tran (2016) Provincial TFP Decomposition (Technical Change vs. Scale Effects):**")
        st.dataframe(pd.DataFrame(PROVINCIAL_TFP_DECOMPOSITION), width="stretch", hide_index=True)

    with st.expander(
        f"2B. StatCan 36-10-0595-01 Intra-Provincial vs. Inter-Provincial IO Multiplier Retention Matrix ({region}, {target_io_year})",
        expanded=False,
    ):
        ret_rows = cached_io_retention_table(region_name=region, target_yr=target_io_year)
        if ret_rows:
            df_ret_io = pd.DataFrame(ret_rows)
            st.dataframe(
                df_ret_io.style.format({
                    "Direct GDP Mult": "{:.3f}",
                    "Within-Province Output Mult": "{:.3f}",
                    "All-Provinces Output Mult": "{:.3f}",
                    "ROC Output Spillover": "{:.3f}",
                    "Within-Province GDP Mult": "{:.3f}",
                    "All-Provinces GDP Mult": "{:.3f}",
                    "ROC GDP Spillover": "{:.3f}",
                    "Provincial GDP Retention (%)": "{:.1f}%",
                    "Within-Province Jobs (/ $1M)": "{:.2f}",
                    "All-Provinces Jobs (/ $1M)": "{:.2f}",
                    "Provincial Jobs Retention (%)": "{:.1f}%",
                }),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                label=f"📥 Download StatCan 36-10-0595-01 IO Scope Retention Matrix ({region}, CSV)",
                data=df_ret_io.to_csv(index=False).encode("utf-8"),
                file_name=f"OFA_StatCan_3610059501_IO_Scope_Retention_{region}_{target_io_year}.csv",
                mime="text/csv",
                key="btn_dl_io_ret_csv",
            )

    with st.expander("3. Sister Engine 7B: Export Official OFA Executive Briefing (.docx), Board Deck (.pptx) & Audit CSV", expanded=True):
        export_sig = (
            region,
            farm_type,
            preset_name,
            round(base_output_dollars, 2),
            round(base_gdp_dollars, 2),
            round(combined_yr1_output_impact_m, 4),
            round(combined_yr1_gdp_impact_m, 4),
            round(combined_yr1_jobs_impact, 2),
            round(tariff_pct, 2),
            tariff_commodity,
            sel_ent_key,
            round(sel_acres, 2),
            inc_cash_rent,
            pt_custom,
            round(shock_fuel, 3),
            round(shock_fert, 3),
            round(shock_chem, 3),
            round(shock_seed, 3),
            round(shock_feed, 3),
            round(shock_wages, 3),
            round(shock_ir_bps, 2),
            round(shock_capex_pct, 3),
            round(tfp_growth_pct, 3),
            round(direct_rev_shock_pct, 3),
            round(total_yr1_brm_cash_m, 3),
            auto_brm_toggle,
            brm_regime,
            horizon,
            n_sims,
            io_geo_scope,
        )
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            word_bytes = cached_word_briefing_bytes(
                scenario_sig=export_sig,
                _df_dyn=df_dyn,
                _df_base_dyn=df_base_dyn,
                _mc_summary=mc_summary,
                _abp_res=abp_res,
                _avg_farm_res=avg_farm_res,
                _ent_sim=ent_sim,
            )
            if word_bytes:
                st.download_button(
                    label="📄 Download OFA Executive Briefing (.docx)",
                    data=word_bytes,
                    file_name=f"OFA_Executive_Briefing_{region}_{farm_type[:15].strip()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
        with dl_c2:
            pptx_bytes = cached_pptx_deck_bytes(
                scenario_sig=export_sig,
                _df_dyn=df_dyn,
                _df_base_dyn=df_base_dyn,
                _abp_res=abp_res,
                _avg_farm_res=avg_farm_res,
                _ent_sim=ent_sim,
            )
            if pptx_bytes:
                st.download_button(
                    label="📊 Download OFA Board Slide Deck (.pptx)",
                    data=pptx_bytes,
                    file_name=f"OFA_Board_Deck_{region}_{farm_type[:15].strip()}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    width="stretch",
                )
        with dl_c3:
            csv_bytes = df_dyn.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download 10-Year Trajectory Audit (.csv)",
                data=csv_bytes,
                file_name=f"OFA_Growth_Risk_Trajectory_{region}_{farm_type[:15].strip()}.csv",
                mime="text/csv",
                width="stretch",
            )

# Official OFA Brand Guide (2025–26, Page 9) Split-Keyline Contact Bar Footer
st.markdown(
    """
    <div class="ofa-footer-bar">
        <div class="ofa-keyline">
            <div class="ofa-keyline-red"></div>
            <div class="ofa-keyline-slate"></div>
        </div>
        <strong>Ontario Federation of Agriculture</strong> &nbsp;|&nbsp;
        ofa.on.ca &nbsp;//&nbsp; Tel: 519-821-8883 / 1-800-668-3276 &nbsp;|&nbsp;
        <span style="color:#D7182A; font-weight:700;">Farms and Food Forever.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

