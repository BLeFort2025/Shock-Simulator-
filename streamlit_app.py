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

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE if (_HERE / "scripts").exists() else _HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.calibrated_parameters as _mod_cp
import scripts.farm_cost_impact_model as _mod_fcim
import scripts.forecasting_engine as _mod_fe
import scripts.growth_risk_engine as _mod_gre

importlib.reload(_mod_cp)
importlib.reload(_mod_fcim)
importlib.reload(_mod_fe)
importlib.reload(_mod_gre)

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
    get_debt_amortization_params,
    get_debt_repricing_schedule,
    get_economic_depreciation_rate,
    get_hotelling_sector_profile,
    get_mc_covariance_params,
)
from scripts.farm_cost_impact_model import ENTERPRISE_BUDGETS, simulate_enterprise_budget
from scripts.growth_risk_engine import (
    compute_basket_impact,
    compute_cost_commodity_breakdown,
    compute_downstream_abp_impact,
    compute_effective_tariff_price_shock,
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
    load_fipi_input_cost_kpis,
    load_section338_and_counter_tariff_data,
    load_subsector_io_multipliers,
    run_dynamic_projection,
    run_monte_carlo_projections,
    run_short_term_nci_bridge,
)
from scripts.io_multipliers_engine import load_baskets_config

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & INSTITUTIONAL STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="OFA Growth & Risk Simulator | Macro-to-Farm-Gate Decision System",
    page_icon="🔮",
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
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #14532d 100%);
        color: #ffffff;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 6px solid #22c55e;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.96rem;
        color: #cbd5e1;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.9rem 1.05rem;
        box-shadow: 0 2px 6px rgba(15,23,42,0.05);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.2rem;
    }
    .badge-primary-ag {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        border: 1px solid #86efac;
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
    io_df = pd.read_parquet(PROJECT_ROOT / "data" / "latest" / "io_multipliers_standardized.parquet")
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


# ------------------------------------------------------------------------------
# SCENARIO PRESETS (100% Grounded in StatCan FIPI, OMAFA Pub 60, & Sept 8 Counter-Tariffs)
# ------------------------------------------------------------------------------
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom Scenario (Manual Sliders)": {
        "desc": "Configure custom input cost, Section 338 tariff, CapEx, TFP, and BRM policy shocks.",
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2026 Section 338 Tariff (25% U.S. Duty + Sept 8 Machinery Counter-Tariff)": {
        "desc": (
            "Empirically grounded in the 25% U.S. Section 338 border duty paired with Canada's Sept 8, 2026 "
            "counter-tariff schedule (`canada_counter_tariffs_sept8_2026.csv`: 24.5% mean surtax across 96 "
            "HS 82/84/85/87 machinery & tool lines -> -5.4% Jorgenson CapEx shock via ε_{I,c}^{SR} = -0.220; "
            "zero surtax on bulk diesel HS 27 or fertilizer HS 31)."
        ),
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 25.0, "include_proc": True,
        "capex_pct": -5.4, "tfp_pct": None, "brm_m": 0.0,
    },
    "Statutory Max Section 338 Stress Test (50% U.S. Duty + 50% Steel/Plastics Surtax)": {
        "desc": (
            "Tests the statutory maximum 50% U.S. Section 338 tariff (Bank of Canada SWP 2026-36 Atkeson-Burstein "
            "markup compression) paired with Canada's 50% Sept 8, 2026 surtax tier on 413 steel, aluminum & plastic "
            "lines (HS 39/72/73/76 -> -11.0% Jorgenson CapEx shock via ε_{I,c}^{SR} = -0.220 × 50%)."
        ),
        "fuel": 0.0, "fert": 0.0, "chem": 0.0, "seed": 0.0, "feed": 0.0, "wages": 0.0, "ir_bps": 0.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 50.0, "include_proc": True,
        "capex_pct": -11.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2025 → 2026 OMAFA Pub 60 Empirical Cost Shift (Exact Row-Crop Benchmark)": {
        "desc": (
            "Exact acreage-weighted 2025→2026 OMAFA Publication 60 cost shift across Ontario Corn, Soybeans & "
            "Winter Wheat: Fuel -7.0% ($1.20→$1.12/L diesel), Fertilizer +19.0% (+14.6% Corn / +19.0% Soy / +33.0% Wheat), "
            "Pesticides & Custom Spraying +8.5% ($13.75→$18.00/ac pass), Seed +2.5%, Commercial Feed -2.3% (2025 ON FIPI), "
            "Labour +9.0%, and Operating Interest -150 bps (6.95%→5.45%)."
        ),
        "fuel": -7.0, "fert": 19.0, "chem": 8.5, "seed": 2.5, "feed": -2.3, "wages": 9.0, "ir_bps": -150.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "2022 StatCan FIPI Energy, Fertilizer & BoC Rate Hike Replay (Historical Benchmark)": {
        "desc": (
            "Exact historical replay of the 2022 Ontario Farm Input Price Index (StatCan Table 18-10-0258-01) "
            "and Bank of Canada tightening cycle: Machinery Fuel +61.2%, Fertilizer +71.0%, Commercial Seed +5.0%, "
            "Commercial Feed +18.9%, Hired Labour +7.5%, and +400 bps interest rate tightening."
        ),
        "fuel": 61.2, "fert": 71.0, "chem": 15.0, "seed": 5.0, "feed": 18.9, "wages": 7.5, "ir_bps": 400.0,
        "tariff_comm": "All Commodities (Provincial FCR Weighted)", "tariff_pct": 0.0, "include_proc": True,
        "capex_pct": 0.0, "tfp_pct": None, "brm_m": 0.0,
    },
    "Leamington & Niagara CEA Greenhouse / Nursery Stress (NAICS 1114 Primary Ag)": {
        "desc": (
            "Calibrated to 2025/2026 OGVG & NAICS 1114 benchmarks (Deep Research Output #4 Sections 5–6): "
            "25% U.S. Section 338 tariff on Greenhouse Vegetables & HS Ch. 06 Nursery/Floriculture, paired with "
            "+25.0% Enbridge Zone 2 natural gas heating & carbon escalation, +14.0% water-soluble fertility, "
            "+12.0% biological IPM predators, +8.0% grafted seedlings/liners, and +5.5% ESDC SAWP/TFW wage increase."
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
            help="+100 bps = +1.00 percentage point added to the effective interest rate on farm debt.",
        )

    with st.expander("🛡️ 2. Section 338 Tariff & Downstream ABP", expanded=True):
        comm_options = ["All Commodities (Provincial FCR Weighted)"] + list(PRIMARY_AG_TARIFF_CONFIG.keys())
        default_comm_idx = comm_options.index(preset["tariff_comm"]) if preset["tariff_comm"] in comm_options else 0
        tariff_commodity = st.selectbox("Target Primary Ag Commodity", comm_options, index=default_comm_idx, key=f"sel_comm_{preset_name}")
        tariff_pct = st.slider("U.S. Section 338 Border Tariff (%)", 0.0, 50.0, float(preset["tariff_pct"]), 1.0, key=f"sl_tariff_{preset_name}")
        direct_rev_shock_pct = st.slider("Additional Non-Tariff Market Revenue Shock (%)", -30.0, 30.0, 0.0, 0.5, key=f"sl_rev_{preset_name}")
        include_processing = st.checkbox(
            "Include Downstream Processing (NAICS 311/312 Analysis-by-Parts)",
            value=bool(preset["include_proc"]),
            key=f"chk_proc_{preset_name}",
            help="Applies official provincial StatCan IO multipliers for NAICS 3111-3121A with SUT basic-price farm input coefficients a_{farm, proc}^{dom} netted out to prevent double-counting.",
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
            "Pinned Custom Snapshot (Scenario A)",
        ] + compare_preset_options

        if "pinned_scenario_a" not in st.session_state:
            st.session_state["pinned_scenario_a"] = None

        default_cmp_idx = 0
        if st.session_state.get("force_pinned_mode") and st.session_state.get("pinned_scenario_a") is not None:
            default_cmp_idx = 2
            st.session_state["force_pinned_mode"] = False

        compare_mode = st.selectbox(
            "Compare Active Scenario (B) Against Benchmark (A)",
            options=compare_options,
            index=default_cmp_idx,
            key="sel_compare_mode",
            help=(
                "Select any reference scenario as Benchmark Scenario A (or pin your current slider configuration) "
                "to view side-by-side KPI deltas (B − A) and overlaid 10-year trajectory curves."
            ),
        )

        auto_pin_label = (
            f"Pinned ({preset_name[:22]}...)"
            if preset_name != "Custom Scenario (Manual Sliders)"
            else f"Pinned (Tariff {tariff_pct:.0f}%, IR {shock_ir_bps:+.0f}bps, Fert {shock_fert:+.0f}%)"
        )
        pin_label_input = st.text_input(
            "Custom Label for Pinned Scenario A",
            value=auto_pin_label,
            key="txt_pin_label",
        )
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            pin_current_btn = st.button("📌 Pin Current as A", use_container_width=True)
        with p_col2:
            clear_pin_btn = st.button(
                "🗑️ Clear Pin",
                use_container_width=True,
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
fcr_info = compute_provincial_fcr_weights(df_fcr, region=region)
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
    else ("112A" if "112" in farm_type else ("111A" if "111" in farm_type else None))
)

if primary_join_code in ("111A", "1114A", "112A"):
    sub_mults = load_subsector_io_multipliers(io_df, region=region, year=target_io_year)
    fallback_mults = {
        "111A": {"output": 1.5430, "gdp": 0.7200, "jobs": 4.9690, "wages": 0.2390},
        "1114A": {"output": 1.4915, "gdp": 0.7380, "jobs": 9.9510, "wages": 0.4485},
        "112A": {"output": 2.1120, "gdp": 0.7120, "jobs": 7.0350, "wages": 0.3170},
    }
    m_unit = sub_mults.get(primary_join_code, fallback_mults[primary_join_code])
    gdp_coeff = float(m_unit["gdp"])
    out_mult = float(m_unit["output"])
    jobs_mult_per_m = float(m_unit["jobs"])
    wages_coeff = float(m_unit["wages"])
else:
    unit_basket = compute_basket_impact(io_df, baskets_cfg, region, target_io_year, "primary_agriculture", 1.0)
    gdp_coeff = float(unit_basket.get("gdp", 0.52))
    out_mult = float(unit_basket.get("output", 1.65))
    jobs_mult_per_m = float(unit_basket.get("jobs", 6.2)) * 1e6
    wages_coeff = float(unit_basket.get("wages", 0.25))

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

# 7. Compute Tariff & Revenue Shock Parameters (Fixes Bug #1, Gap #6, Gap #7)
if tariff_commodity == "All Commodities (Provincial FCR Weighted)":
    comp_params = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"])
    active_tariff_params = {
        "commodity": f"Composite ({farm_type})",
        "tariff_pct": tariff_pct,
        "e_sr": comp_params["e_sr"],
        "e_lr": comp_params["e_lr"],
        "lam": comp_params["lam"],
        "pass_through": comp_params["pass_through"],
        "us_price_share": comp_params["us_price_share"],
        "us_lop_spillover": comp_params["us_lop_spillover"],
        "weight": 1.0,
    }
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

# Year-1 Tariff Revenue/Output Shock ($) + Direct Non-Tariff Market Shock ($) + BRM Offset ($)
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
)

mc_summary = run_monte_carlo_projections(
    n_sims=n_sims,
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
    mc_basis=mc_basis,
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
)

# 9. Isolate Pure Shock Impact (Shocked Trajectory minus Unshocked Baseline Trajectory)
#    AND Downstream Processing Analysis-by-Parts (ABP) Impact
base_yr1_row = df_base_dyn.iloc[0]
base_yrN_row = df_base_dyn.iloc[-1]
yr1_row = df_dyn.iloc[0]
yrN_row = df_dyn.iloc[-1]

yr1_primary_delta_output = float(yr1_row["Gross_Output"]) - float(base_yr1_row["Gross_Output"])
yr1_primary_delta_gdp = float(yr1_row["Direct_GDP"]) - float(base_yr1_row["Direct_GDP"])
yr1_nci_delta = float(yr1_row["Net_Cash_Income"]) - float(base_yr1_row["Net_Cash_Income"])
yr1_hh_induced_gdp_m = float(yr1_row["Induced_HH_GDP_Delta_M"]) - float(base_yr1_row["Induced_HH_GDP_Delta_M"])
yr1_hh_induced_jobs = float(yr1_row["Induced_HH_Jobs_Delta"]) - float(base_yr1_row["Induced_HH_Jobs_Delta"])
yr1_primary_delta_jobs = ((yr1_primary_delta_output / 1e6) * jobs_mult_per_m) + yr1_hh_induced_jobs
yr1_primary_delta_wages = yr1_primary_delta_output * wages_coeff

yr1_trend_primary_output_m = (float(base_yr1_row["Gross_Output"]) - base_output_dollars) / 1e6
yr1_trend_primary_gdp_m = (float(base_yr1_row["Direct_GDP"]) - base_gdp_dollars) / 1e6

abp_res = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=yr1_primary_delta_output,
    primary_cost_shock_dollars=intermediate_cost_shock_dollars,
)

if include_processing:
    combined_yr1_output_impact_m = (yr1_primary_delta_output / 1e6) + abp_res["proc_net_output_m"]
    combined_yr1_gdp_impact_m = (yr1_primary_delta_gdp / 1e6) + abp_res["proc_net_gdp_m"]
    combined_yr1_jobs_impact = yr1_primary_delta_jobs + abp_res["proc_net_jobs"]
    combined_yr1_wages_impact_m = (yr1_primary_delta_wages / 1e6) + abp_res["proc_net_wages_m"]
else:
    combined_yr1_output_impact_m = yr1_primary_delta_output / 1e6
    combined_yr1_gdp_impact_m = yr1_primary_delta_gdp / 1e6
    combined_yr1_jobs_impact = yr1_primary_delta_jobs
    combined_yr1_wages_impact_m = yr1_primary_delta_wages / 1e6

yrN_primary_shock_gdp_m = (float(yrN_row["Direct_GDP"]) - float(base_yrN_row["Direct_GDP"])) / 1e6
yrN_primary_shock_out_m = (float(yrN_row["Gross_Output"]) - float(base_yrN_row["Gross_Output"])) / 1e6
cum_10yr_shock_gdp_m = float((df_dyn["Direct_GDP"] - df_base_dyn["Direct_GDP"]).sum()) / 1e6

abp_res_yrN = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=yrN_primary_shock_out_m * 1e6,
    primary_cost_shock_dollars=intermediate_cost_shock_dollars,
    is_long_run=True,
)
if include_processing:
    combined_yrN_gdp_impact_m = yrN_primary_shock_gdp_m + abp_res_yrN["proc_net_gdp_m"]
    combined_yrN_out_impact_m = yrN_primary_shock_out_m + abp_res_yrN["proc_net_output_m"]
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
        }

df_dyn_a: Optional[pd.DataFrame] = None
comparison_summary: Optional[Dict[str, Any]] = None

if scen_a_cfg is not None:
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
    if t_comm_a == "All Commodities (Provincial FCR Weighted)":
        comp_p_a = fcr_info["composites"].get(farm_type, fcr_info["composites"]["All farm types"])
        t_params_a = {
            "commodity": f"Composite ({farm_type})",
            "tariff_pct": t_pct_a,
            "e_sr": comp_p_a["e_sr"],
            "e_lr": comp_p_a["e_lr"],
            "lam": comp_p_a["lam"],
            "pass_through": comp_p_a["pass_through"],
            "us_price_share": comp_p_a["us_price_share"],
            "us_lop_spillover": comp_p_a["us_lop_spillover"],
            "weight": 1.0,
        }
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
            "us_price_share": cp_a["us_price_share"],
            "us_lop_spillover": cp_a["us_lop_spillover"],
            "weight": c_wt_a,
        }

    dir_mkt_a = base_output_dollars * (float(scen_a_cfg.get("direct_rev_pct", 0.0)) / 100.0)
    brm_off_a = float(scen_a_cfg.get("brm_m", 0.0)) * 1e6

    df_dyn_a = run_dynamic_projection(
        horizon=horizon,
        base_output=base_output_dollars,
        base_gdp=base_gdp_dollars,
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
    )

    yr1_prim_out_a = float(df_dyn_a.iloc[0]["Gross_Output"]) - float(base_yr1_row["Gross_Output"])
    yrN_prim_out_a = float(df_dyn_a.iloc[-1]["Gross_Output"]) - float(base_yrN_row["Gross_Output"])
    abp_yr1_a = compute_downstream_abp_impact(
        io_df=io_df,
        region=region,
        year=target_io_year,
        farm_type=farm_type,
        primary_supply_shock_dollars=yr1_prim_out_a,
        primary_cost_shock_dollars=int_cost_a,
        is_long_run=False,
    )
    abp_yrN_a = compute_downstream_abp_impact(
        io_df=io_df,
        region=region,
        year=target_io_year,
        farm_type=farm_type,
        primary_supply_shock_dollars=yrN_prim_out_a,
        primary_cost_shock_dollars=int_cost_a,
        is_long_run=True,
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
        jobs_mult_per_m=jobs_mult_per_m,
        include_proc_a=bool(scen_a_cfg.get("include_proc", True)),
        include_proc_b=include_processing,
        brm_adhoc_m_a=float(scen_a_cfg.get("brm_m", 0.0)),
        brm_adhoc_m_b=brm_subsidy_m,
    )

# Factor-by-Factor Attribution using Deep Research Output #5 & #6 Duality, ABP & Repricing Equations
hh_reg = "Ontario" if region == "Ontario" else "Canada"
hh_cfg = FARM_HOUSEHOLD_MPC_MULTIPLIERS[hh_reg]
eff_hh_gdp_sr = float(hh_cfg["eff_induced_gdp_per_dollar_nci_sr"])        # 0.0864 ON / 0.1056 CAN
eff_hh_fte_sr = float(hh_cfg["eff_induced_fte_per_million_nci_sr"])       # 0.972 ON / 1.140 CAN
inp_elasts = hotelling_prof["inputs"]
sector_yr1_margin_share = float(hotelling_prof["yr1_margin_share"])

# Exact linear ABP response coefficients from `compute_downstream_abp_impact` (eliminates hardcoded 1.45/0.42/3.20)
abp_unit_supply = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=1e6,
    primary_cost_shock_dollars=0.0,
)
abp_unit_cost = compute_downstream_abp_impact(
    io_df=io_df,
    region=region,
    year=target_io_year,
    farm_type=farm_type,
    primary_supply_shock_dollars=0.0,
    primary_cost_shock_dollars=1e6,
)
proc_out_per_m_supply = float(abp_unit_supply["proc_net_output_m"]) if include_processing else 0.0
proc_gdp_per_m_supply = float(abp_unit_supply["proc_net_gdp_m"]) if include_processing else 0.0
proc_jobs_per_m_supply = float(abp_unit_supply["proc_net_jobs"]) if include_processing else 0.0
proc_out_per_m_cost = float(abp_unit_cost["proc_net_output_m"]) if include_processing else 0.0
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
    nci_delta_d = (phys_out_d * gdp_coeff) - margin_comp_d
    # For intermediate inputs, margin compression reduces Value-Added GDP directly;
    # for Hired Labour (CSNA Primary Income), wages reduce NCI but only affect VA GDP via physical output + induced HH
    va_gdp_d = (phys_out_d * gdp_coeff) - (0.0 if is_labor else margin_comp_d) + (nci_delta_d * eff_hh_gdp_sr)
    int_cost_for_abp_m = 0.0 if is_labor else (cost_dollars / 1e6)
    proc_out_m = ((phys_out_d / 1e6) * proc_out_per_m_supply) + (int_cost_for_abp_m * proc_out_per_m_cost)
    proc_gdp_m = ((phys_out_d / 1e6) * proc_gdp_per_m_supply) + (int_cost_for_abp_m * proc_gdp_per_m_cost)
    proc_jobs = ((phys_out_d / 1e6) * proc_jobs_per_m_supply) + (int_cost_for_abp_m * proc_jobs_per_m_cost)
    jobs_d = ((phys_out_d / 1e6) * jobs_mult_per_m) + proc_jobs + ((nci_delta_d / 1e6) * eff_hh_fte_sr)
    return {
        "Shock Factor": label,
        "Slider Setting": setting,
        "Yr 1 Farm Margin / NCI ($M)": nci_delta_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": phys_out_d / 1e6,
        "Downstream Proc ABP Output ($M)": proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (va_gdp_d / 1e6) + proc_gdp_m,
        "Employment Impact (FTEs)": jobs_d,
        "Economic Transmission Mechanism": (
            f"Hotelling ε_SR={eps_sr:.3f} ({m_share*100:.1f}% Yr-1 Margin Compression / {(1-m_share)*100:.1f}% Physical Output)"
        ),
    }

factor_attribution_rows: List[Dict[str, Any]] = []
for row_cand in [
    _build_input_factor_row("⛽ Fuel, Heating Gas & Electricity", f"{shock_fuel:+.1f}%", "fuel", shock_fuel, cost_shock_fuel),
    _build_input_factor_row("🌱 Fertilizer, Lime & Nutrients", f"{shock_fert:+.1f}%", "fertilizer", shock_fert, cost_shock_fert),
    _build_input_factor_row("🧪 Pesticides & Biological IPM", f"{shock_chem:+.1f}%", "pesticides", shock_chem, cost_shock_chem),
    _build_input_factor_row("🌾 Seed, Plugs, Liners & Media", f"{shock_seed:+.1f}%", "seed", shock_seed, cost_shock_seed),
    _build_input_factor_row("🐄 Commercial Livestock Feed", f"{shock_feed:+.1f}%", "feed", shock_feed, cost_shock_feed),
    _build_input_factor_row("👷 Hired Farm & Packhouse Labour", f"{shock_wages:+.1f}%", "labor", shock_wages, cost_shock_wages, is_labor=True),
]:
    if row_cand is not None:
        factor_attribution_rows.append(row_cand)

# Interest Rate Factor Row (Module 1 + Module 3 + Module 4)
if abs(shock_ir_bps) > 1e-4:
    eff_yr1_bps = shock_ir_bps * yr1_theta
    ir_phys_out_d = base_output_dollars * (-0.0015) * (eff_yr1_bps / 100.0)
    ir_nci_d = (ir_phys_out_d * gdp_coeff) - cost_shock_int_yr1
    ir_hh_gdp_d = ir_nci_d * eff_hh_gdp_sr
    ir_va_gdp_d = (ir_phys_out_d * gdp_coeff) + ir_hh_gdp_d
    ir_proc_out_m = (ir_phys_out_d / 1e6) * proc_out_per_m_supply
    ir_proc_gdp_m = (ir_phys_out_d / 1e6) * proc_gdp_per_m_supply
    ir_proc_jobs = (ir_phys_out_d / 1e6) * proc_jobs_per_m_supply
    ir_jobs_d = ((ir_phys_out_d / 1e6) * jobs_mult_per_m) + ir_proc_jobs + ((ir_nci_d / 1e6) * eff_hh_fte_sr)
    factor_attribution_rows.append({
        "Shock Factor": "🏦 Farm Debt Carrying Cost (Interest Rate)",
        "Slider Setting": f"{shock_ir_bps:+.0f} bps (Yr 1 Eff: {eff_yr1_bps:+.1f} bps on θ_yr1={yr1_theta*100:.0f}%)",
        "Yr 1 Farm Margin / NCI ($M)": ir_nci_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": ir_phys_out_d / 1e6,
        "Downstream Proc ABP Output ($M)": ir_proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (ir_va_gdp_d / 1e6) + ir_proc_gdp_m,
        "Employment Impact (FTEs)": ir_jobs_d,
        "Economic Transmission Mechanism": (
            f"θ_yr1={yr1_theta*100:.0f}% Debt Repriced in Yr 1 ({-cost_shock_int_yr1/1e6:+,.1f}M vs. {-cost_shock_int_full/1e6:+,.1f}M Yr 5) + MPC_SR=0.12 + Yr 2–10 CapEx Crowd-Out"
        ),
    })

# Tariff, Market Revenue, CapEx, TFP, and BRM Factor Rows
# Note: Year-1 CapEx changes K_next (Year 2 capital stock); Year-1 TFP drift deviation scales Y_1
yr1_tfp_dev_out_d = base_output_dollars * ((tfp_growth_pct - default_tfp) / 100.0) if abs(tfp_growth_pct - default_tfp) > 1e-4 else 0.0

for f_lbl, f_set, f_out_d, f_extra_nci_d, f_mech in [
    (
        f"🛡️ U.S. Section 338 Tariff ({tariff_commodity})",
        f"{tariff_pct:.1f}% tariff (-{price_shock_info['effective_price_drop_pct']:.2f}% eff. price)",
        yr1_tariff_shock_dollars,
        yr1_tariff_price_margin_dollars,
        f"Envelope Price Drop (-{price_shock_info['effective_price_drop_pct']:.2f}%) + Nerlove ε_SR={active_tariff_params['e_sr']:.2f} → ε_LR={active_tariff_params['e_lr']:.2f}",
    ),
    (
        "📉 Non-Tariff Market Revenue Shock",
        f"{direct_rev_shock_pct:+.1f}%",
        direct_mkt_shock_dollars,
        0.0,
        "Direct farm-gate revenue & supply shock",
    ),
    (
        "📈 TFP Productivity Drift Deviation",
        f"{tfp_growth_pct - default_tfp:+.2f}% vs. {default_tfp:.2f}% base",
        yr1_tfp_dev_out_d,
        0.0,
        "KLEMS Hicks-neutral productivity shift",
    ),
    (
        "🏛️ Ad-Hoc BRM / Disaster Direct Offset",
        f"+${brm_subsidy_m:,.0f}M payment",
        0.0,
        brm_offset_dollars,
        "CSNA Direct Cash Transfer to NCI (100% Margin Relief + MPC_SR=0.12 + Yr 2–10 CapEx Stabilization)",
    ),
]:
    if abs(f_out_d) < 1e3 and abs(f_extra_nci_d) < 1e3:
        continue
    f_nci_d = (f_out_d * gdp_coeff) + f_extra_nci_d
    f_va_gdp_d = (f_out_d * gdp_coeff) + (f_nci_d * eff_hh_gdp_sr)
    f_proc_out_m = (f_out_d / 1e6) * proc_out_per_m_supply
    f_proc_gdp_m = (f_out_d / 1e6) * proc_gdp_per_m_supply
    f_proc_jobs = (f_out_d / 1e6) * proc_jobs_per_m_supply
    f_jobs_d = ((f_out_d / 1e6) * jobs_mult_per_m) + f_proc_jobs + ((f_nci_d / 1e6) * eff_hh_fte_sr)
    factor_attribution_rows.append({
        "Shock Factor": f_lbl,
        "Slider Setting": f_set,
        "Yr 1 Farm Margin / NCI ($M)": f_nci_d / 1e6,
        "Yr 1 Physical Ag Output ($M)": f_out_d / 1e6,
        "Downstream Proc ABP Output ($M)": f_proc_out_m,
        "Yr 1 Value-Added GDP ($M)": (f_va_gdp_d / 1e6) + f_proc_gdp_m,
        "Employment Impact (FTEs)": f_jobs_d,
        "Economic Transmission Mechanism": f_mech,
    })

yr1_auto_brm_cash_m = float(yr1_row.get("Auto_BRM_Cash_M", 0.0))
yr1_auto_brm_ent_m = float(yr1_row.get("Auto_BRM_Entitlement_M", 0.0))
yr1_auto_brm_lag_m = max(0.0, yr1_auto_brm_ent_m - yr1_auto_brm_cash_m)
yr1_eff_beta_brm_pct = float(yr1_row.get("Effective_Beta_BRM_Pct", 0.0))
yr1_brm_tier = str(yr1_row.get("BRM_Severity_Tier", "none"))

if auto_brm_toggle:
    regime_short = "80% SCAP + RMP" if brm_regime == "statutory_scap_80pct" else "90% Enhanced + RMP"
    if yr1_auto_brm_cash_m > 1e-3:
        auto_brm_d = yr1_auto_brm_cash_m * 1e6
        auto_brm_gdp_d = auto_brm_d * eff_hh_gdp_sr
        auto_brm_jobs_d = yr1_auto_brm_cash_m * eff_hh_fte_sr
        factor_attribution_rows.append({
            "Shock Factor": "🏛️ Statutory BRM Auto-Stabilizer (AgriStability + ON RMP)",
            "Slider Setting": f"{regime_short} (β_BRM={yr1_eff_beta_brm_pct:.1f}%)",
            "Yr 1 Farm Margin / NCI ($M)": yr1_auto_brm_cash_m,
            "Yr 1 Physical Ag Output ($M)": 0.0,
            "Downstream Proc ABP Output ($M)": 0.0,
            "Yr 1 Value-Added GDP ($M)": auto_brm_gdp_d / 1e6,
            "Employment Impact (FTEs)": auto_brm_jobs_d,
            "Economic Transmission Mechanism": (
                f"Yr 1 Entitlement +${yr1_auto_brm_ent_m:,.1f}M → +${yr1_auto_brm_cash_m:,.1f}M Yr 1 Cash + "
                f"+${yr1_auto_brm_lag_m:,.1f}M Yr 2 Tax-Filing Settlement (λ_allow=0.78, λ_interest=0.0)"
            ),
        })
    elif abs(shock_ir_bps) > 1e-4:
        factor_attribution_rows.append({
            "Shock Factor": "🏛️ Statutory BRM Auto-Stabilizer (AgriStability + ON RMP)",
            "Slider Setting": f"{regime_short} (λ_interest = 0.0)",
            "Yr 1 Farm Margin / NCI ($M)": 0.0,
            "Yr 1 Physical Ag Output ($M)": 0.0,
            "Downstream Proc ABP Output ($M)": 0.0,
            "Yr 1 Value-Added GDP ($M)": 0.0,
            "Employment Impact (FTEs)": 0.0,
            "Economic Transmission Mechanism": (
                "Statutory Rule: Debt interest expenses are 100% non-allowable under AgriStability (λ_interest = 0.0) → $0.00M payout"
            ),
        })

if abs(shock_capex_pct) > 1e-4:
    capex_yr2_k_delta_m = (base_capex * (shock_capex_pct / 100.0) * (ramp_schedule[0] if ramp_schedule else 1.0)) / 1e6
    factor_attribution_rows.append({
        "Shock Factor": "🏗️ Capital Investment (CapEx / Machinery Surtax)",
        "Slider Setting": f"{shock_capex_pct:+.1f}% CapEx ({capex_yr2_k_delta_m:+,.1f}M Yr-1 Investment)",
        "Yr 1 Farm Margin / NCI ($M)": 0.0,
        "Yr 1 Physical Ag Output ($M)": 0.0,
        "Downstream Proc ABP Output ($M)": 0.0,
        "Yr 1 Value-Added GDP ($M)": 0.0,
        "Employment Impact (FTEs)": 0.0,
        "Economic Transmission Mechanism": (
            f"Jorgenson Capital Gestation: Yr-1 CapEx shock ({capex_yr2_k_delta_m:+,.1f}M) enters K_2..K_10 (ICOR={icor_cfg['icor']})"
        ),
    })


# ------------------------------------------------------------------------------
# HERO BANNER & EXECUTIVE KPI RIBBON
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <h1>🔮 OFA Growth & Risk Simulator — Standalone Decision Support System</h1>
        <p>
            <strong>Jurisdiction:</strong> {region} &nbsp;|&nbsp;
            <strong>Primary Ag Sector:</strong> {farm_type} &nbsp;|&nbsp;
            <strong>Baselines:</strong> StatCan IO ({target_io_year}) · FCR ({fcr_year}) · Expense Profiles ({latest_exp_yr}) · Balance Sheet ({health_year})
            &nbsp;&nbsp;<span class="badge-primary-ag">NAICS 1114 / HS Ch. 06 = Primary Agriculture</span>
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

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Year 1 Physical Output Shock</div>
            <div class="kpi-val" style="color: {'#64748b' if abs(combined_yr1_output_impact_m) < 0.05 else ('#16a34a' if combined_yr1_output_impact_m > 0 else '#dc2626')};">
                {combined_yr1_output_impact_m:+,.1f} $M
            </div>
            <div class="kpi-sub">Primary: {yr1_primary_delta_output/1e6:+,.1f}M | Proc ABP: {abp_res['proc_net_output_m'] if include_processing else 0.0:+,.1f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Year 1 GDP & Margin Shock</div>
            <div class="kpi-val" style="color: {'#64748b' if abs(combined_yr1_gdp_impact_m) < 0.05 and abs(yr1_nci_delta/1e6) < 0.05 else ('#16a34a' if (combined_yr1_gdp_impact_m if abs(combined_yr1_gdp_impact_m) >= abs(yr1_nci_delta/1e6) else yr1_nci_delta/1e6) > 0 else '#dc2626')};">
                {combined_yr1_gdp_impact_m:+,.1f} $M GDP
            </div>
            <div class="kpi-sub">Yr 1 Farm NCI Margin: {yr1_nci_delta/1e6:+,.1f} $M | HH Induced: {yr1_hh_induced_gdp_m:+,.1f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Year 1 Employment Shock</div>
            <div class="kpi-val" style="color: {'#64748b' if abs(combined_yr1_jobs_impact) < 0.5 else ('#16a34a' if combined_yr1_jobs_impact > 0 else '#dc2626')};">
                {combined_yr1_jobs_impact:+,.0f} FTEs
            </div>
            <div class="kpi-sub">Supply Chain + HH Induced ({yr1_hh_induced_jobs:+,.0f} FTEs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Year {horizon} GDP Shock (Multi-Yr)</div>
            <div class="kpi-val" style="color: {'#64748b' if abs(combined_yrN_gdp_impact_m) < 0.05 else ('#16a34a' if combined_yrN_gdp_impact_m > 0 else '#dc2626')};">
                {combined_yrN_gdp_impact_m:+,.1f} $M
            </div>
            <div class="kpi-sub">Primary: {yrN_primary_shock_gdp_m:+,.1f}M | Proc ABP (LR): {abp_res_yrN['proc_net_gdp_m'] if include_processing else 0.0:+,.1f}M | {horizon}-Yr Cum: {cum_10yr_shock_gdp_m:+,.0f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Debt Service Coverage (DSCR)</div>
            <div class="kpi-val" style="color: {'#16a34a' if yr1_row['DSCR'] >= 1.5 else ('#d97706' if yr1_row['DSCR'] >= 1.15 else '#dc2626')};">
                {yr1_row['DSCR']:.2f}x ({dscr_yr1_delta:+.2f}x)
            </div>
            <div class="kpi-sub">Yr 1 Repriced (θ={yr1_theta*100:.0f}%): {implied_debt_rate*100:.2f}% → {yr1_row['Effective_Debt_Rate_Pct']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k6:
    distress_p = float(yr1_row.get("Distress_Prob_Pct", 8.3))
    base_distress_p = float(base_yr1_row.get("Distress_Prob_Pct", 8.3))
    distress_delta = distress_p - base_distress_p
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Liquidity Resilience (LRF)</div>
            <div class="kpi-val" style="color: {'#16a34a' if yr1_row['LRF'] >= 1.0 else ('#d97706' if yr1_row['LRF'] >= 0.85 else '#dc2626')};">
                {yr1_row['LRF']:.2f} ({lrf_yr1_delta:+.2f})
            </div>
            <div class="kpi-sub">FSF: {yr1_row['FSF']:.2f}% ({fsf_yr1_delta:+.2f} pp) | Distress P: {distress_p:.1f}% ({distress_delta:+.1f} pp)</div>
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
        cmp_c1, cmp_c2 = st.columns([1.45, 1.0])
        with cmp_c1:
            df_cmp_tbl = pd.DataFrame(comparison_summary["comparison_rows"])
            st.dataframe(df_cmp_tbl, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Side-by-Side Scenario Comparison (CSV)",
                data=df_cmp_tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"OFA_Scenario_Comparison_{region}.csv",
                mime="text/csv",
                key="btn_dl_cmp_csv",
            )
        with cmp_c2:
            df_cmp_bar = pd.DataFrame([
                {"Metric": "Yr 1 Combined Output ($M)", f"Scenario A ({lbl_a})": m_a["yr1_output_m"], f"Scenario B ({lbl_b})": m_b["yr1_output_m"]},
                {"Metric": "Yr 1 Value-Added GDP ($M)", f"Scenario A ({lbl_a})": m_a["yr1_gdp_m"], f"Scenario B ({lbl_b})": m_b["yr1_gdp_m"]},
                {"Metric": "Yr 1 Farm NCI Margin ($M)", f"Scenario A ({lbl_a})": m_a["yr1_nci_m"], f"Scenario B ({lbl_b})": m_b["yr1_nci_m"]},
                {"Metric": f"Yr {horizon} Multi-Yr GDP ($M)", f"Scenario A ({lbl_a})": m_a["yrN_gdp_m"], f"Scenario B ({lbl_b})": m_b["yrN_gdp_m"]},
            ])
            fig_cmp = px.bar(
                df_cmp_bar,
                y="Metric",
                x=[f"Scenario A ({lbl_a})", f"Scenario B ({lbl_b})"],
                orientation="h",
                barmode="group",
                title=f"Scenario A vs. Scenario B Impact Comparison ({region}, $M)",
                labels={"value": "Shock Impact vs. Baseline ($M)", "variable": "Scenario"},
                color_discrete_map={
                    f"Scenario A ({lbl_a})": "#d97706",
                    f"Scenario B ({lbl_b})": "#0284c7",
                },
                height=340,
            )
            fig_cmp.update_layout(margin=dict(t=35, b=10, l=10, r=10), legend_orientation="h", legend_y=-0.25)
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption(
                f"📌 **Net Policy / Scenario Delta (B − A):** Year 1 Farm NCI Margin **`{d_ba['yr1_nci_m']:+,.1f} $M`** | "
                f"Year 1 DSCR **`{d_ba['yr1_dscr']:+.2f}x`** | Year 1 Distress Prob **`{d_ba['yr1_distress_pct']:+.2f} pp`** | "
                f"{horizon}-Yr Cumulative Primary GDP **`{d_ba['cum_10yr_gdp_m']:+,.0f} $M`**."
            )

# Active Shock Factor Attribution Panel (immediately shows the exact magnitude of every active slider)
if factor_attribution_rows:
    with st.expander(f"🎯 Active Shock Factor Attribution — Exact Margin vs. Physical Output Split of {len(factor_attribution_rows)} Modeled Factor(s)", expanded=True):
        df_attr_chart = pd.DataFrame(factor_attribution_rows)
        df_attr_table = df_attr_chart.copy()
        if len(factor_attribution_rows) > 1:
            tot_row = pd.DataFrame([{
                "Shock Factor": "∑ Total Combined Active Shocks (Reconciled to KPI Banner)",
                "Slider Setting": f"{len(factor_attribution_rows)} active channels",
                "Yr 1 Farm Margin / NCI ($M)": float(df_attr_chart["Yr 1 Farm Margin / NCI ($M)"].sum()),
                "Yr 1 Physical Ag Output ($M)": float(df_attr_chart["Yr 1 Physical Ag Output ($M)"].sum()),
                "Downstream Proc ABP Output ($M)": float(df_attr_chart["Downstream Proc ABP Output ($M)"].sum()),
                "Yr 1 Value-Added GDP ($M)": float(df_attr_chart["Yr 1 Value-Added GDP ($M)"].sum()),
                "Employment Impact (FTEs)": float(df_attr_chart["Employment Impact (FTEs)"].sum()),
                "Economic Transmission Mechanism": (
                    f"Combined Primary + ABP Output = {float(df_attr_chart['Yr 1 Physical Ag Output ($M)'].sum() + df_attr_chart['Downstream Proc ABP Output ($M)'].sum()):+,.1f} $M"
                ),
            }])
            df_attr_table = pd.concat([df_attr_chart, tot_row], ignore_index=True)

        a_col1, a_col2 = st.columns([1.45, 1.0])
        with a_col1:
            st.dataframe(
                df_attr_table.style.format({
                    "Yr 1 Farm Margin / NCI ($M)": "{:+,.2f} $M",
                    "Yr 1 Physical Ag Output ($M)": "{:+,.2f} $M",
                    "Downstream Proc ABP Output ($M)": "{:+,.2f} $M",
                    "Yr 1 Value-Added GDP ($M)": "{:+,.2f} $M",
                    "Employment Impact (FTEs)": "{:+,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"💡 **CSNA & Hotelling Duality Accounting:** Interest rate shocks reprice via the 5-year maturity schedule (`θ_yr1 = {yr1_theta*100:.0f}%` in Year 1 → `100%` by Year 5), compressing **Net Cash Income (`NCI`)** and **Multi-Year CapEx** while leaving Year-1 physical output virtually intact (`-0.15%` per `+100 bps`). Intermediate input cost spikes compress **Year-1 Value-Added GDP & Margins (`{hotelling_prof['yr1_margin_share']*100:.1f}%`)** while reducing Year-1 physical output by only **`{hotelling_prof['yr1_output_share']*100:.1f}%`** (`ε_SR`)."
            )
        with a_col2:
            fig_attr = px.bar(
                df_attr_chart,
                y="Shock Factor",
                x=["Yr 1 Farm Margin / NCI ($M)", "Yr 1 Value-Added GDP ($M)", "Yr 1 Physical Ag Output ($M)"],
                orientation="h",
                barmode="group",
                title="Year 1 Margin Squeeze vs. Value-Added GDP vs. Physical Output ($M)",
                labels={"value": "Year 1 Impact ($M)", "variable": "Channel"},
                color_discrete_map={
                    "Yr 1 Farm Margin / NCI ($M)": "#dc2626",
                    "Yr 1 Value-Added GDP ($M)": "#d97706",
                    "Yr 1 Physical Ag Output ($M)": "#1e3a5f",
                },
                height=max(240, 85 * len(df_attr_chart) + 95),
            )
            fig_attr.update_layout(margin=dict(t=35, b=10, l=10, r=10), legend_orientation="h", legend_y=-0.28)
            st.plotly_chart(fig_attr, use_container_width=True)
else:
    st.info(
        f"**No Active Shock Sliders (`0.0 $M` Shock Impact):** All input cost, interest rate, tariff, CapEx, and BRM sliders are currently at `0`. "
        f"Move any slider in the left sidebar to see its exact isolated Farm Margin (`NCI`), Value-Added GDP, Physical Output, and Multi-Year CapEx impact (baseline trend growth of `+{default_tfp:.2f}%/yr` TFP = `+{yr1_trend_primary_output_m:,.1f} $M` Yr 1 Primary Output is held as the reference baseline)."
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

    col_left, col_right = st.columns([1.15, 1.0])

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
            fig_cost = px.bar(
                df_cost_bd,
                y="Commodity",
                x=["Fuel ($M)", "Fertilizer ($M)", "Pesticides ($M)", "Seed & Plants ($M)", "Feed ($M)", "Labour ($M)", "Interest ($M)"],
                orientation="h",
                title=f"Year 1 Direct Cost Shock by NAICS Sub-Sector ({region}, $M — Incl. Sub-Sector θ_yr1 Debt Repricing)",
                labels={"value": "Year 1 Cost Shock ($M)", "variable": "Input Category"},
                height=430,
            )
            fig_cost.update_layout(yaxis={"categoryorder": "total ascending"}, legend_orientation="h", legend_y=-0.22)
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Adjust any Input Cost or Debt Interest Rate slider in the sidebar to inspect sub-sector cost exposure.")
            if not df_cost_bd.empty:
                st.dataframe(
                    df_cost_bd[["Commodity", "Total Revenue ($M)", "Total Expenses ($M)"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with col_right:
        st.markdown("#### Primary Agriculture Output Disposition & Downstream ABP Netting")
        disp = abp_res["disposition_shares"]
        disp_df = pd.DataFrame([
            {"Channel": "1. Sold to Domestic NAICS 311/312 Processors (s_proc)", "Share (%)": disp["s_proc"] * 100.0},
            {"Channel": "2. Intra-Ag / Feedlot / Seed / Landscaping (s_intra_ag)", "Share (%)": disp["s_intra_ag"] * 100.0},
            {"Channel": "3. Direct Fresh Household / Foodservice (s_fresh)", "Share (%)": disp["s_fresh"] * 100.0},
            {"Channel": "4. Direct Primary Exports & ROC (s_exp)", "Share (%)": disp["s_exp"] * 100.0},
        ])
        fig_disp = px.pie(
            disp_df,
            names="Channel",
            values="Share (%)",
            hole=0.45,
            title=f"StatCan SUT Output Disposition: {farm_type} ({region})",
            height=300,
        )
        fig_disp.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_disp, use_container_width=True)

        st.markdown(
            f"""
            **Analysis-by-Parts (ABP) Non-Double-Counting Summary ({'ACTIVE' if include_processing else 'INACTIVE'}):**
            - **Farm Output Sold to Domestic Processors ($s_{{\\text{{proc}}}} = {abp_res['s_proc']*100:.1f}\\%$):** `${abp_res['farm_to_proc_flow_m']:+,.1f}M`
            - **ABP Netted Downstream GDP Impact (`NAICS 311/312`):** `${abp_res['proc_net_gdp_m']:+,.1f}M` (`{abp_res['proc_net_jobs']:+,.0f}` FTEs)
            - **Double-Counted Upstream Farm GDP Removed via $a_{{\\text{{farm, proc}}}}^{{\\text{{dom}}}}$:** `${abp_res['double_counted_farm_gdp_removed_m']:+,.1f}M`
            """
        )

    st.markdown("#### Downstream Food & Beverage Manufacturing (`NAICS 311/312`) ABP Sub-Sector Table")
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
                "ABP Net Jobs (FTEs)": "{:+,.0f}",
                "ABP Net Wages ($M)": "{:+,.2f}",
                "Upstream Farm GDP Netted Out ($M)": "{:+,.2f}",
            }),
            use_container_width=True,
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
        st.dataframe(fisc_df.style.format({"Fiscal Revenue Impact ($M)": "{:+,.2f} $M"}), use_container_width=True, hide_index=True)
        total_yr1_brm_cash_m = brm_subsidy_m + yr1_auto_brm_cash_m
        if total_yr1_brm_cash_m > 0:
            sub_imp = compute_simulator_subsidy_summary(total_yr1_brm_cash_m * 1e6, region=region, year=target_io_year)
            st.success(
                f"**Active BRM Cash Stabilization (`+${total_yr1_brm_cash_m:,.1f}M` Yr 1 Cash Inflow = "
                f"`+${yr1_auto_brm_cash_m:,.1f}M` Statutory + `+${brm_subsidy_m:,.1f}M` Ad-Hoc; "
                f"`+${yr1_auto_brm_lag_m:,.1f}M` Yr 2 Tax-Filing Settlement Carryover):** "
                f"Generates **`${sub_imp['gdp']/1e6:+,.1f}M`** in induced/indirect GDP and preserves **`{sub_imp['jobs']:+,.0f}`** jobs across rural {region} supply chains."
            )
        elif auto_brm_toggle and abs(shock_ir_bps) > 1e-4:
            st.info(
                "**Statutory BRM Auto-Stabilizer Active (`$0.00M` Entitlement):** Under the Sustainable CAP AgriStability Handbook, "
                "farm debt interest expenses are 100% non-allowable (`λ_interest = 0.0`). Isolated interest rate shocks do not trigger statutory BRM payouts."
            )

    with f_col2:
        st.markdown("#### 🍎 Sister Engine 5: OAG Horticulture Multi-Stage Value-Chain (HLRF)")
        try:
            hlrf_rows = compute_simulator_hlrf_summary(region=region, year=target_io_year)
            if hlrf_rows:
                st.dataframe(
                    pd.DataFrame(hlrf_rows).style.format({
                        "Direct Spend ($M)": "${:,.2f}M",
                        "Output Impact ($M)": "${:,.2f}M",
                        "GDP Impact ($M)": "${:,.2f}M",
                        "Jobs (FTEs)": "{:,.0f}",
                    }),
                    use_container_width=True,
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

    c_fan1, c_fan2 = st.columns(2)

    with c_fan1:
        # Gross Output Fan Chart
        out_base = [v / 1e9 for v in df_base_dyn["Gross_Output"]]
        out_det = [v / 1e9 for v in df_dyn["Gross_Output"]]
        out_p10 = [mc_summary[t]["Gross_Output_p10"] / 1e9 for t in t_vals]
        out_p90 = [mc_summary[t]["Gross_Output_p90"] / 1e9 for t in t_vals]

        fig_out = go.Figure()
        fig_out.add_trace(go.Scatter(x=years_x + years_x[::-1], y=out_p90 + out_p10[::-1], fill="toself", fillcolor="rgba(30, 58, 95, 0.18)", line=dict(color="rgba(255,255,255,0)"), name="P10–P90 Monte Carlo Band"))
        fig_out.add_trace(go.Scatter(x=years_x, y=out_base, mode="lines", line=dict(color="#64748b", dash="dash", width=2), name=f"Unshocked Baseline Trend (+{default_tfp:.2f}% TFP)"))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_out.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Gross_Output"]],
                mode="lines+markers",
                line=dict(color="#d97706", dash="dashdot", width=2.5),
                name=f"Scenario A: {comparison_summary['label_a']} ($B)",
            ))
        fig_out.add_trace(go.Scatter(x=years_x, y=out_det, mode="lines+markers", line=dict(color="#1e3a5f", width=3), name="Active Scenario B ($B)"))
        fig_out.update_layout(title=f"Gross Output Trajectory: Shocked vs. Baseline Trend ({region}, $B)", yaxis_title="Gross Output ($B)", height=390, legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig_out, use_container_width=True)

    with c_fan2:
        # Direct Value-Added GDP Fan Chart
        gdp_base = [v / 1e9 for v in df_base_dyn["Direct_GDP"]]
        gdp_det = [v / 1e9 for v in df_dyn["Direct_GDP"]]
        gdp_p10 = [mc_summary[t]["Direct_GDP_p10"] / 1e9 for t in t_vals]
        gdp_p90 = [mc_summary[t]["Direct_GDP_p90"] / 1e9 for t in t_vals]

        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(x=years_x + years_x[::-1], y=gdp_p90 + gdp_p10[::-1], fill="toself", fillcolor="rgba(22, 163, 74, 0.18)", line=dict(color="rgba(255,255,255,0)"), name="P10–P90 Monte Carlo Band"))
        fig_gdp.add_trace(go.Scatter(x=years_x, y=gdp_base, mode="lines", line=dict(color="#64748b", dash="dash", width=2), name=f"Unshocked Baseline Trend (+{default_tfp:.2f}% TFP)"))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_gdp.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Direct_GDP"]],
                mode="lines+markers",
                line=dict(color="#d97706", dash="dashdot", width=2.5),
                name=f"Scenario A: {comparison_summary['label_a']} ($B)",
            ))
        fig_gdp.add_trace(go.Scatter(x=years_x, y=gdp_det, mode="lines+markers", line=dict(color="#16a34a", width=3), name="Active Scenario B GDP ($B)"))
        fig_gdp.update_layout(title=f"Direct Value-Added GDP: Shocked vs. Baseline Trend ({region}, $B)", yaxis_title="Value-Added GDP ($B)", height=390, legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig_gdp, use_container_width=True)

    c_fan3, c_fan4 = st.columns(2)
    with c_fan3:
        # DSCR & LRF Trajectory
        fig_fin = go.Figure()
        fig_fin.add_trace(go.Scatter(x=years_x, y=df_base_dyn["DSCR"], mode="lines", line=dict(color="#94a3b8", dash="dash", width=2), name="Unshocked Baseline DSCR (x)"))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_fin.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["DSCR"],
                mode="lines+markers",
                line=dict(color="#d97706", dash="dashdot", width=2.5),
                name=f"Scenario A DSCR: {comparison_summary['label_a']}",
            ))
        fig_fin.add_trace(go.Scatter(x=years_x, y=df_dyn["DSCR"], mode="lines+markers", line=dict(color="#0284c7", width=3), name="Scenario B DSCR (x)"))
        fig_fin.add_trace(go.Scatter(x=years_x, y=df_dyn["LRF"], mode="lines+markers", line=dict(color="#7c3aed", width=3), name="Scenario B LRF"))
        fig_fin.add_hline(y=1.15, line_dash="dot", line_color="#dc2626", annotation_text="DSCR Covenant Threshold (1.15x)")
        fig_fin.update_layout(title="Solvency & Liquidity Trajectory: DSCR and LRF vs. Baseline", yaxis_title="Ratio / Index", height=360, legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig_fin, use_container_width=True)

    with c_fan4:
        # Capital Stock & Net Cash Income Trajectory
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(x=years_x, y=[v / 1e9 for v in df_dyn["Capital_Stock"]], name="Productive Capital Stock ($B)", marker_color="#94a3b8"))
        fig_cap.add_trace(go.Scatter(x=years_x, y=[v / 1e9 for v in df_base_dyn["Net_Cash_Income"]], mode="lines", name="Unshocked Baseline NCI ($B)", yaxis="y2", line=dict(color="#fdba74", dash="dash", width=2)))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_cap.add_trace(go.Scatter(
                x=years_x,
                y=[v / 1e9 for v in df_dyn_a["Net_Cash_Income"]],
                mode="lines+markers",
                name=f"Scenario A NCI: {comparison_summary['label_a']} ($B)",
                yaxis="y2",
                line=dict(color="#0284c7", dash="dashdot", width=2.5),
            ))
        fig_cap.add_trace(go.Scatter(x=years_x, y=[v / 1e9 for v in df_dyn["Net_Cash_Income"]], mode="lines+markers", name="Scenario B NCI ($B)", yaxis="y2", line=dict(color="#ea580c", width=3)))
        fig_cap.update_layout(
            title=f"Capital Stock Accumulation & Net Cash Income (ICOR={icor_cfg['icor']})",
            yaxis=dict(title="Capital Stock ($B)"),
            yaxis2=dict(title="Net Cash Income ($B)", overlaying="y", side="right"),
            height=360,
            legend_orientation="h",
            legend_y=-0.2,
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    c_fan5, c_fan6 = st.columns(2)
    with c_fan5:
        # 5-Year Debt Rollover Staircase (Theta_t) & Effective Portfolio Interest Rate (%)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Bar(
            x=years_x,
            y=df_dyn["Debt_Repricing_Share_Pct"],
            name="Cumulative Debt Repriced Θ_t (%)",
            marker_color="rgba(14, 165, 233, 0.30)",
        ))
        fig_roll.add_trace(go.Scatter(
            x=years_x,
            y=df_base_dyn["Effective_Debt_Rate_Pct"],
            mode="lines",
            name="Baseline Effective Rate (%)",
            yaxis="y2",
            line=dict(color="#64748b", dash="dash", width=2),
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_roll.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["Effective_Debt_Rate_Pct"],
                mode="lines+markers",
                name=f"Scenario A Eff. Rate: {comparison_summary['label_a']} (%)",
                yaxis="y2",
                line=dict(color="#d97706", dash="dashdot", width=2.5),
            ))
        fig_roll.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["Effective_Debt_Rate_Pct"],
            mode="lines+markers",
            name="Scenario B Effective Rate (%)",
            yaxis="y2",
            line=dict(color="#dc2626", width=3),
        ))
        fig_roll.update_layout(
            title="5-Year Contractual Debt Rollover (Θ_t) & Effective Portfolio Rate (%)",
            yaxis=dict(title="Debt Repriced Share Θ_t (%)", range=[0, 110]),
            yaxis2=dict(title="Effective Interest Rate (%)", overlaying="y", side="right"),
            height=340,
            legend_orientation="h",
            legend_y=-0.22,
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with c_fan6:
        # Endogenous CapEx Crowd-Out (%) & FFSC/FCC Solvency Logit Distress Probability (P_t, %)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=years_x,
            y=df_dyn["CapEx_CrowdOut_Pct"],
            name="Scenario B CapEx Shift (%)",
            marker_color="#f59e0b",
        ))
        fig_dist.add_trace(go.Scatter(
            x=years_x,
            y=df_base_dyn["Distress_Prob_Pct"],
            mode="lines",
            name="Baseline Arrears/Distress Prob (%)",
            yaxis="y2",
            line=dict(color="#94a3b8", dash="dash", width=2),
        ))
        if df_dyn_a is not None and comparison_summary is not None:
            fig_dist.add_trace(go.Scatter(
                x=years_x,
                y=df_dyn_a["Distress_Prob_Pct"],
                mode="lines+markers",
                name=f"Scenario A Distress P_t: {comparison_summary['label_a']} (%)",
                yaxis="y2",
                line=dict(color="#0284c7", dash="dashdot", width=2.5),
            ))
        fig_dist.add_trace(go.Scatter(
            x=years_x,
            y=df_dyn["Distress_Prob_Pct"],
            mode="lines+markers",
            name="Scenario B Distress Prob P_t (%)",
            yaxis="y2",
            line=dict(color="#991b1b", width=3),
        ))
        fig_dist.update_layout(
            title="Jorgenson/Pecking-Order CapEx Shift (%) & Solvency Distress Probability (P_t)",
            yaxis=dict(title="Endogenous CapEx Shift (%)"),
            yaxis2=dict(title="Distress Probability P_t (%)", overlaying="y", side="right"),
            height=340,
            legend_orientation="h",
            legend_y=-0.22,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Complete Year-by-Year Trajectory Table (Fixes Bug #5 — includes Capital Stock, DSCR, FSF, LRF, Debt Rollover, CapEx Crowd-Out)
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
            "Isolated Output Shock ($M)": f"{isolated_out_shock_m:+,.1f}M",
            "GDP ($B) [P10–P90]": f"${r['Direct_GDP']/1e9:.2f}B [${mc_t['Direct_GDP_p10']/1e9:.2f}B–${mc_t['Direct_GDP_p90']/1e9:.2f}B]",
            "Isolated GDP Shock ($M)": f"{isolated_gdp_shock_m:+,.1f}M",
            "Isolated NCI Margin Shock ($M)": f"{isolated_nci_shock_m:+,.1f}M",
        }
        if df_dyn_a is not None:
            r_a = df_dyn_a.iloc[idx_r]
            delta_gdp_ba_m = (float(r["Direct_GDP"]) - float(r_a["Direct_GDP"])) / 1e6
            delta_nci_ba_m = (float(r["Net_Cash_Income"]) - float(r_a["Net_Cash_Income"])) / 1e6
            delta_dscr_ba = float(r["DSCR"]) - float(r_a["DSCR"])
            row_dict["Δ vs. Scen A (GDP / NCI / DSCR)"] = f"{delta_gdp_ba_m:+,.1f}M GDP | {delta_nci_ba_m:+,.1f}M NCI | {delta_dscr_ba:+.2f}x"
        row_dict.update({
            "Statutory BRM Cash ($M)": (
                f"+${auto_brm_cash_r:,.1f}M (β={auto_brm_beta_r:.1f}%, Ref={olympic_ref_r:.0f}%)"
                if auto_brm_toggle
                else "Off"
            ),
            "Debt Repriced (Θ_t) & Eff. Rate": f"{r['Debt_Repricing_Share_Pct']:.0f}% ({r['Effective_Debt_Rate_Pct']:.2f}%)",
            "CapEx Crowd-Out (%)": f"{r['CapEx_CrowdOut_Pct']:+.2f}%",
            "Capital Stock ($B)": f"${r['Capital_Stock']/1e9:.2f}B",
            "DSCR [P10–P90]": f"{r['DSCR']:.2f}x (Δ {r['DSCR']-b_r['DSCR']:+.2f}x) [{mc_t['DSCR_p10']:.2f}x–{mc_t['DSCR_p90']:.2f}x]",
            "FSF (%)": f"{r['FSF']:.2f}% (Δ {r['FSF']-b_r['FSF']:+.2f} pp)",
            "LRF [P10–P90]": f"{r['LRF']:.2f} [{mc_t['LRF_p10']:.2f}–{mc_t['LRF_p90']:.2f}]",
            "Distress Prob (P_t)": f"{r.get('Distress_Prob_Pct', 8.3):.1f}% (φ={r.get('Phi_Debt_Regime', 0.58):.2f})",
            "Effective Tariff Elasticity (ε_t)": f"{r['Effective_Elasticity']:.3f}" if tariff_pct > 0 else "—",
            "Nerlove Convergence Ω(t)": f"{r['Nerlove_Omega_Pct']:.1f}%" if tariff_pct > 0 else "—",
        })
        table_rows.append(row_dict)
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Sister Engine 7A: Short-Term Net Cash Income (NCI) & Realized Net Income (RNI) Bridge (`scripts/forecasting_engine.py`)
    total_yr1_farm_rev_shock_pct = (
        ((-price_shock_info["effective_price_drop_pct"] + (yr1_tariff_pct_output_change * 100.0)) * active_tariff_params["weight"])
        + direct_rev_shock_pct
    )
    with st.expander("🔗 Sister Engine 7A: Short-Term Net Cash Income (NCI) & AgriStability / Agricorp RMP Bridge (`scripts/forecasting_engine.py`)", expanded=False):
        nci_bridge = run_short_term_nci_bridge(
            region=region,
            grain_price_shock_pct=total_yr1_farm_rev_shock_pct,
            livestock_price_shock_pct=total_yr1_farm_rev_shock_pct,
            fuel_shock_pct=shock_fuel,
            fert_shock_pct=shock_fert,
            wage_shock_pct=shock_wages,
            general_inflation_pct=(shock_chem + shock_seed + shock_feed) / 3.0,
            interest_shock_bps=shock_ir_bps,
        )
        if nci_bridge:
            nb1, nb2, nb3, nb4 = st.columns(4)
            nb1.metric("Trend Baseline NCI ($M)", f"${nci_bridge['base_nci_m']:,.0f}M")
            nb2.metric("Shocked Short-Term NCI ($M)", f"${nci_bridge['det_nci_m']:,.0f}M", delta=f"{nci_bridge['det_nci_m']-nci_bridge['base_nci_m']:+,.0f}M")
            nb3.metric("Shocked Realized Net Income ($M)", f"${nci_bridge['det_rni_m']:,.0f}M", delta=f"{nci_bridge['det_rni_m']-nci_bridge['base_rni_m']:+,.0f}M")
            nb4.metric(
                "Empirical BRM Entitlement ($M)",
                f"${nci_bridge['brm_auto_payout_m']:+,.1f}M",
                delta=f"Yr 1 Cash: ${nci_bridge.get('brm_yr1_cash_payout_m', 0.0):+,.1f}M | Yr 2 Lag: ${nci_bridge.get('brm_yr2_lagged_payout_m', 0.0):+,.1f}M",
                help=(
                    f"Effective Marginal BRM Payout Rate β_BRM = {nci_bridge.get('beta_brm_effective', 0.0)*100:.1f}% "
                    f"(Tier: {nci_bridge.get('brm_severity_tier', 'none')}). Reflects AgriStability allowable expense rules "
                    f"(interest rate spikes are non-allowable, λ_interest=0.0) plus Ontario Agricorp RMP & SDRM."
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
        Mapped Primary Agriculture commodities cover **`${fcr_info['mapped_sum_m']:,.1f}M`** out of **`${fcr_info['total_market_fcr_m']:,.1f}M`**
        in total {region} market receipts (**`{fcr_info['mapped_coverage_pct']:.1f}%` empirical coverage**).
        **Nursery, Floriculture & Sod (`HS Ch. 06`)** and **Greenhouse Vegetables (`NAICS 1114`)** are modeled strictly as **Primary Agriculture**.
        """
    )

    # Build 18-sector dynamic tariff vulnerability table for the active tariff rate (or 25% benchmark if slider is 0)
    eval_tariff_pct = tariff_pct if tariff_pct > 0 else 25.0
    st.markdown(f"#### 18-Sector Primary Agriculture Tariff Exposure & Nerlove Trajectory (Evaluated at `{eval_tariff_pct:.0f}%` U.S. Section 338 Tariff in {region})")

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

        # Output response at Year 1, Year 5, Year 10
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
            "Direct U.S. Share (s_US)": p["us_price_share"],
            "LOP Spillover (ψ_LOP)": p["us_lop_spillover"],
            "Pass-Through (θ_PT)": p["pass_through"],
            "Eff. Farm Price Drop (%)": -ps_comm["effective_price_drop_pct"],
            "Year 1 Output Δ (%)": dq_yr1_pct,
            "Year 5 Output Δ (%)": dq_yr5_pct,
            "Year 10 Output Δ (%)": dq_yr10_pct,
            "Year 1 Physical Output Δ ($M)": fcr_val_m * (dq_yr1_pct / 100.0),
            "Year 1 Total FCR / Margin Loss ($M)": fcr_val_m * (yr1_tot_rev_pct / 100.0),
            "Year 10 Total FCR / Output Loss ($M)": fcr_val_m * (yr10_tot_rev_pct / 100.0),
        })

    df_tariff_18 = pd.DataFrame(tariff_rows).sort_values("Year 10 Total FCR / Output Loss ($M)", ascending=True)
    st.dataframe(
        df_tariff_18.style.format({
            f"{fcr_year} {region} FCR ($M)": "${:,.1f}M",
            "Share of Market FCR (%)": "{:.2f}%",
            "Direct U.S. Share (s_US)": "{:.2f}",
            "LOP Spillover (ψ_LOP)": "{:.2f}",
            "Pass-Through (θ_PT)": "{:.2f}",
            "Eff. Farm Price Drop (%)": "{:+.1f}%",
            "Year 1 Output Δ (%)": "{:+.1f}%",
            "Year 5 Output Δ (%)": "{:+.1f}%",
            "Year 10 Output Δ (%)": "{:+.1f}%",
            "Year 1 Physical Output Δ ($M)": "${:+,.1f}M",
            "Year 1 Total FCR / Margin Loss ($M)": "${:+,.1f}M",
            "Year 10 Total FCR / Output Loss ($M)": "${:+,.1f}M",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Sister Engine 2: Live SQLite DB (`trade_exposure.db`) & Counter-Tariffs (`canada_counter_tariffs_sept8_2026.csv`)
    st.markdown("---")
    trade_db = load_section338_and_counter_tariff_data(region=region)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### 🔎 U.S. Section 338 Historical Statutory Tariff Lines (`trade_exposure.db`)")
        if trade_db["has_db"] and not trade_db["section338_df"].empty:
            st.dataframe(trade_db["section338_df"], use_container_width=True, hide_index=True, height=320)
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
            st.dataframe(ct_df, use_container_width=True, hide_index=True, height=280)


# ==============================================================================
# TAB 4: FARM-GATE TRANSLATOR ("AVERAGE PER FARM" + 26 OMAFA PUB 60 / CEA BUDGETS)
# ==============================================================================
with tab4:
    st.subheader("Farm-Gate Impact Translator: StatCan 'Average Per Farm' & 26 Ontario Enterprise Budgets")

    fg_col1, fg_col2 = st.columns([1.0, 1.15])

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
            brm_subsidy_dollars=brm_offset_dollars,
        )
        if avg_farm_res:
            st.markdown(
                f"""
                - **NAICS Farm Type:** `{avg_farm_res['farm_type']}` ({avg_farm_res['year']} StatCan Benchmark)
                - **Farms Reporting in {avg_farm_res['region']}:** `{avg_farm_res['farms_reporting']:,}` farms
                - **Baseline Operating Revenue / Farm:** `${avg_farm_res['base_rev_per_farm']:,.0f}` → **Shocked:** `${avg_farm_res['new_rev_per_farm']:,.0f}` (`${avg_farm_res['delta_rev_per_farm']:+,.0f}/farm`)
                - **Baseline Operating Expenses / Farm:** `${avg_farm_res['base_exp_per_farm']:,.0f}` → **Shocked:** `${avg_farm_res['new_exp_per_farm']:,.0f}` (`${avg_farm_res['delta_exp_per_farm']:+,.0f}/farm`)
                - **Net Operating Income (NOI) / Farm:** `${avg_farm_res['base_noi_per_farm']:,.0f}` → **`${avg_farm_res['new_noi_per_farm']:,.0f}` (`${avg_farm_res['delta_noi_per_farm']:+,.0f}/farm`)**
                """
            )
            st.dataframe(
                pd.DataFrame(avg_farm_res["line_items"]).style.format({
                    "Baseline ($/farm)": "${:,.0f}",
                    "Shocked ($/farm)": "${:,.0f}",
                    "Delta ($/farm)": "${:+,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with fg_col2:
        st.markdown("#### 2. OMAFA Publication 60 (2025/2026), OAG, OPVG, OPB & NAICS 1114 Per-Acre Calculator")
        ent_keys = list(ENTERPRISE_BUDGETS.keys())
        # Auto-select a matching default enterprise based on farm_type
        default_ent_key = "soybeans_rr_notill_2026"
        if "1114" in farm_type:
            default_ent_key = "gh_tomatoes_acre_2025"
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

        # Enterprise-specific tariff price pass-through (Fixes Audit Finding #5)
        ent_tariff_comm = get_enterprise_tariff_commodity(sel_ent_key)
        if tariff_pct > 0:
            if tariff_commodity == "All Commodities (Provincial FCR Weighted)" or tariff_commodity == ent_tariff_comm:
                ent_t_cfg = get_commodity_tariff_params(ent_tariff_comm, region=region)
                ent_t_shock = compute_effective_tariff_price_shock(
                    tariff_pct=tariff_pct,
                    pass_through=ent_t_cfg["pass_through"],
                    us_price_share=ent_t_cfg["us_price_share"],
                    us_lop_spillover=ent_t_cfg["us_lop_spillover"],
                )
                ent_tariff_price_drop_pct = float(ent_t_shock["effective_price_drop_pct"])
            else:
                ent_tariff_price_drop_pct = float(price_shock_info["effective_price_drop_pct"]) * float(active_tariff_params["weight"])
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
            - **Operating Cost / Acre:** `${ent_sim['base_op_ac']:,.2f}` → **`${ent_sim['new_op_ac']:,.2f}/ac`** (`${ent_sim['delta_op_ac']:+,.2f}/ac` / `{ent_sim['pct_op_increase']:+.1f}%`)
            - **Net Margin / Acre (After Overhead):** `${ent_sim['base_net_margin_ac']:,.2f}` → **`${ent_sim['new_net_margin_ac']:,.2f}/ac`** (`${ent_sim['delta_net_margin_ac']:+,.2f}/ac`)
            - **Whole-Enterprise ({ent_sim['acres']:,.0f} {ent_sim['unit_name']}s) Net Margin Impact:** **`${ent_sim['farm_delta_net_margin']:+,.0f}`** (Baseline: `${ent_sim['farm_base_net_margin']:,.0f}` → New: `${ent_sim['farm_new_net_margin']:,.0f}`)
            """
        )
        st.dataframe(
            pd.DataFrame(ent_sim["line_items"]).style.format({
                "Baseline ($/ac)": "${:,.2f}",
                "Shocked ($/ac)": "${:,.2f}",
                "Delta ($/ac)": "${:+,.2f}",
                "Whole Farm Delta ($)": "${:+,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Sister Engine 3: StatCan FIPI (18-10-0258-01) & Live Commodity Futures Signals
    st.markdown("---")
    st.markdown("#### 📡 Sister Engine 3: StatCan FIPI (`18-10-0258-01`) Input Price Index & Live Commodity Futures")
    fipi_kpis = load_fipi_input_cost_kpis(region=region)
    if fipi_kpis:
        fipi_cols = st.columns(len(fipi_kpis))
        for idx, fk in enumerate(fipi_kpis):
            yoy_str = f"{fk['yoy_change']:+.1f}% YoY" if fk.get("yoy_change") is not None else "N/A"
            fipi_cols[idx].metric(
                f"{fk['category']} ({fk['quarter_label']})",
                f"{fk['value']:.1f}",
                delta=yoy_str,
                delta_color="inverse",
            )
    if not df_mkt.empty:
        st.dataframe(df_mkt, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 5: ECONOMETRIC CALIBRATION MATRICES & EXECUTIVE BRIEFINGS
# ==============================================================================
with tab5:
    st.subheader("Verified Econometric Calibration Matrices & Executive Report Exports")
    st.markdown(
        """
        Every parameter in this simulator is grounded in official Statistics Canada tables, OMAFA Publication 60
        worksheets, and peer-reviewed agricultural economics literature—with **zero contrived heuristics**.
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
                "ε_SR (Point)": v["e_sr"],
                "ε_SR (P10–P90)": f"{v['e_sr_p10_p90'][0]:.2f} to {v['e_sr_p10_p90'][1]:.2f}",
                "ε_LR (Point)": v["e_lr"],
                "ε_LR (P10–P90)": f"{v['e_lr_p10_p90'][0]:.2f} to {v['e_lr_p10_p90'][1]:.2f}",
                "Nerlove λ": v["lam"],
                "Pass-Through θ_PT": v["pass_through"],
                "U.S. Share CAN": v["us_price_share"],
                "U.S. Share ON": v["us_price_share_on"],
                "LOP Spillover ψ_LOP": v["us_lop_spillover"],
                "Citations": v["citations"],
            })
        st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

    with st.expander("2. Empirical 5x5 Stochastic Variance-Covariance Matrix & Provincial TFP Decomposition", expanded=False):
        mc_c1, mc_c2 = st.columns(2)
        labels_5 = ["1. FIPI (% YoY)", "2. TFP (% YoY)", "3. Interest Rate (Δ pp)", "4. Net Cash Income (% YoY)", "5. Farm CapEx (% YoY)"]
        with mc_c1:
            st.markdown(f"**Correlation Matrix $\\mathbf{{R}}$ ({mc_info['region']} — {mc_info['basis']}):**")
            st.dataframe(pd.DataFrame(mc_info["corr"], index=labels_5, columns=labels_5).style.format("{:.2f}"), use_container_width=True)
        with mc_c2:
            st.markdown(f"**Variance-Covariance Matrix $\\boldsymbol{{\\Sigma}}$ ({mc_info['region']} — {mc_info['basis']}):**")
            st.dataframe(pd.DataFrame(mc_info["cov"], index=labels_5, columns=labels_5).style.format("{:.3f}"), use_container_width=True)

        st.markdown("**Darku, Malla & Tran (2016) Provincial TFP Decomposition (Technical Change vs. Scale Effects):**")
        st.dataframe(pd.DataFrame(PROVINCIAL_TFP_DECOMPOSITION), use_container_width=True, hide_index=True)

    with st.expander("3. Sister Engine 7B: Export Executive Briefing (.docx), Board Deck (.pptx) & Audit CSV", expanded=True):
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            word_buf = generate_simulator_word_briefing(
                region=region,
                farm_type=farm_type,
                preset_name=preset_name,
                df_dyn=df_dyn,
                mc_summary=mc_summary,
                abp_res=abp_res,
                avg_farm_res=avg_farm_res,
                ent_sim=ent_sim,
                base_output_dollars=base_output_dollars,
                base_gdp_dollars=base_gdp_dollars,
                combined_yr1_output_impact_m=combined_yr1_output_impact_m,
                combined_yr1_gdp_impact_m=combined_yr1_gdp_impact_m,
                combined_yr1_jobs_impact=combined_yr1_jobs_impact,
                tariff_pct=tariff_pct,
                tariff_commodity=tariff_commodity,
            )
            if word_buf:
                st.download_button(
                    label="📄 Download Executive Briefing (.docx)",
                    data=word_buf,
                    file_name=f"OFA_Executive_Briefing_{region}_{farm_type[:15].strip()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
        with dl_c2:
            pptx_buf = generate_simulator_pptx_deck(
                region=region,
                farm_type=farm_type,
                preset_name=preset_name,
                df_dyn=df_dyn,
                abp_res=abp_res,
                avg_farm_res=avg_farm_res,
                ent_sim=ent_sim,
                base_output_dollars=base_output_dollars,
                base_gdp_dollars=base_gdp_dollars,
                combined_yr1_output_impact_m=combined_yr1_output_impact_m,
                combined_yr1_gdp_impact_m=combined_yr1_gdp_impact_m,
                combined_yr1_jobs_impact=combined_yr1_jobs_impact,
            )
            if pptx_buf:
                st.download_button(
                    label="📊 Download Board Slide Deck (.pptx)",
                    data=pptx_buf,
                    file_name=f"OFA_Board_Deck_{region}_{farm_type[:15].strip()}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
        with dl_c3:
            csv_bytes = df_dyn.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download 10-Year Trajectory Audit (.csv)",
                data=csv_bytes,
                file_name=f"OFA_Growth_Risk_Trajectory_{region}_{farm_type[:15].strip()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

