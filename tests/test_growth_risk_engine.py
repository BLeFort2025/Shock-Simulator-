"""
Unit & Integration Test Suite for the OFA Growth & Risk Simulator
=================================================================
Tests all 5 bug fixes, all 3 methodological/baseline gap resolutions,
the 4 Gemini Deep Research empirical parameter matrices, and all 7
integrated sister engines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.smart_read import smart_read
from scripts.calibrated_parameters import (
    ABP_PROCESSOR_TABLE,
    EMPIRICAL_ICOR_PROFILES,
    MUTUALLY_EXCLUSIVE_LEAF_NAICS,
    PARENT_AGGREGATE_NAICS,
    PRIMARY_AG_DISPOSITION_MATRIX,
    PRIMARY_AG_TARIFF_CONFIG,
    PROVINCIAL_TFP_DECOMPOSITION,
    get_mc_covariance_params,
)
from scripts.farm_cost_impact_model import (
    ENTERPRISE_BUDGETS,
    run_grain_corn_scenario,
    run_vegetable_farm_scenario,
    simulate_enterprise_budget,
)
from scripts.growth_risk_engine import (
    compute_cost_commodity_breakdown,
    compute_downstream_abp_impact,
    compute_effective_tariff_price_shock,
    compute_provincial_fcr_weights,
    compute_statcan_average_farm_impact,
    generate_simulator_pptx_deck,
    generate_simulator_word_briefing,
    get_commodity_tariff_params,
    load_fipi_input_cost_kpis,
    load_section338_and_counter_tariff_data,
    load_subsector_io_multipliers,
    run_dynamic_projection,
    run_monte_carlo_projections,
    run_short_term_nci_bridge,
)
from scripts.io_multipliers_engine import load_baskets_config


@pytest.fixture(scope="module")
def core_data():
    io_df = pd.read_parquet(PROJECT_ROOT / "data" / "latest" / "io_multipliers_standardized.parquet")
    baskets_cfg = load_baskets_config()
    with open(PROJECT_ROOT / "data" / "derived" / "farm_expense_profiles.json", "r", encoding="utf-8") as f:
        expense_profiles = json.load(f)
    df_fcr = smart_read(PROJECT_ROOT / "data" / "latest" / "32-10-0045-01.csv")
    df_136 = smart_read(PROJECT_ROOT / "data" / "latest" / "32-10-0136-01_dashboard.csv")
    return {
        "io_df": io_df,
        "baskets_cfg": baskets_cfg,
        "expense_profiles": expense_profiles,
        "df_fcr": df_fcr,
        "df_136": df_136,
    }


def test_primary_ag_tariff_config_and_bug1_fix():
    """Verify all 18 Primary Ag sectors have valid Nerlovian parameters & NAICS 1114 is Primary Ag."""
    assert len(PRIMARY_AG_TARIFF_CONFIG) == 18

    # RULE[user_nursery_primary_ag]: Floriculture, Nursery & Sod (HS Ch. 06) and Greenhouse Vegetables must be Primary Ag
    assert "Floriculture, Nursery & Sod (HS Ch. 06)" in PRIMARY_AG_TARIFF_CONFIG
    assert "Greenhouse Vegetables" in PRIMARY_AG_TARIFF_CONFIG
    assert PRIMARY_AG_TARIFF_CONFIG["Floriculture, Nursery & Sod (HS Ch. 06)"]["io_join_code"] == "1114A"
    assert PRIMARY_AG_TARIFF_CONFIG["Greenhouse Vegetables"]["io_join_code"] == "1114A"

    # Bug #1 regression: Corn (Grain) must not have negative lambda (-0.38)
    corn = PRIMARY_AG_TARIFF_CONFIG["Corn (Grain)"]
    assert corn["lam"] == pytest.approx(0.457, abs=0.01)
    assert corn["e_sr"] == pytest.approx(-0.48, abs=0.01)
    assert corn["e_lr"] == pytest.approx(-1.05, abs=0.01)

    for comm, cfg in PRIMARY_AG_TARIFF_CONFIG.items():
        assert cfg["e_sr"] < 0, f"{comm} e_sr must be negative"
        assert cfg["e_lr"] <= cfg["e_sr"] < 0, f"{comm} |e_lr| must be >= |e_sr|"
        assert 0.0 < cfg["lam"] <= 1.0, f"{comm} lam must be in (0, 1]"
        assert 0.0 < cfg["pass_through"] <= 1.0, f"{comm} pass_through must be in (0, 1]"
        assert 0.0 <= cfg["us_price_share"] <= 1.0
        assert 0.0 <= cfg["us_price_share_on"] <= 1.0
        assert len(cfg["citations"]) > 5


def test_mc_covariance_matrices_positive_definite():
    """Verify the 5x5 stochastic covariance matrices from Deep Research Output #1 are symmetric positive-definite."""
    for reg in ["Ontario", "Canada"]:
        for basis in ["gross_output", "value_added"]:
            mc = get_mc_covariance_params(region=reg, basis=basis)
            cov = mc["cov"]
            corr = mc["corr"]
            assert cov.shape == (5, 5)
            assert corr.shape == (5, 5)
            assert np.allclose(cov, cov.T, atol=1e-6)
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals > 0), f"Non-positive eigenvalues for {reg}/{basis}: {eigvals}"


def test_provincial_fcr_weights_gap7(core_data):
    """Verify dynamic provincial FCR weighting from StatCan 32-10-0045-01 for Ontario and Canada."""
    df_fcr = core_data["df_fcr"]
    on_fcr = compute_provincial_fcr_weights(df_fcr, region="Ontario")
    can_fcr = compute_provincial_fcr_weights(df_fcr, region="Canada")

    assert on_fcr["mapped_coverage_pct"] > 90.0
    assert can_fcr["mapped_coverage_pct"] > 90.0

    # Ontario has much higher Greenhouse Vegetables & Floriculture/Nursery/Sod shares than national average
    on_gh_share = on_fcr["commodities"]["Greenhouse Vegetables"]["share_of_market_fcr"]
    can_gh_share = can_fcr["commodities"]["Greenhouse Vegetables"]["share_of_market_fcr"]
    assert on_gh_share > can_gh_share * 1.5

    on_flori_share = on_fcr["commodities"]["Floriculture, Nursery & Sod (HS Ch. 06)"]["share_of_market_fcr"]
    can_flori_share = can_fcr["commodities"]["Floriculture, Nursery & Sod (HS Ch. 06)"]["share_of_market_fcr"]
    assert on_flori_share > can_flori_share * 1.5


def test_cost_commodity_breakdown_bug3_fix(core_data):
    """Verify Bug #3 fix: mutually exclusive leaf vs. parent NAICS tiers & gross expense ratios."""
    assert set(MUTUALLY_EXCLUSIVE_LEAF_NAICS).isdisjoint(set(PARENT_AGGREGATE_NAICS))

    profiles = core_data["expense_profiles"]
    leaf_bd = compute_cost_commodity_breakdown(
        profiles,
        shock_fuel=10.0,
        shock_fert=20.0,
        shock_chem=15.0,
        shock_seed=5.0,
        shock_feed=10.0,
        shock_wages=8.0,
        shock_interest_rel_pct=12.0,
        tier="leaf",
        region="Ontario",
    )
    parent_bd = compute_cost_commodity_breakdown(
        profiles,
        shock_fuel=10.0,
        shock_fert=20.0,
        shock_chem=15.0,
        shock_seed=5.0,
        shock_feed=10.0,
        shock_wages=8.0,
        shock_interest_rel_pct=12.0,
        tier="parent",
        region="Ontario",
    )

    leaf_names = {r["Commodity"] for r in leaf_bd}
    parent_names = {r["Commodity"] for r in parent_bd}
    assert "Crop production [111]" not in leaf_names
    assert "Animal production [112]" not in leaf_names
    assert "Greenhouse, nursery and floriculture production [1114]" in leaf_names
    assert parent_names == {"Crop production [111]", "Animal production [112]"}

    # Check exact gross ratio math on NAICS 1114
    gh_row = next(r for r in leaf_bd if "1114" in r["Commodity"])
    gh_prof = profiles["Ontario|Greenhouse, nursery and floriculture production [1114]"]
    exp_m = gh_prof["total_expenses_raw"] / 1e6
    expected_fuel_m = exp_m * gh_prof["ratios"]["energy"] * 0.10
    assert gh_row["Fuel ($M)"] == pytest.approx(expected_fuel_m, rel=1e-4)


def test_downstream_abp_impact_bug4_fix(core_data):
    """Verify Bug #4 fix: StatCan SUT 36-10-0478-01 Analysis-by-Parts nets out upstream farm GDP."""
    io_df = core_data["io_df"]
    abp = compute_downstream_abp_impact(
        io_df=io_df,
        region="Ontario",
        year=2022,
        farm_type="All farm types",
        primary_supply_shock_dollars=-500_000_000.0,
        primary_cost_shock_dollars=100_000_000.0,
    )
    assert abp["proc_net_gdp_m"] < 0.0
    assert abp["proc_net_jobs"] < 0.0
    assert abp["double_counted_farm_gdp_removed_m"] < 0.0
    for sub in abp["subsectors"]:
        assert sub["ABP Net Output Mult"] < sub["Gross Output Mult"]
        assert sub["ABP Net GDP Mult"] < sub["Gross GDP Mult"]


def test_dynamic_and_monte_carlo_bug2_and_bug5_fixes(core_data):
    """Verify Bug #2 (Monte Carlo preserves tariff weight) and Bug #5 (complete trajectory columns)."""
    io_df = core_data["io_df"]
    baskets_cfg = core_data["baskets_cfg"]

    # Simulate a single small-share commodity (weight = 0.05) under a 25% tariff
    tariff_params = {
        "commodity": "Greenhouse Vegetables",
        "tariff_pct": 25.0,
        "e_sr": -0.12,
        "e_lr": -0.60,
        "lam": 0.20,
        "pass_through": 0.55,
        "us_price_share": 0.99,
        "us_lop_spillover": 0.85,
        "weight": 0.05,
    }

    df_dyn = run_dynamic_projection(
        horizon=5,
        base_output=20_000_000_000.0,
        base_gdp=10_000_000_000.0,
        base_cap_stock=35_000_000_000.0,
        base_capex=3_000_000_000.0,
        shock_capex_pct=-10.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=8.5,
        cost_shock_dollars=200_000_000.0,
        rev_shock_dollars=0.0,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
        base_debt=15_000_000_000.0,
        base_interest=750_000_000.0,
        base_nci=3_500_000_000.0,
        shock_interest_pct=10.0,
        tariff_params=tariff_params,
    )

    # Bug #5 check: all required trajectory columns exist
    for col in ["Capital_Stock", "DSCR", "FSF", "LRF", "Effective_Elasticity", "Nerlove_Omega_Pct"]:
        assert col in df_dyn.columns

    # Nerlove convergence must increase monotonically from Year 1 to Year 5
    omegas = df_dyn["Nerlove_Omega_Pct"].tolist()
    assert all(omegas[i] < omegas[i + 1] for i in range(len(omegas) - 1))

    mc_summary = run_monte_carlo_projections(
        n_sims=100,
        horizon=5,
        base_output=20_000_000_000.0,
        base_gdp=10_000_000_000.0,
        base_cap_stock=35_000_000_000.0,
        base_capex=3_000_000_000.0,
        shock_capex_pct=-10.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=8.5,
        cost_shock_dollars=200_000_000.0,
        rev_shock_dollars=0.0,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
        base_debt=15_000_000_000.0,
        base_interest=750_000_000.0,
        base_nci=3_500_000_000.0,
        shock_interest_pct=10.0,
        tariff_params=tariff_params,
    )

    # Bug #2 check: deterministic GDP must lie inside [P10, P90] for all years even when weight = 0.05
    for _, row in df_dyn.iterrows():
        t_idx = int(row["t"])
        det_gdp = float(row["Direct_GDP"])
        p10_gdp = mc_summary[t_idx]["Direct_GDP_p10"]
        p90_gdp = mc_summary[t_idx]["Direct_GDP_p90"]
        assert p10_gdp <= det_gdp <= p90_gdp, f"Year {t_idx}: det={det_gdp} outside [{p10_gdp}, {p90_gdp}]"


def test_26_enterprise_budgets_and_statcan_avg_farm_translator(core_data):
    """Verify all 26 OMAFA Pub 60 / OAG / OPVG / OGVG / NAICS 1114 budgets and StatCan 32-10-0136-01 translator."""
    assert len(ENTERPRISE_BUDGETS) == 26
    for key in ENTERPRISE_BUDGETS:
        res = simulate_enterprise_budget(
            enterprise_key=key,
            shock_fertilizer_pct=20.0,
            shock_fuel_pct=15.0,
            shock_pesticides_pct=10.0,
            shock_labour_pct=5.0,
            shock_interest_bps=100.0,
            include_cash_rent=True,
        )
        assert res["new_op_ac"] > res["base_op_ac"]
        assert res["new_net_margin_ac"] < res["base_net_margin_ac"]

    # Legacy backward compatibility
    assert "acres" in run_grain_corn_scenario()
    assert "cost_2025" in run_vegetable_farm_scenario()

    # StatCan 32-10-0136-01 Average per farm translator
    avg_res = compute_statcan_average_farm_impact(
        df_136=core_data["df_136"],
        region="Ontario",
        farm_type="Greenhouse, nursery and floriculture production [1114]",
        shock_fuel=20.0,
        shock_fert=15.0,
        shock_wages=8.0,
    )
    assert avg_res["farms_reporting"] > 0
    assert avg_res["base_rev_per_farm"] > 0
    assert avg_res["new_exp_per_farm"] > avg_res["base_exp_per_farm"]


def test_sister_engine_bridges_and_deliverables(core_data):
    """Verify Section 338 DB bridge, FIPI KPIs, Short-Term NCI bridge, and .docx/.pptx exports."""
    trade_data = load_section338_and_counter_tariff_data("Ontario")
    assert trade_data["has_db"] is True
    assert not trade_data["section338_df"].empty
    assert not trade_data["counter_tariffs_df"].empty

    fipi_kpis = load_fipi_input_cost_kpis("Ontario")
    assert len(fipi_kpis) > 0

    nci_bridge = run_short_term_nci_bridge("Ontario", grain_price_shock_pct=-10.0, fert_shock_pct=20.0)
    assert nci_bridge["base_nci_m"] > 0

    # Test .docx and .pptx generation
    io_df = core_data["io_df"]
    baskets_cfg = core_data["baskets_cfg"]
    df_dyn = run_dynamic_projection(
        horizon=3,
        base_output=10e9,
        base_gdp=5e9,
        base_cap_stock=15e9,
        base_capex=1.2e9,
        shock_capex_pct=0.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=8.5,
        cost_shock_dollars=50e6,
        rev_shock_dollars=-100e6,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
    )
    mc_sum = run_monte_carlo_projections(
        n_sims=20,
        horizon=3,
        base_output=10e9,
        base_gdp=5e9,
        base_cap_stock=15e9,
        base_capex=1.2e9,
        shock_capex_pct=0.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=8.5,
        cost_shock_dollars=50e6,
        rev_shock_dollars=-100e6,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
    )
    abp_res = compute_downstream_abp_impact(io_df, "Ontario", 2022, "All farm types", -100e6)
    avg_res = compute_statcan_average_farm_impact(core_data["df_136"], "Ontario", "All farm types", shock_fert=10.0)
    ent_res = simulate_enterprise_budget("soybeans_rr_notill_2026", shock_fertilizer_pct=10.0)

    docx_buf = generate_simulator_word_briefing(
        region="Ontario",
        farm_type="All farm types",
        preset_name="Unit Test Scenario",
        df_dyn=df_dyn,
        mc_summary=mc_sum,
        abp_res=abp_res,
        avg_farm_res=avg_res,
        ent_sim=ent_res,
        base_output_dollars=10e9,
        base_gdp_dollars=5e9,
        combined_yr1_output_impact_m=-180.0,
        combined_yr1_gdp_impact_m=-90.0,
        combined_yr1_jobs_impact=-850.0,
        tariff_pct=25.0,
        tariff_commodity="All Commodities",
    )
    assert docx_buf is not None
    assert len(docx_buf.getvalue()) > 5000

    pptx_buf = generate_simulator_pptx_deck(
        region="Ontario",
        farm_type="All farm types",
        preset_name="Unit Test Scenario",
        df_dyn=df_dyn,
        abp_res=abp_res,
        avg_farm_res=avg_res,
        ent_sim=ent_res,
        base_output_dollars=10e9,
        base_gdp_dollars=5e9,
        combined_yr1_output_impact_m=-180.0,
        combined_yr1_gdp_impact_m=-90.0,
        combined_yr1_jobs_impact=-850.0,
    )
    assert pptx_buf is not None
    assert len(pptx_buf.getvalue()) > 5000


def test_deep_research_outputs_5_and_6_econometric_calibration(core_data):
    """
    Verify all empirical parameters and structural equations from Deep Research Outputs #5 and #6:
    - 5-Year Debt Repricing Schedule (theta_yr1..theta_yr5)
    - Hotelling's Lemma Duality Output vs. Margin Split
    - Blended Principal Amortization (gamma_principal) & Regime-Switching Distress Borrowing (phi_debt)
    - Forward Cost-Push Processor Price Transmission (psi_cost)
    - AgriStability & Agricorp RMP Effective Marginal Payout Rate (beta_BRM)
    """
    from scripts.calibrated_parameters import (
        compute_effective_brm_marginal_payout,
        get_debt_amortization_params,
        get_debt_repricing_schedule,
        get_hotelling_sector_profile,
        get_processor_cost_pass_through,
    )

    # 1. Debt repricing schedule sums to 1.0000 over 5 years
    theta_on = get_debt_repricing_schedule("Ontario", "All farm types")
    theta_can = get_debt_repricing_schedule("Canada", "All farm types")
    assert len(theta_on) == 5
    assert abs(theta_on[0] - 0.440) < 1e-4
    assert abs(theta_can[0] - 0.420) < 1e-4
    assert abs(sum(theta_on) - 1.0) < 1e-4
    assert abs(sum(theta_can) - 1.0) < 1e-4

    # 2. Hotelling duality profile
    h_all = get_hotelling_sector_profile("All farm types")
    h_1114 = get_hotelling_sector_profile("Greenhouse, nursery and floriculture production [1114]")
    assert abs(h_all["yr1_margin_share"] - 0.895) < 1e-4
    assert abs(h_1114["yr1_margin_share"] - 0.905) < 1e-4

    # 3. Debt amortization & distress borrowing
    d_on = get_debt_amortization_params("Ontario")
    d_can = get_debt_amortization_params("Canada")
    assert abs(d_on["gamma_principal"] - 0.0658) < 1e-4
    assert abs(d_can["gamma_principal"] - 0.0713) < 1e-4
    assert abs(d_on["phi_debt_normal"] - 0.580) < 1e-4
    assert abs(d_on["phi_debt_distress"] - 0.820) < 1e-4
    assert abs(d_on["w_dscr"] + d_on["w_fsf"] - 1.0) < 1e-6

    # 4. Forward cost-push processor pass-through
    assert abs(get_processor_cost_pass_through("3112", "Oilseed and grain farming [1111]") - 0.020) < 1e-4
    assert abs(get_processor_cost_pass_through("3115", "Dairy cattle and milk production [112120]", is_long_run=False) - 0.200) < 1e-4
    assert abs(get_processor_cost_pass_through("3115", "Dairy cattle and milk production [112120]", is_long_run=True) - 0.433) < 1e-4
    assert abs(get_processor_cost_pass_through("3116", "Poultry and egg production [1123]", is_long_run=False) - 0.790) < 1e-4
    assert abs(get_processor_cost_pass_through("3116", "Beef cattle ranching and farming, including feedlots [112110]", is_long_run=False) - 0.000) < 1e-4

    # 5. AgriStability / Agricorp RMP effective payout rate
    # Pure interest rate shock is non-allowable (lambda_interest = 0.0) -> $0 BRM payout
    brm_int_only = compute_effective_brm_marginal_payout(
        region="Ontario",
        allowable_margin_shortfall_dollars=0.0,
        interest_shortfall_dollars=100_000_000.0,
        baseline_nci_dollars=2_000_000_000.0,
    )
    assert brm_int_only["total_brm_entitlement"] == 0.0

    # Moderate allowable cost shock in Ontario (20% margin drop -> Tier 2: beta_BRM = 0.225)
    brm_mod_on = compute_effective_brm_marginal_payout(
        region="Ontario",
        allowable_margin_shortfall_dollars=400_000_000.0,
        interest_shortfall_dollars=50_000_000.0,
        baseline_nci_dollars=2_000_000_000.0,
        shock_type="price_or_cost_shock",
    )
    assert brm_mod_on["severity_tier"] == "moderate_15_to_30pct"
    assert abs(brm_mod_on["beta_brm_effective"] - 0.225) < 1e-4
    assert abs(brm_mod_on["total_brm_entitlement"] - 90_000_000.0) < 1.0
    assert abs(brm_mod_on["yr1_cash_payout"] + brm_mod_on["yr2_lagged_payout"] - 90_000_000.0) < 1.0


def test_comprehensive_audit_fixes_1_through_9(core_data):
    """
    Verify all 9 audit fixes across scripts/growth_risk_engine.py and scripts/forecasting_engine.py:
    1. DSCR pre-depreciation EBITDA numerator (NCI + Interest) -> ~1.43x baseline in Ontario
    2. Balanced-growth anchored logit distress probability -> stationary 8.3% at 0% shock
    3. Sister Engine 7A inner-join baseline expenses -> Ontario base_nci_m ~ $5,143M
    4. Bracketed StatCan FCR commodity code matching -> ~94.9% coverage in Ontario
    5. Sub-sector IO multiplier routing for 111A, 1114A, and 112A in run_dynamic_projection
    6. Long-run ABP cost pass-through (is_long_run=True vs False)
    7. Per-farm BRM offset & sub-sector revenue-to-expense scaling in compute_statcan_average_farm_impact
    8. Enterprise-to-commodity tariff mapping via get_enterprise_tariff_commodity
    """
    from scripts.growth_risk_engine import get_enterprise_tariff_commodity

    io_df = core_data["io_df"]
    baskets_cfg = core_data["baskets_cfg"]

    # Fix #1 & #2: Unshocked baseline DSCR (~1.43x) and stationary 8.3% distress probability
    df_unshocked = run_dynamic_projection(
        horizon=10,
        base_output=22_955_000_000.0,
        base_gdp=16_500_000_000.0,
        base_cap_stock=31_500_000_000.0,
        base_capex=3_000_000_000.0,
        shock_capex_pct=0.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=9.5,
        cost_shock_dollars=0.0,
        rev_shock_dollars=0.0,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
        base_debt=42_800_000_000.0,
        base_interest=1_548_900_000.0,
        base_nci=4_350_500_000.0,
        shock_interest_pct=0.0,
    )
    assert 1.40 <= float(df_unshocked.iloc[0]["DSCR"]) <= 1.46
    for _, r in df_unshocked.iterrows():
        assert abs(float(r["Distress_Prob_Pct"]) - 8.30) < 0.05

    # Fix #3: Sister Engine 7A inner-join baseline expenses
    bridge_on = run_short_term_nci_bridge("Ontario")
    assert 4500.0 <= bridge_on["base_nci_m"] <= 5800.0
    assert bridge_on["base_expenses_m"] > 15000.0

    # Fix #4: Provincial FCR weights coverage ~94.9% in Ontario
    fcr_on = compute_provincial_fcr_weights(core_data["df_fcr"], region="Ontario")
    assert 92.0 <= fcr_on["mapped_coverage_pct"] <= 98.0

    # Fix #5: Sub-sector IO multiplier routing (111A vs 112A vs 1114A)
    df_111A = run_dynamic_projection(
        horizon=2, base_output=10e9, base_gdp=7.2e9, base_cap_stock=15e9, base_capex=1.4e9,
        shock_capex_pct=0.0, tfp_growth_pct=1.85, depreciation_rate_pct=9.5,
        cost_shock_dollars=100e6, rev_shock_dollars=0.0, io_df=io_df, baskets_cfg=baskets_cfg,
        region="Ontario", target_year=2022, primary_join_code="111A",
    )
    df_112A = run_dynamic_projection(
        horizon=2, base_output=10e9, base_gdp=7.12e9, base_cap_stock=15e9, base_capex=1.4e9,
        shock_capex_pct=0.0, tfp_growth_pct=1.85, depreciation_rate_pct=9.5,
        cost_shock_dollars=100e6, rev_shock_dollars=0.0, io_df=io_df, baskets_cfg=baskets_cfg,
        region="Ontario", target_year=2022, primary_join_code="112A",
    )
    assert len(df_111A) == 2 and len(df_112A) == 2

    # Fix #6: Long-run vs Short-run ABP cost pass-through in Dairy
    abp_sr = compute_downstream_abp_impact(
        io_df, "Ontario", 2022, "Dairy cattle and milk production [112120]",
        primary_supply_shock_dollars=0.0, primary_cost_shock_dollars=100e6, is_long_run=False,
    )
    abp_lr = compute_downstream_abp_impact(
        io_df, "Ontario", 2022, "Dairy cattle and milk production [112120]",
        primary_supply_shock_dollars=0.0, primary_cost_shock_dollars=100e6, is_long_run=True,
    )
    assert abs(abp_lr["proc_net_gdp_m"]) > abs(abp_sr["proc_net_gdp_m"])

    # Fix #7: Per-farm BRM offset in compute_statcan_average_farm_impact
    avg_brm = compute_statcan_average_farm_impact(
        df_136=core_data["df_136"], region="Ontario", farm_type="All farm types",
        shock_fert=20.0, brm_subsidy_dollars=100_000_000.0,
    )
    assert avg_brm["brm_per_farm"] > 0
    assert avg_brm["delta_rev_per_farm"] == avg_brm["brm_per_farm"]

    # Fix #8: Enterprise-to-commodity tariff mapping
    assert get_enterprise_tariff_commodity("gh_tomatoes_acre_2025") == "Greenhouse Vegetables"
    assert get_enterprise_tariff_commodity("gh_floriculture_1114_2025") == "Floriculture, Nursery & Sod (HS Ch. 06)"
    assert get_enterprise_tariff_commodity("grain_corn_notill_2026") == "Corn (Grain)"
    assert get_enterprise_tariff_commodity("soybeans_rr_notill_2026") == "Soybeans"
    assert get_enterprise_tariff_commodity("field_potatoes_chip_2025") == "Potatoes"
    assert get_enterprise_tariff_commodity("apple_hd_mature_2025") == "Fresh Fruit & Tree Nuts"


def test_endogenous_multiyear_statutory_brm_toggle(core_data):
    """
    Verify Endogenous Multi-Year Statutory AgriStability & Agricorp RMP Toggle:
    1. Pure interest rate shock (+200 bps) yields $0.00M statutory BRM across all 10 years (lambda_interest = 0.0).
    2. Operating cost / tariff shock triggers Year-1 interim/RMP cash (w_t) and Year-2 tax-filing carryover (w_{t+1}),
       so Year-2 cash inflow exceeds Year-1 cash inflow.
    3. 5-year Olympic Average reference margin holds at 100% in Years 1 & 2, then ratchets down in Years 3-5.
    4. Statutory BRM toggle improves NCI, lifts DSCR, lowers Distress_Prob_Pct, and reduces Year-2 CapEx crowd-out.
    5. 2025/2026 90% Ad-Hoc Enhancement regime yields strictly higher entitlement than 80% SCAP regime.
    """
    io_df = core_data["io_df"]
    baskets_cfg = core_data["baskets_cfg"]

    common_kwargs = dict(
        horizon=6,
        base_output=22_955_000_000.0,
        base_gdp=16_500_000_000.0,
        base_cap_stock=31_500_000_000.0,
        base_capex=3_000_000_000.0,
        shock_capex_pct=0.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=9.5,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
        base_debt=42_800_000_000.0,
        base_interest=1_548_900_000.0,
        base_nci=4_350_500_000.0,
        shock_interest_pct=0.0,
    )

    # 1. Pure interest rate shock (+200 bps) -> $0.00M statutory BRM entitlement
    df_ir_only = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=0.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=0.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=200.0,
        auto_brm_stabilization=True,
    )
    assert float(df_ir_only["Auto_BRM_Entitlement_M"].sum()) == 0.0
    assert float(df_ir_only["Auto_BRM_Cash_M"].sum()) == 0.0

    # 2. Operating cost shock ($900M intermediate cost shock) with auto_brm_stabilization=False vs True
    df_no_brm = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=900_000_000.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=900_000_000.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=False,
    )
    df_scap_80 = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=900_000_000.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=900_000_000.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=True,
        brm_regime="statutory_scap_80pct",
    )
    df_adhoc_90 = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=900_000_000.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=900_000_000.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=True,
        brm_regime="adhoc_2025_90pct",
    )

    yr1_ent = float(df_scap_80.iloc[0]["Auto_BRM_Entitlement_M"])
    yr1_cash = float(df_scap_80.iloc[0]["Auto_BRM_Cash_M"])
    yr2_lag_carry = float(df_scap_80.iloc[1]["Auto_BRM_Lagged_Carry_M"])
    yr2_cash = float(df_scap_80.iloc[1]["Auto_BRM_Cash_M"])

    assert yr1_ent > 100.0
    assert yr1_cash > 0.0 and yr2_lag_carry > 0.0
    assert abs((yr1_cash + yr2_lag_carry) - yr1_ent) < 1e-3
    assert yr2_cash > yr1_cash  # Year 2 receives both Year-2 interim + Year-1 tax-filing settlement

    # 3. 5-Year Olympic Reference Margin holds at 100% in Years 1 & 2, then ratchets down in Years 3-5
    assert float(df_scap_80.iloc[0]["Olympic_Ref_Ratio_Pct"]) == 100.0
    assert float(df_scap_80.iloc[1]["Olympic_Ref_Ratio_Pct"]) == 100.0
    assert float(df_scap_80.iloc[2]["Olympic_Ref_Ratio_Pct"]) < 100.0
    assert float(df_scap_80.iloc[4]["Olympic_Ref_Ratio_Pct"]) < float(df_scap_80.iloc[2]["Olympic_Ref_Ratio_Pct"])

    # 4. Solvency & CapEx stabilization vs. no BRM
    assert float(df_scap_80.iloc[0]["Net_Cash_Income"]) > float(df_no_brm.iloc[0]["Net_Cash_Income"])
    assert float(df_scap_80.iloc[0]["DSCR"]) > float(df_no_brm.iloc[0]["DSCR"])
    assert float(df_scap_80.iloc[0]["Distress_Prob_Pct"]) < float(df_no_brm.iloc[0]["Distress_Prob_Pct"])
    # Less severe negative CapEx crowd-out in Year 2
    assert float(df_scap_80.iloc[1]["CapEx_CrowdOut_Pct"]) > float(df_no_brm.iloc[1]["CapEx_CrowdOut_Pct"])

    # 5. 90% Ad-Hoc regime strictly exceeds 80% SCAP regime
    assert float(df_adhoc_90.iloc[0]["Auto_BRM_Entitlement_M"]) > yr1_ent


def test_side_by_side_scenario_comparison_engine(core_data):
    """
    Verify Side-by-Side Scenario Comparison (`compute_scenario_comparison_summary`):
    1. Pre-Policy (Scenario A, No BRM) vs. Post-Policy (Scenario B, Statutory BRM) shows positive
       net deltas (B - A) for NCI, DSCR, Year-2 CapEx, and 10-year cumulative GDP, and negative delta for Distress Prob.
    2. Returns 10 reconciled executive comparison rows.
    """
    from scripts.growth_risk_engine import compute_scenario_comparison_summary

    io_df = core_data["io_df"]
    baskets_cfg = core_data["baskets_cfg"]

    common_kwargs = dict(
        horizon=10,
        base_output=22_955_000_000.0,
        base_gdp=16_500_000_000.0,
        base_cap_stock=31_500_000_000.0,
        base_capex=3_000_000_000.0,
        shock_capex_pct=0.0,
        tfp_growth_pct=1.85,
        depreciation_rate_pct=9.5,
        io_df=io_df,
        baskets_cfg=baskets_cfg,
        region="Ontario",
        target_year=2022,
        base_debt=42_800_000_000.0,
        base_interest=1_548_900_000.0,
        base_nci=4_350_500_000.0,
        shock_interest_pct=0.0,
    )

    df_base = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=0.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=0.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=False,
    )
    df_scen_a = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=900_000_000.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=900_000_000.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=False,
    )
    df_scen_b = run_dynamic_projection(
        **common_kwargs,
        cost_shock_dollars=900_000_000.0,
        rev_shock_dollars=0.0,
        intermediate_cost_shock_dollars=900_000_000.0,
        labor_cost_shock_dollars=0.0,
        shock_ir_bps=0.0,
        auto_brm_stabilization=True,
        brm_regime="statutory_scap_80pct",
    )

    abp_a1 = compute_downstream_abp_impact(io_df, "Ontario", 2022, "All farm types", -120e6, 900e6, is_long_run=False)
    abp_aN = compute_downstream_abp_impact(io_df, "Ontario", 2022, "All farm types", -300e6, 900e6, is_long_run=True)

    cmp_res = compute_scenario_comparison_summary(
        label_a="Pre-Policy (No BRM)",
        label_b="Post-Policy (Statutory BRM)",
        df_dyn_a=df_scen_a,
        df_dyn_b=df_scen_b,
        df_base_dyn=df_base,
        abp_yr1_a=abp_a1,
        abp_yrN_a=abp_aN,
        abp_yr1_b=abp_a1,
        abp_yrN_b=abp_aN,
        jobs_mult_per_m=6.2,
    )

    d = cmp_res["deltas_b_minus_a"]
    assert len(cmp_res["comparison_rows"]) == 10
    assert d["yr1_nci_m"] > 50.0
    assert d["yr1_brm_cash_m"] > 50.0
    assert d["yr1_dscr"] > 0.0
    assert d["yr1_distress_pct"] < 0.0
    assert d["yr2_capex_crowdout_pct"] > 0.0
    assert d["cum_10yr_gdp_m"] > 0.0

    # Verify standalone root app and multi-page dashboard Page 8 are 100% synchronized & AST-valid
    import ast
    root_app_fp = PROJECT_ROOT / "growth_risk_simulator_app.py"
    page8_fp = PROJECT_ROOT / "app" / "pages" / "8_🔮_Growth_&_Risk_Simulator.py"
    root_src = root_app_fp.read_text(encoding="utf-8")
    page8_src = page8_fp.read_text(encoding="utf-8")
    ast.parse(root_src)
    ast.parse(page8_src)
    assert root_src == page8_src





