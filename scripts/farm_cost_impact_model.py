"""
Ontario Farm Cost Impact Model v2
===================================
Rebuilt using OMAFRA Publication 60 (2025 Field Crop Budgets) per-acre costs
as the 2025 baseline. Scenario 1: 500 acres grain corn (no-till).

All 2025 baseline numbers are directly from Pub 60, page 12.
2026 estimates use validated escalation factors from World Bank / DTN / Ridgetown.
"""

import json
import os

# ============================================================
# OMAFRA PUB 60 - 2025 FIELD CROP BUDGET: GRAIN CORN (NO-TILL)
# Source: Publication 60: Field Crop Budgets, 2025, page 12
# Projected yield: 4.90 tonnes/acre (193 bu/acre)
# ============================================================

OMAFRA_2025 = {
    "source": "OMAFRA Publication 60: Field Crop Budgets, 2025 (p.12, No-Till column)",
    "yield_tonnes_per_acre": 4.90,
    "yield_bu_per_acre": 193,

    # Fertility — per acre
    "n_fertilizer_per_acre": 147.35,      # 171 kg/ha N as 611 kg/ha UAN 28-0-0
    "p2o5_fertilizer_per_acre": 75.95,    # 89 kg/ha P₂O₅ (crop removal: 7.25 kg/t)
    "k2o_fertilizer_per_acre": 28.45,     # 60 kg/ha K₂O (crop removal: 4.9 kg/t)

    # Fuel + lubricant — per acre
    "fuel_and_lube_per_acre": 22.90,      # 16 L diesel (no-till) + lubricant

    # Drying — per acre
    "drying_per_acre": 102.20,            # $20.85/tonne × 4.90 t/ac, 8 points removal

    # Trucking — per acre
    "trucking_per_acre": 53.95,           # $11.00/tonne × 4.90 t/ac
}

# ============================================================
# 2026 ESCALATION FACTORS
# Source: World Bank Pink Sheet (March 2026), DTN (March 2026),
#         Ridgetown 2025 baseline, Trading Economics (April 7, 2026)
# ============================================================

ESCALATION = {
    "nitrogen": {
        # UAN/Urea track together — both nitrogen products driven by nat gas
        # WB: urea +53.7% in March; DTN: +35% MoM; News: 40-50% since conflict
        "conservative": 0.45,  # Low end of news range
        "mid": 0.50,           # Midpoint
        "worst": 0.54,         # WB single-month spike
        "source": "World Bank urea +53.7% (March 2026); DTN US retail +35% MoM"
    },
    "phosphate": {
        # MAP: WB escalation vs DTN cross-reference
        "conservative": 0.06,
        "mid": 0.16,
        "worst": 0.26,
        "source": "Triangulated: WB MAP escalation +26%, DTN MAP cross-ref +6%"
    },
    "potash": {
        # MOP: least affected — Canada is world's largest producer
        "conservative": 0.04,
        "mid": 0.15,
        "worst": 0.26,
        "source": "Triangulated: WB potash escalation +26%, DTN potash cross-ref +4%"
    },
    "fuel": {
        # Diesel: Ridgetown 2025 $1.16/L → estimated $2.04-$2.22/L farm bulk
        # Applied to entire fuel+lube line (lube is also petroleum-derived)
        "conservative": 0.76,
        "mid": 0.85,
        "worst": 0.91,
        "source": "CTV retail $2.37/L → farm bulk via validated 14% discount"
    },
    "drying": {
        # Custom drying rate is driven primarily by propane cost
        # Propane estimated from crude oil correlation ($65 → $110 Brent)
        # Propane is ~60-70% of custom drying charge; remainder is equipment/labour
        # Blended escalation: ~60% × propane increase + ~40% × flat
        "conservative": 0.19,  # 60% × 32% propane increase
        "mid": 0.28,           # 60% × 46%
        "worst": 0.36,         # 60% × 60%
        "source": "Estimated: propane ~60% of $20.85/t drying charge; propane up 32-60% est.",
        "caveat": "WEAKEST ESTIMATE — propane pricing derived from crude oil correlation"
    },
}


def fmt(val, decimals=0):
    if decimals == 0:
        return f"${val:,.0f}"
    return f"${val:,.{decimals}f}"


def run_grain_corn_scenario(acres=500):
    """500-acre grain corn farm using OMAFRA Pub 60 baseline."""
    o = OMAFRA_2025
    e = ESCALATION

    print("=" * 80)
    print(f"SCENARIO 1: {acres}-ACRE GRAIN CORN FARM")
    print("=" * 80)
    print(f"\nBaseline: OMAFRA Publication 60, 2025 (No-Till)")
    print(f"Projected yield: {o['yield_bu_per_acre']} bu/acre ({o['yield_tonnes_per_acre']} t/ac)")
    print(f"Total acres: {acres}")

    # 2025 per-acre and total costs
    categories = {
        "N Fertilizer (UAN 28-0-0)": {
            "per_acre_2025": o["n_fertilizer_per_acre"],
            "escalation": e["nitrogen"],
        },
        "P₂O₅ Fertilizer": {
            "per_acre_2025": o["p2o5_fertilizer_per_acre"],
            "escalation": e["phosphate"],
        },
        "K₂O Fertilizer": {
            "per_acre_2025": o["k2o_fertilizer_per_acre"],
            "escalation": e["potash"],
        },
        "Fuel + Lubricant": {
            "per_acre_2025": o["fuel_and_lube_per_acre"],
            "escalation": e["fuel"],
        },
        "Drying": {
            "per_acre_2025": o["drying_per_acre"],
            "escalation": e["drying"],
        },
    }

    # Calculate
    results = {}
    for name, cat in categories.items():
        pa = cat["per_acre_2025"]
        esc = cat["escalation"]
        total_2025 = pa * acres
        total_cons = total_2025 * (1 + esc["conservative"])
        total_mid = total_2025 * (1 + esc["mid"])
        total_worst = total_2025 * (1 + esc["worst"])
        results[name] = {
            "per_acre_2025": pa,
            "total_2025": total_2025,
            "total_2026_cons": total_cons,
            "total_2026_mid": total_mid,
            "total_2026_worst": total_worst,
            "esc_cons": esc["conservative"],
            "esc_mid": esc["mid"],
            "esc_worst": esc["worst"],
        }

    # Print detail
    print(f"\n{'Category':<28} {'$/ac 2025':>10} {'Total 2025':>12} {'Esc %':>8} "
          f"{'2026 Cons':>12} {'2026 Mid':>12} {'2026 Worst':>12}")
    print("-" * 100)

    grand_2025 = 0
    grand_cons = 0
    grand_mid = 0
    grand_worst = 0

    # Subtotals for fertilizer
    fert_2025 = 0
    fert_cons = 0
    fert_mid = 0
    fert_worst = 0

    for name, r in results.items():
        print(f"{name:<28} {fmt(r['per_acre_2025'], 2):>10} {fmt(r['total_2025']):>12} "
              f"{r['esc_mid']*100:>7.0f}% "
              f"{fmt(r['total_2026_cons']):>12} {fmt(r['total_2026_mid']):>12} "
              f"{fmt(r['total_2026_worst']):>12}")
        grand_2025 += r["total_2025"]
        grand_cons += r["total_2026_cons"]
        grand_mid += r["total_2026_mid"]
        grand_worst += r["total_2026_worst"]

        if "Fertilizer" in name or "Fert" in name:
            fert_2025 += r["total_2025"]
            fert_cons += r["total_2026_cons"]
            fert_mid += r["total_2026_mid"]
            fert_worst += r["total_2026_worst"]

    fuel_r = results["Fuel + Lubricant"]
    dry_r = results["Drying"]

    print("-" * 100)
    print(f"{'TOTAL':<28} {'':>10} {fmt(grand_2025):>12} {'':>8} "
          f"{fmt(grand_cons):>12} {fmt(grand_mid):>12} {fmt(grand_worst):>12}")

    inc_cons = grand_cons - grand_2025
    inc_mid = grand_mid - grand_2025
    inc_worst = grand_worst - grand_2025

    pac_cons = round(inc_cons / acres)
    pac_mid = round(inc_mid / acres)
    pac_worst = round(inc_worst / acres)

    pct_cons = round((grand_cons / grand_2025 - 1) * 100)
    pct_mid = round((grand_mid / grand_2025 - 1) * 100)
    pct_worst = round((grand_worst / grand_2025 - 1) * 100)

    print(f"\n{'Metric':<28} {'Conservative':>12} {'Midpoint':>12} {'Worst-Case':>12}")
    print("-" * 68)
    print(f"{'Total Increase ($)':<28} {fmt(inc_cons):>12} {fmt(inc_mid):>12} {fmt(inc_worst):>12}")
    print(f"{'Increase (%)':<28} {'+' + str(pct_cons) + '%':>12} {'+' + str(pct_mid) + '%':>12} {'+' + str(pct_worst) + '%':>12}")
    print(f"{'Per Acre Increase':<28} {'$' + str(pac_cons) + '/ac':>12} {'$' + str(pac_mid) + '/ac':>12} {'$' + str(pac_worst) + '/ac':>12}")

    return {
        "scenario": f"{acres}-acre grain corn farm",
        "source": "OMAFRA Publication 60, 2025 (No-Till)",
        "acres": acres,
        "yield_bu_per_acre": o["yield_bu_per_acre"],
        "total_2025": round(grand_2025),
        "total_2026_range": [round(grand_cons), round(grand_mid), round(grand_worst)],
        "increase_range": [round(inc_cons), round(inc_mid), round(inc_worst)],
        "per_acre_increase": [pac_cons, pac_mid, pac_worst],
        "pct_increase": [pct_cons, pct_mid, pct_worst],
        "details": {
            "fertilizer": {"2025": round(fert_2025), "2026_mid": round(fert_mid)},
            "fuel": {"2025": round(fuel_r["total_2025"]), "2026_mid": round(fuel_r["total_2026_mid"])},
            "drying": {"2025": round(dry_r["total_2025"]), "2026_mid": round(dry_r["total_2026_mid"])},
        }
    }


def run_vegetable_farm_scenario():
    """200-acre vegetable farm — nitrogen/urea focus (unchanged)."""
    # This scenario is simpler — just urea pricing on nitrogen demand
    # Using OMAFRA Pub 839 vegetable N rates

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "farm_cost_impact_data.json")
    with open(data_path, "r") as f:
        DATA = json.load(f)

    RATES = DATA["application_rates"]
    CONV = RATES["conversions"]
    fert_base = DATA["baseline_2025"]["fertilizer"]["urea_46_0_0"]
    fert_est = DATA["estimated_april_2026_ontario"]["fertilizer"]["urea_46_0_0"]

    crops = {
        "sweet_corn":          {"acres": 60, "n_rate": 135},
        "potatoes":            {"acres": 50, "n_rate": 175},
        "tomatoes_processing": {"acres": 40, "n_rate": 120},
        "peppers":             {"acres": 25, "n_rate": 115},
        "leafy_greens":        {"acres": 25, "n_rate": 100},
    }
    total_acres = 200

    print("\n\n" + "=" * 80)
    print("SCENARIO 2: 200-ACRE VEGETABLE FARM (Nitrogen / Urea Focus)")
    print("=" * 80)

    total_n_lbs = 0
    for crop_name, info in crops.items():
        n_total = info["acres"] * info["n_rate"]
        total_n_lbs += n_total

    urea_lbs = total_n_lbs / CONV["urea_n_content"]
    urea_tonnes = urea_lbs / CONV["lbs_per_tonne"]

    price_2025 = fert_base["price_per_tonne"]
    price_2026_low = fert_est["working_range"]["low"]
    price_2026_mid = fert_est["midpoint"]
    price_2026_high = fert_est["working_range"]["high"]

    cost_2025 = urea_tonnes * price_2025
    cost_cons = urea_tonnes * price_2026_low
    cost_mid = urea_tonnes * price_2026_mid
    cost_worst = urea_tonnes * price_2026_high

    print(f"\n  Total nitrogen demand: {total_n_lbs:,} lbs N")
    print(f"  Urea equivalent: {urea_tonnes:.1f} tonnes")
    print(f"  2025 cost: {fmt(cost_2025)}")
    print(f"  2026 range: {fmt(cost_cons)} - {fmt(cost_mid)} - {fmt(cost_worst)}")
    print(f"  Increase: {fmt(cost_cons - cost_2025)} - {fmt(cost_worst - cost_2025)}")

    return {
        "scenario": "200-acre vegetable farm",
        "urea_tonnes": round(urea_tonnes, 1),
        "total_n_lbs": total_n_lbs,
        "cost_2025": round(cost_2025),
        "cost_2026_range": [round(cost_cons), round(cost_mid), round(cost_worst)],
        "increase_range": [round(cost_cons - cost_2025), round(cost_mid - cost_2025), round(cost_worst - cost_2025)],
        "per_acre_cost_2025": round(cost_2025 / total_acres),
        "per_acre_cost_2026_mid": round(cost_mid / total_acres),
    }


# ==============================================================================
# MULTI-SECTOR ENTERPRISE BUDGET MATRIX (2025 & 2026 VERIFIED BENCHMARKS)
# Sources: OMAFA Publication 60 (2025 & 2026 Field Crop Budgets, John Molenhuis),
#          Ontario Apple Growers (OAG) / OMAFA High-Density COP,
#          Ontario Processing Vegetable Growers (OPVG) / Agricorp,
#          Ontario Potato Board (OPB) / VCMI,
#          Ontario Greenhouse Vegetable Growers (OGVG) / StatCan NAICS 1114,
#          StatCan / Landscape Ontario Commercial Floriculture & Nursery (1114).
# ==============================================================================

CUSTOM_WORK_PASS_THROUGH_WEIGHTS = {
    "fuel": 0.22,
    "repairs": 0.18,
    "labour": 0.25,
    "capital_depr": 0.35,
}

ENTERPRISE_BUDGETS = {
    "grain_corn_notill_2025": {
        "label": "Grain Corn — No-Till (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60 (p.12)",
        "default_acres": 500,
        "unit_name": "acre",
        "yield_qty": 4.90,
        "yield_unit": "tonnes/ac (193 bu/ac)",
        "benchmark_price": 230.00,
        "price_unit": "$/tonne ($5.84/bu)",
        "gross_revenue": 1127.00,
        "seed": 135.50,
        "fertilizer": 251.75,
        "pesticides": 47.70,
        "fuel": 22.90,
        "repairs": 24.50,
        "custom_work": 44.50,
        "hired_labour": 14.20,
        "drying_storage_marketing": 176.45,
        "insurance_rmp": 26.80,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 295.00,
        "operating_interest": 29.85,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 6.0,
        "overhead_depr": 52.40,
    },
    "grain_corn_notill_2026": {
        "label": "Grain Corn — No-Till (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60 (p.12)",
        "default_acres": 500,
        "unit_name": "acre",
        "yield_qty": 4.90,
        "yield_unit": "tonnes/ac (193 bu/ac)",
        "benchmark_price": 225.00,
        "price_unit": "$/tonne ($5.72/bu)",
        "gross_revenue": 1102.50,
        "seed": 138.20,
        "fertilizer": 288.40,
        "pesticides": 48.50,
        "fuel": 21.30,
        "repairs": 27.10,
        "custom_work": 53.00,
        "hired_labour": 15.40,
        "drying_storage_marketing": 182.10,
        "insurance_rmp": 24.50,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 305.00,
        "operating_interest": 25.10,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 61.50,
    },
    "soybeans_rr_notill_2025": {
        "label": "Soybeans — Roundup Ready No-Till (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60 (p.13)",
        "default_acres": 500,
        "unit_name": "acre",
        "yield_qty": 1.36,
        "yield_unit": "tonnes/ac (50 bu/ac)",
        "benchmark_price": 495.00,
        "price_unit": "$/tonne ($13.47/bu)",
        "gross_revenue": 673.20,
        "seed": 108.85,
        "fertilizer": 68.05,
        "pesticides": 25.10,
        "fuel": 21.55,
        "repairs": 17.35,
        "custom_work": 44.50,
        "hired_labour": 11.45,
        "drying_storage_marketing": 29.75,
        "insurance_rmp": 15.95,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 275.00,
        "operating_interest": 12.20,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 6.0,
        "overhead_depr": 36.05,
    },
    "soybeans_rr_notill_2026": {
        "label": "Soybeans — Roundup Ready No-Till (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60 (p.13)",
        "default_acres": 500,
        "unit_name": "acre",
        "yield_qty": 1.36,
        "yield_unit": "tonnes/ac (50 bu/ac)",
        "benchmark_price": 485.00,
        "price_unit": "$/tonne ($13.20/bu)",
        "gross_revenue": 659.60,
        "seed": 109.50,
        "fertilizer": 80.95,
        "pesticides": 25.60,
        "fuel": 20.05,
        "repairs": 19.50,
        "custom_work": 53.00,
        "hired_labour": 12.55,
        "drying_storage_marketing": 32.10,
        "insurance_rmp": 13.55,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 285.00,
        "operating_interest": 10.05,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 45.00,
    },
    "soybeans_nongmo_notill_2025": {
        "label": "Soybeans — Non-GMO IP No-Till (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60",
        "default_acres": 400,
        "unit_name": "acre",
        "yield_qty": 1.36,
        "yield_unit": "tonnes/ac (50 bu/ac)",
        "benchmark_price": 570.00,
        "price_unit": "$/tonne ($15.51/bu)",
        "gross_revenue": 775.20,
        "seed": 81.10,
        "fertilizer": 68.05,
        "pesticides": 84.20,
        "fuel": 21.55,
        "repairs": 17.35,
        "custom_work": 44.50,
        "hired_labour": 11.45,
        "drying_storage_marketing": 29.75,
        "insurance_rmp": 15.95,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 275.00,
        "operating_interest": 13.30,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 6.0,
        "overhead_depr": 36.05,
    },
    "soybeans_nongmo_notill_2026": {
        "label": "Soybeans — Non-GMO IP No-Till (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60",
        "default_acres": 400,
        "unit_name": "acre",
        "yield_qty": 1.36,
        "yield_unit": "tonnes/ac (50 bu/ac)",
        "benchmark_price": 560.00,
        "price_unit": "$/tonne ($15.24/bu)",
        "gross_revenue": 761.60,
        "seed": 83.10,
        "fertilizer": 80.95,
        "pesticides": 84.55,
        "fuel": 20.05,
        "repairs": 19.50,
        "custom_work": 53.00,
        "hired_labour": 12.55,
        "drying_storage_marketing": 32.10,
        "insurance_rmp": 13.55,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 285.00,
        "operating_interest": 10.95,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 45.00,
    },
    "winter_wheat_srw_2025": {
        "label": "Soft Red Winter Wheat — No-Till Grain Only (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60 (p.15)",
        "default_acres": 300,
        "unit_name": "acre",
        "yield_qty": 2.53,
        "yield_unit": "tonnes/ac (93 bu/ac)",
        "benchmark_price": 255.00,
        "price_unit": "$/tonne ($6.94/bu)",
        "gross_revenue": 645.15,
        "seed": 80.65,
        "fertilizer": 136.00,
        "pesticides": 52.30,
        "fuel": 22.90,
        "repairs": 19.50,
        "custom_work": 58.30,
        "hired_labour": 11.80,
        "drying_storage_marketing": 28.80,
        "insurance_rmp": 13.50,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 250.00,
        "operating_interest": 25.80,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 9.5,
        "overhead_depr": 30.90,
    },
    "winter_wheat_srw_2026": {
        "label": "Soft Red Winter Wheat — No-Till Grain Only (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60 (p.15)",
        "default_acres": 300,
        "unit_name": "acre",
        "yield_qty": 2.63,
        "yield_unit": "tonnes/ac (96.6 bu/ac)",
        "benchmark_price": 250.00,
        "price_unit": "$/tonne ($6.80/bu)",
        "gross_revenue": 657.50,
        "seed": 88.25,
        "fertilizer": 180.90,
        "pesticides": 52.30,
        "fuel": 21.30,
        "repairs": 21.85,
        "custom_work": 71.00,
        "hired_labour": 12.90,
        "drying_storage_marketing": 31.45,
        "insurance_rmp": 11.75,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 260.00,
        "operating_interest": 23.10,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 9.5,
        "overhead_depr": 40.70,
    },
    "winter_wheat_srw_straw_2025": {
        "label": "Soft Red Winter Wheat + Baled Straw (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60 (p.15 Combined)",
        "default_acres": 300,
        "unit_name": "acre",
        "yield_qty": 2.53,
        "yield_unit": "2.53t grain + 2.41t straw/ac",
        "benchmark_price": 359.78,
        "price_unit": "$255/t grain + $110/t straw",
        "gross_revenue": 910.25,
        "seed": 80.65,
        "fertilizer": 175.85,
        "pesticides": 52.30,
        "fuel": 33.25,
        "repairs": 25.10,
        "custom_work": 71.10,
        "hired_labour": 28.20,
        "drying_storage_marketing": 28.80,
        "insurance_rmp": 13.50,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 250.00,
        "operating_interest": 25.80,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 9.5,
        "overhead_depr": 40.40,
    },
    "spring_canola_ht_2025": {
        "label": "Spring Canola — Herbicide-Tolerant (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60",
        "default_acres": 250,
        "unit_name": "acre",
        "yield_qty": 1.03,
        "yield_unit": "tonnes/ac (45.4 bu/ac)",
        "benchmark_price": 560.00,
        "price_unit": "$/tonne",
        "gross_revenue": 576.80,
        "seed": 93.25,
        "fertilizer": 135.65,
        "pesticides": 74.70,
        "fuel": 48.10,
        "repairs": 31.30,
        "custom_work": 58.30,
        "hired_labour": 15.45,
        "drying_storage_marketing": 15.05,
        "insurance_rmp": 42.60,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 185.00,
        "operating_interest": 17.90,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 6.0,
        "overhead_depr": 56.20,
    },
    "spring_canola_ht_2026": {
        "label": "Spring Canola — Herbicide-Tolerant (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60",
        "default_acres": 250,
        "unit_name": "acre",
        "yield_qty": 1.03,
        "yield_unit": "tonnes/ac (45.4 bu/ac)",
        "benchmark_price": 560.00,
        "price_unit": "$/tonne",
        "gross_revenue": 576.80,
        "seed": 84.75,
        "fertilizer": 176.95,
        "pesticides": 74.70,
        "fuel": 44.80,
        "repairs": 34.90,
        "custom_work": 71.00,
        "hired_labour": 15.45,
        "drying_storage_marketing": 15.65,
        "insurance_rmp": 33.80,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 195.00,
        "operating_interest": 14.80,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 84.80,
    },
    "winter_canola_2026": {
        "label": "Winter Canola — Hybrid (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60",
        "default_acres": 200,
        "unit_name": "acre",
        "yield_qty": 1.50,
        "yield_unit": "tonnes/ac (66.1 bu/ac)",
        "benchmark_price": 560.00,
        "price_unit": "$/tonne",
        "gross_revenue": 840.00,
        "seed": 70.20,
        "fertilizer": 273.30,
        "pesticides": 76.95,
        "fuel": 29.45,
        "repairs": 26.60,
        "custom_work": 88.00,
        "hired_labour": 13.05,
        "drying_storage_marketing": 22.80,
        "insurance_rmp": 39.35,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 210.00,
        "operating_interest": 17.45,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 9.5,
        "overhead_depr": 59.00,
    },
    "spring_barley_feed_2026": {
        "label": "Spring Barley — Feed Average (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60 (p.4)",
        "default_acres": 250,
        "unit_name": "acre",
        "yield_qty": 1.52,
        "yield_unit": "tonnes/ac (70 bu/ac)",
        "benchmark_price": 235.00,
        "price_unit": "$/tonne",
        "gross_revenue": 357.20,
        "seed": 66.60,
        "fertilizer": 101.50,
        "pesticides": 51.70,
        "fuel": 30.65,
        "repairs": 38.60,
        "custom_work": 53.00,
        "hired_labour": 14.85,
        "drying_storage_marketing": 18.90,
        "insurance_rmp": 19.50,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 175.00,
        "operating_interest": 13.30,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 91.35,
    },
    "spring_barley_malt_2026": {
        "label": "Spring Barley — Malting (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60 (p.4)",
        "default_acres": 250,
        "unit_name": "acre",
        "yield_qty": 1.52,
        "yield_unit": "tonnes/ac (70 bu/ac)",
        "benchmark_price": 290.00,
        "price_unit": "$/tonne",
        "gross_revenue": 440.80,
        "seed": 66.60,
        "fertilizer": 78.95,
        "pesticides": 51.70,
        "fuel": 30.65,
        "repairs": 38.60,
        "custom_work": 53.00,
        "hired_labour": 14.85,
        "drying_storage_marketing": 18.90,
        "insurance_rmp": 19.50,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 175.00,
        "operating_interest": 12.70,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 91.35,
    },
    "spring_oats_conv_2025": {
        "label": "Spring Oats — Conventional (2025 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2025 OMAFA Pub 60",
        "default_acres": 200,
        "unit_name": "acre",
        "yield_qty": 1.45,
        "yield_unit": "tonnes/ac (94 bu/ac)",
        "benchmark_price": 245.00,
        "price_unit": "$/tonne ($3.78/bu)",
        "gross_revenue": 355.25,
        "seed": 44.60,
        "fertilizer": 85.70,
        "pesticides": 58.80,
        "fuel": 31.65,
        "repairs": 35.50,
        "custom_work": 58.30,
        "hired_labour": 17.55,
        "drying_storage_marketing": 17.15,
        "insurance_rmp": 20.30,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 160.00,
        "operating_interest": 9.35,
        "base_interest_rate_pct": 6.95,
        "months_borrowed": 6.0,
        "overhead_depr": 61.80,
    },
    "spring_oats_conv_2026": {
        "label": "Spring Oats — Conventional (2026 Pub 60)",
        "category": "Field Crops (NAICS 1111)",
        "naics": "Oilseed and grain farming [1111]",
        "basis": "2026 OMAFA Pub 60",
        "default_acres": 200,
        "unit_name": "acre",
        "yield_qty": 1.45,
        "yield_unit": "tonnes/ac (94 bu/ac)",
        "benchmark_price": 245.00,
        "price_unit": "$/tonne ($3.78/bu)",
        "gross_revenue": 355.25,
        "seed": 44.60,
        "fertilizer": 96.95,
        "pesticides": 58.80,
        "fuel": 30.65,
        "repairs": 38.60,
        "custom_work": 71.00,
        "hired_labour": 14.85,
        "drying_storage_marketing": 18.05,
        "insurance_rmp": 17.95,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 165.00,
        "operating_interest": 7.60,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 91.35,
    },
    # --------------------------------------------------------------------------
    # PERENNIAL FRUIT & FIELD VEGETABLES (NAICS 1113 & 1112)
    # --------------------------------------------------------------------------
    "apple_hd_establish_yr1": {
        "label": "High-Density Apple Orchard — Year 1 Planting (OAG/OMAFA)",
        "category": "Fruit & Tree Nuts (NAICS 1113)",
        "naics": "Fruit and tree nut farming [1113]",
        "basis": "OAG/OMAFA High-Density Tall Spindle (1,210 trees/ac)",
        "default_acres": 40,
        "unit_name": "acre",
        "yield_qty": 0.0,
        "yield_unit": "bu/ac (Non-bearing Year 1)",
        "benchmark_price": 0.0,
        "price_unit": "$/lb",
        "gross_revenue": 0.00,
        "seed": 33585.00,
        "fertilizer": 485.00,
        "pesticides": 240.00,
        "fuel": 265.00,
        "repairs": 240.00,
        "custom_work": 0.00,
        "hired_labour": 4650.00,
        "drying_storage_marketing": 0.00,
        "insurance_rmp": 185.00,
        "land_rent_worksheet": 40.00,
        "market_land_rent": 450.00,
        "operating_interest": 870.00,
        "base_interest_rate_pct": 5.95,
        "months_borrowed": 12.0,
        "overhead_depr": 765.00,
    },
    "apple_hd_mature_2025": {
        "label": "High-Density Apple Orchard — Mature Producing Acre (OAG/OMAFA)",
        "category": "Fruit & Tree Nuts (NAICS 1113)",
        "naics": "Fruit and tree nut farming [1113]",
        "basis": "OAG/OMAFA High-Density Mature (Years 5-20, 55 bins/ac)",
        "default_acres": 85,
        "unit_name": "acre",
        "yield_qty": 46200.0,
        "yield_unit": "lbs/ac (1,100 bu / 55 bins)",
        "benchmark_price": 0.44,
        "price_unit": "$/lb ($18.48/bu bin blend)",
        "gross_revenue": 20328.00,
        "seed": 220.00,
        "fertilizer": 495.00,
        "pesticides": 1580.00,
        "fuel": 485.00,
        "repairs": 560.00,
        "custom_work": 0.00,
        "hired_labour": 5840.00,
        "drying_storage_marketing": 1815.00,
        "insurance_rmp": 465.00,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 450.00,
        "operating_interest": 488.00,
        "base_interest_rate_pct": 5.95,
        "months_borrowed": 8.0,
        "overhead_depr": 5655.00,
    },
    "field_tomatoes_proc_2025": {
        "label": "Field Processing Tomatoes — Irrigated Contract (OPVG/OMAFA)",
        "category": "Field Vegetables & Potatoes (NAICS 1112)",
        "naics": "Other vegetable (except potato) and melon farming [111219]",
        "basis": "OPVG / Agricorp / OMAFA Irrigated Benchmark",
        "default_acres": 150,
        "unit_name": "acre",
        "yield_qty": 45.0,
        "yield_unit": "short tons/ac (40.82 t/ac)",
        "benchmark_price": 152.00,
        "price_unit": "$/short ton ($167.55/t)",
        "gross_revenue": 6840.00,
        "seed": 952.00,
        "fertilizer": 485.00,
        "pesticides": 545.00,
        "fuel": 260.00,
        "repairs": 295.00,
        "custom_work": 0.00,
        "hired_labour": 860.00,
        "drying_storage_marketing": 495.00,
        "insurance_rmp": 145.00,
        "land_rent_worksheet": 450.00,
        "market_land_rent": 450.00,
        "operating_interest": 162.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 790.00,
    },
    "field_potatoes_chip_2025": {
        "label": "Processing Potatoes — Irrigated Chipstock (OPB/OMAFA)",
        "category": "Field Vegetables & Potatoes (NAICS 1112)",
        "naics": "Potato farming [111211]",
        "basis": "Ontario Potato Board / OMAFA Irrigated Chipstock",
        "default_acres": 300,
        "unit_name": "acre",
        "yield_qty": 300.0,
        "yield_unit": "cwt/ac (30,000 lbs/ac)",
        "benchmark_price": 17.50,
        "price_unit": "$/cwt ($350/short ton)",
        "gross_revenue": 5250.00,
        "seed": 760.00,
        "fertilizer": 640.00,
        "pesticides": 685.00,
        "fuel": 245.00,
        "repairs": 280.00,
        "custom_work": 0.00,
        "hired_labour": 440.00,
        "drying_storage_marketing": 380.00,
        "insurance_rmp": 115.00,
        "land_rent_worksheet": 350.00,
        "market_land_rent": 350.00,
        "operating_interest": 140.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 860.00,
    },
    "field_potatoes_fresh_2025": {
        "label": "Fresh Table Potatoes — Irrigated & Graded (OPB/VCMI)",
        "category": "Field Vegetables & Potatoes (NAICS 1112)",
        "naics": "Potato farming [111211]",
        "basis": "Ontario Potato Board / VCMI Table Stock Benchmark",
        "default_acres": 250,
        "unit_name": "acre",
        "yield_qty": 300.0,
        "yield_unit": "cwt/ac field (246 cwt packed)",
        "benchmark_price": 20.50,
        "price_unit": "$/cwt field-run ($25/cwt packed)",
        "gross_revenue": 6150.00,
        "seed": 820.00,
        "fertilizer": 640.00,
        "pesticides": 710.00,
        "fuel": 245.00,
        "repairs": 290.00,
        "custom_work": 0.00,
        "hired_labour": 580.00,
        "drying_storage_marketing": 695.00,
        "insurance_rmp": 125.00,
        "land_rent_worksheet": 350.00,
        "market_land_rent": 350.00,
        "operating_interest": 155.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 950.00,
    },
    # --------------------------------------------------------------------------
    # CONTROLLED-ENVIRONMENT AGRICULTURE & NURSERY/FLORICULTURE (NAICS 1114)
    # --------------------------------------------------------------------------
    "gh_tomatoes_acre_2025": {
        "label": "Greenhouse Tomatoes — High-Wire Hydroponic (OGVG/StatCan 1114)",
        "category": "Greenhouse, Nursery & Floriculture (NAICS 1114 — Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "basis": "OGVG / StatCan Leamington Glass/Poly Benchmark ($127.01/m²)",
        "default_acres": 15,
        "unit_name": "acre (4,047 m²)",
        "yield_qty": 242811.0,
        "yield_unit": "kg/ac (60.0 kg/m²)",
        "benchmark_price": 2.20,
        "price_unit": "$/kg ($1.00/lb)",
        "gross_revenue": 534184.00,
        "seed": 28500.00,
        "fertilizer": 31500.00,
        "pesticides": 11500.00,
        "fuel": 92000.00,
        "repairs": 16200.00,
        "custom_work": 0.00,
        "hired_labour": 154000.00,
        "drying_storage_marketing": 114500.00,
        "insurance_rmp": 6200.00,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 0.00,
        "operating_interest": 7600.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 4.5,
        "overhead_depr": 52000.00,
    },
    "gh_cucumbers_acre_2025": {
        "label": "Greenhouse Seedless Cucumbers — 3-Crop System (OGVG/StatCan 1114)",
        "category": "Greenhouse, Nursery & Floriculture (NAICS 1114 — Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "basis": "OGVG / StatCan Leamington Benchmark ($135.66/m²)",
        "default_acres": 15,
        "unit_name": "acre (4,047 m²)",
        "yield_qty": 250905.0,
        "yield_unit": "kg/ac (62.0 kg/m²)",
        "benchmark_price": 2.12,
        "price_unit": "$/kg (~$1.06/cucumber)",
        "gross_revenue": 531919.00,
        "seed": 46500.00,
        "fertilizer": 28200.00,
        "pesticides": 13800.00,
        "fuel": 96500.00,
        "repairs": 16800.00,
        "custom_work": 0.00,
        "hired_labour": 166000.00,
        "drying_storage_marketing": 113200.00,
        "insurance_rmp": 6400.00,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 0.00,
        "operating_interest": 8100.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 4.5,
        "overhead_depr": 53500.00,
    },
    "gh_peppers_acre_2025": {
        "label": "Greenhouse Sweet Bell Peppers — Hydroponic (OGVG/StatCan 1114)",
        "category": "Greenhouse, Nursery & Floriculture (NAICS 1114 — Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "basis": "OGVG / StatCan Leamington Benchmark ($142.01/m²)",
        "default_acres": 15,
        "unit_name": "acre (4,047 m²)",
        "yield_qty": 97125.0,
        "yield_unit": "kg/ac (24.0 kg/m²)",
        "benchmark_price": 5.55,
        "price_unit": "$/kg ($2.52/lb)",
        "gross_revenue": 539044.00,
        "seed": 34500.00,
        "fertilizer": 26800.00,
        "pesticides": 18500.00,
        "fuel": 104000.00,
        "repairs": 17500.00,
        "custom_work": 0.00,
        "hired_labour": 171500.00,
        "drying_storage_marketing": 128500.00,
        "insurance_rmp": 6800.00,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 0.00,
        "operating_interest": 8600.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 4.5,
        "overhead_depr": 58000.00,
    },
    "gh_floriculture_1114_2025": {
        "label": "Commercial Greenhouse Floriculture — Potted & Bedding (NAICS 111422)",
        "category": "Greenhouse, Nursery & Floriculture (NAICS 1114 — Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "basis": "StatCan / Landscape Ontario Heated Range ($14.38/sq.ft.)",
        "default_acres": 8,
        "unit_name": "acre (43,560 sq.ft.)",
        "yield_qty": 87120.0,
        "yield_unit": "finished trays/pots per acre",
        "benchmark_price": 7.86,
        "price_unit": "$/tray-pot ($15.73/sq.ft.)",
        "gross_revenue": 685000.00,
        "seed": 168000.00,
        "fertilizer": 18500.00,
        "pesticides": 14500.00,
        "fuel": 78000.00,
        "repairs": 19500.00,
        "custom_work": 0.00,
        "hired_labour": 182000.00,
        "drying_storage_marketing": 64000.00,
        "insurance_rmp": 8500.00,
        "land_rent_worksheet": 0.00,
        "market_land_rent": 0.00,
        "operating_interest": 11500.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 6.0,
        "overhead_depr": 62000.00,
    },
    "nursery_container_1114_2025": {
        "label": "Commercial Container Nursery Stock — Woody & Perennials (NAICS 111421)",
        "category": "Greenhouse, Nursery & Floriculture (NAICS 1114 — Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "basis": "StatCan / CNLA Container Nursery Benchmark ($51,000/ac)",
        "default_acres": 60,
        "unit_name": "acre",
        "yield_qty": 11000.0,
        "yield_unit": "finished #2-#5 containers/ac/yr",
        "benchmark_price": 5.40,
        "price_unit": "$/container wholesale FOB",
        "gross_revenue": 59400.00,
        "seed": 12800.00,
        "fertilizer": 3150.00,
        "pesticides": 1850.00,
        "fuel": 2650.00,
        "repairs": 2400.00,
        "custom_work": 0.00,
        "hired_labour": 14725.00,
        "drying_storage_marketing": 3850.00,
        "insurance_rmp": 0.00,
        "land_rent_worksheet": 850.00,
        "market_land_rent": 850.00,
        "operating_interest": 1925.00,
        "base_interest_rate_pct": 5.45,
        "months_borrowed": 18.0,
        "overhead_depr": 6800.00,
    },
}


def simulate_enterprise_budget(
    enterprise_key: str = "soybeans_rr_notill_2026",
    acres: float | None = None,
    shock_seed_pct: float = 0.0,
    shock_fertilizer_pct: float = 0.0,
    shock_pesticides_pct: float = 0.0,
    shock_fuel_pct: float = 0.0,
    shock_repairs_pct: float = 0.0,
    shock_labour_pct: float = 0.0,
    shock_interest_bps: float = 0.0,
    shock_output_price_pct: float = 0.0,
    include_cash_rent: bool = False,
    pass_through_custom_work: bool = True,
) -> dict:
    """
    Simulate per-acre and whole-enterprise operating & total margin impacts
    under macro input cost shocks, interest rate basis-point changes, and
    output price shocks.

    Implements:
    1. Custom fieldwork pass-through (22% Fuel, 18% Repairs, 25% Operator Labour).
    2. Endogenous operating interest feedback loop based on working-capital
       lock-up duration (months_borrowed) and shocked pre-harvest inputs.
    3. Optional cash-rent tenure switch (substituting market_land_rent).
    """
    if enterprise_key not in ENTERPRISE_BUDGETS:
        enterprise_key = "soybeans_rr_notill_2026"
    b = ENTERPRISE_BUDGETS[enterprise_key]
    n_acres = float(acres) if acres is not None else float(b["default_acres"])

    # Baseline line items ($/acre)
    base_seed = float(b["seed"])
    base_fert = float(b["fertilizer"])
    base_pest = float(b["pesticides"])
    base_fuel = float(b["fuel"])
    base_rep = float(b["repairs"])
    base_cust = float(b["custom_work"])
    base_lab = float(b["hired_labour"])
    base_dsm = float(b["drying_storage_marketing"])
    base_ins = float(b["insurance_rmp"])
    base_rent = float(b["market_land_rent"]) if include_cash_rent else float(b["land_rent_worksheet"])
    base_int = float(b["operating_interest"])
    base_ovh = float(b["overhead_depr"])

    # Shocked direct items ($/acre)
    new_seed = base_seed * (1.0 + shock_seed_pct / 100.0)
    new_fert = base_fert * (1.0 + shock_fertilizer_pct / 100.0)
    new_pest = base_pest * (1.0 + shock_pesticides_pct / 100.0)
    new_fuel = base_fuel * (1.0 + shock_fuel_pct / 100.0)
    new_rep = base_rep * (1.0 + shock_repairs_pct / 100.0)
    new_lab = base_lab * (1.0 + shock_labour_pct / 100.0)

    # Custom work pass-through (22% Fuel, 18% Repairs, 25% Labour, 35% Fixed Cap)
    if pass_through_custom_work and base_cust > 0:
        cw_pct_shock = (
            CUSTOM_WORK_PASS_THROUGH_WEIGHTS["fuel"] * (shock_fuel_pct / 100.0)
            + CUSTOM_WORK_PASS_THROUGH_WEIGHTS["repairs"] * (shock_repairs_pct / 100.0)
            + CUSTOM_WORK_PASS_THROUGH_WEIGHTS["labour"] * (shock_labour_pct / 100.0)
        )
        new_cust = base_cust * (1.0 + cw_pct_shock)
    else:
        new_cust = base_cust

    # Drying/trucking fuel sensitivity (15% energy/fuel weight on drying/trucking if fuel shocked)
    new_dsm = base_dsm * (1.0 + 0.15 * (shock_fuel_pct / 100.0))
    new_ins = base_ins
    new_rent = base_rent

    # Endogenous Operating Interest feedback loop:
    # Operating interest scales with both the shocked pre-harvest working capital
    # outlay AND the new effective operating interest rate (r_0 + Δbps/10000).
    base_preharvest = base_seed + base_fert + base_pest + base_fuel + base_rep + base_cust + base_lab
    new_preharvest = new_seed + new_fert + new_pest + new_fuel + new_rep + new_cust + new_lab
    working_cap_ratio = (new_preharvest / base_preharvest) if base_preharvest > 0 else 1.0

    r0_pct = float(b.get("base_interest_rate_pct", 5.45))
    r_new_pct = max(0.0, r0_pct + (shock_interest_bps / 100.0))
    rate_ratio = (r_new_pct / r0_pct) if r0_pct > 0 else 1.0
    new_int = base_int * working_cap_ratio * rate_ratio

    # Totals per acre
    base_op_ac = (
        base_seed + base_fert + base_pest + base_fuel + base_rep
        + base_cust + base_lab + base_dsm + base_ins + base_rent + base_int
    )
    new_op_ac = (
        new_seed + new_fert + new_pest + new_fuel + new_rep
        + new_cust + new_lab + new_dsm + new_ins + new_rent + new_int
    )
    base_tot_ac = base_op_ac + base_ovh
    new_tot_ac = new_op_ac + base_ovh

    # Revenue & Margins per acre
    base_rev_ac = float(b["gross_revenue"])
    new_rev_ac = base_rev_ac * (1.0 + shock_output_price_pct / 100.0)

    base_op_margin_ac = base_rev_ac - base_op_ac
    new_op_margin_ac = new_rev_ac - new_op_ac
    base_net_margin_ac = base_rev_ac - base_tot_ac
    new_net_margin_ac = new_rev_ac - new_tot_ac

    line_items = [
        {"Line Item": "Seed / Plants / Growing Media", "Baseline ($/ac)": base_seed, "Shocked ($/ac)": new_seed, "Delta ($/ac)": new_seed - base_seed, "Whole Farm Delta ($)": (new_seed - base_seed) * n_acres},
        {"Line Item": "Fertilizer, Nutrients & CO2", "Baseline ($/ac)": base_fert, "Shocked ($/ac)": new_fert, "Delta ($/ac)": new_fert - base_fert, "Whole Farm Delta ($)": (new_fert - base_fert) * n_acres},
        {"Line Item": "Pesticides & Biological IPM", "Baseline ($/ac)": base_pest, "Shocked ($/ac)": new_pest, "Delta ($/ac)": new_pest - base_pest, "Whole Farm Delta ($)": (new_pest - base_pest) * n_acres},
        {"Line Item": "Machinery & Heating Fuel / Energy", "Baseline ($/ac)": base_fuel, "Shocked ($/ac)": new_fuel, "Delta ($/ac)": new_fuel - base_fuel, "Whole Farm Delta ($)": (new_fuel - base_fuel) * n_acres},
        {"Line Item": "Machinery & Facility Repairs", "Baseline ($/ac)": base_rep, "Shocked ($/ac)": new_rep, "Delta ($/ac)": new_rep - base_rep, "Whole Farm Delta ($)": (new_rep - base_rep) * n_acres},
        {"Line Item": "Custom Fieldwork (Fert/Spray)", "Baseline ($/ac)": base_cust, "Shocked ($/ac)": new_cust, "Delta ($/ac)": new_cust - base_cust, "Whole Farm Delta ($)": (new_cust - base_cust) * n_acres},
        {"Line Item": "Hired & Operator Labour", "Baseline ($/ac)": base_lab, "Shocked ($/ac)": new_lab, "Delta ($/ac)": new_lab - base_lab, "Whole Farm Delta ($)": (new_lab - base_lab) * n_acres},
        {"Line Item": "Drying, Storage, Packing & Mktg", "Baseline ($/ac)": base_dsm, "Shocked ($/ac)": new_dsm, "Delta ($/ac)": new_dsm - base_dsm, "Whole Farm Delta ($)": (new_dsm - base_dsm) * n_acres},
        {"Line Item": "Production Insurance & RMP", "Baseline ($/ac)": base_ins, "Shocked ($/ac)": new_ins, "Delta ($/ac)": 0.0, "Whole Farm Delta ($)": 0.0},
        {"Line Item": "Land Rent / Tenure Charge", "Baseline ($/ac)": base_rent, "Shocked ($/ac)": new_rent, "Delta ($/ac)": 0.0, "Whole Farm Delta ($)": 0.0},
        {"Line Item": "Operating Interest (Endogenous)", "Baseline ($/ac)": base_int, "Shocked ($/ac)": new_int, "Delta ($/ac)": new_int - base_int, "Whole Farm Delta ($)": (new_int - base_int) * n_acres},
    ]

    return {
        "enterprise_key": enterprise_key,
        "label": b["label"],
        "category": b["category"],
        "naics": b["naics"],
        "basis": b["basis"],
        "acres": n_acres,
        "unit_name": b["unit_name"],
        "yield_qty": b["yield_qty"],
        "yield_unit": b["yield_unit"],
        "benchmark_price": b["benchmark_price"],
        "price_unit": b["price_unit"],
        "months_borrowed": b["months_borrowed"],
        "base_interest_rate_pct": r0_pct,
        "new_interest_rate_pct": r_new_pct,
        "include_cash_rent": include_cash_rent,
        "base_rev_ac": base_rev_ac,
        "new_rev_ac": new_rev_ac,
        "base_op_ac": base_op_ac,
        "new_op_ac": new_op_ac,
        "delta_op_ac": new_op_ac - base_op_ac,
        "pct_op_increase": ((new_op_ac / base_op_ac - 1.0) * 100.0) if base_op_ac > 0 else 0.0,
        "base_tot_ac": base_tot_ac,
        "new_tot_ac": new_tot_ac,
        "base_op_margin_ac": base_op_margin_ac,
        "new_op_margin_ac": new_op_margin_ac,
        "delta_op_margin_ac": new_op_margin_ac - base_op_margin_ac,
        "base_net_margin_ac": base_net_margin_ac,
        "new_net_margin_ac": new_net_margin_ac,
        "delta_net_margin_ac": new_net_margin_ac - base_net_margin_ac,
        "farm_base_rev": base_rev_ac * n_acres,
        "farm_new_rev": new_rev_ac * n_acres,
        "farm_base_op_cost": base_op_ac * n_acres,
        "farm_new_op_cost": new_op_ac * n_acres,
        "farm_delta_op_cost": (new_op_ac - base_op_ac) * n_acres,
        "farm_base_net_margin": base_net_margin_ac * n_acres,
        "farm_new_net_margin": new_net_margin_ac * n_acres,
        "farm_delta_net_margin": (new_net_margin_ac - base_net_margin_ac) * n_acres,
        "line_items": line_items,
    }


if __name__ == "__main__":
    grain = run_grain_corn_scenario(500)
    veg = run_vegetable_farm_scenario()

    output = {
        "grain_farm_500ac": grain,
        "vegetable_farm_200ac": veg,
        "data_sources": {
            "2025_baseline_grain": "OMAFRA Publication 60: Field Crop Budgets, 2025 (p.12, No-Till)",
            "2025_baseline_fertilizer_prices": "Ridgetown Farm Input Monitoring Project, April 16, 2025",
            "2026_escalation": "Triangulated: World Bank Pink Sheet (March 2026), DTN Retail Index (March 2026), Trading Economics (April 7, 2026)",
            "2026_fuel": "CTV News retail + validated 14% farm bulk discount",
            "vegetable_rates": "OMAFRA Publication 839",
            "geopolitical": "Strait of Hormuz disruption (Feb 2026 onset)"
        },
        "caveats": [
            "2026 prices are estimates — Ridgetown spring 2026 survey not yet published",
            "Drying cost escalation assumes propane is ~60% of custom drying charge — weakest estimate",
            "StatsCan FIPI Q4 2025 (releases April 10, 2026) may provide additional validation",
            "This analysis excludes crop protection, seed, machinery parts, and labour",
            "World Bank escalation (Method A) represents spot/replacement cost — blended retail may lag"
        ]
    }

    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "farm_cost_impact_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to: {results_path}")
