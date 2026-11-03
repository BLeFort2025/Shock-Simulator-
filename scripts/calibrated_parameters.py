"""
Empirical Macroeconomic, Econometric & Supply-Use Calibration Parameters
========================================================================
Single source of truth for verified empirical parameters across Canadian and
Ontario Primary Agriculture and Agri-Food Value Chains (zero heuristics).

Grounded in:
1. Canadian Productivity Accounts (StatCan Tables 36-10-0208-01 & 36-10-0217-01),
   USDA ERS International Agricultural Productivity (Jan 2026 release), and
   Darku, Malla & Tran (2016) provincial stochastic frontier decomposition.
2. Coupled Nerlove-Armington Dynamic Supply & Tariff Pass-Through Literature
   (AAFC, StatCan, USITC, CJAE, FAPRI, Bank of Canada SWP 2026-22 & 2026-36)
   covering all 18 Primary Agriculture sectors (strictly classifying NAICS 1114 /
   HS Chapter 06 & HS 07 Nursery, Floriculture, Sod & Greenhouse as Primary Ag).
3. Statistics Canada Supply and Use Tables (36-10-0478-01 / 36-10-0438-01) and
   Agriculture and Agri-Food Economic Account (AAEA 36-10-0623-01) technical
   coefficients (a_{farm, proc}^{dom}) and disposition matrices (s_{proc},
   s_{intra-ag}, s_{fresh}, s_{exp}) for Analysis-by-Parts (ABP).
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


# ==============================================================================
# 1. PRODUCTIVITY ACCOUNTS & 5x5 STOCHASTIC VARIANCE-COVARIANCE MATRICES
#    Variables z_t = [1: FIPI (% YoY), 2: TFP (% YoY), 3: Interest Rate (Δ pp),
#                     4: Real Net Cash Income (% YoY), 5: Farm CapEx (% YoY)]
# ==============================================================================

TFP_AND_MC_REGIMES: Dict[str, Dict[str, Any]] = {
    "Canada": {
        "value_added": {
            "source": "StatCan Table 36-10-0208-01 (Value-Added MFP, NAICS 11 / BS11, 1990-2023)",
            "mu_tfp": 1.60,
            "mu_tfp_range": (1.40, 1.85),
            "sigma_tfp": 6.00,
            # [FIPI (% YoY), TFP (% YoY), Interest Rate (Δ pp), Net Cash Income (% YoY), Farm CapEx (% YoY)]
            "stds": [6.50, 6.00, 1.20, 18.50, 12.00],
            "corr": [
                [1.00, -0.18,  0.25, -0.12,  0.15],
                [-0.18, 1.00, -0.08,  0.52,  0.18],
                [0.25, -0.08,  1.00, -0.22, -0.28],
                [-0.12, 0.52, -0.22,  1.00,  0.35],
                [0.15,  0.18, -0.28,  0.35,  1.00],
            ],
        },
        "gross_output": {
            "source": "StatCan KLEMS Table 36-10-0217-01 & USDA ERS IAP (Gross-Output TFP, NAICS 111-112, 1990-2023)",
            "mu_tfp": 1.55,
            "mu_tfp_range": (1.07, 2.20),
            "sigma_tfp": 4.20,
            "stds": [6.50, 4.20, 1.20, 18.50, 12.00],
            "corr": [
                [1.00, -0.18,  0.25, -0.12,  0.15],
                [-0.18, 1.00, -0.08,  0.52,  0.18],
                [0.25, -0.08,  1.00, -0.22, -0.28],
                [-0.12, 0.52, -0.22,  1.00,  0.35],
                [0.15,  0.18, -0.28,  0.35,  1.00],
            ],
        },
    },
    "Ontario": {
        "value_added": {
            "source": "StatCan Table 36-10-0208-01 + Darku, Malla & Tran (2016) Ontario Scale/CEA Adjustment",
            "mu_tfp": 1.85,
            "mu_tfp_range": (1.21, 2.77),
            "sigma_tfp": 4.10,
            "stds": [6.50, 4.10, 1.20, 16.50, 11.00],
            "corr": [
                [1.00, -0.15,  0.25, -0.12,  0.15],
                [-0.15, 1.00, -0.12,  0.42,  0.22],
                [0.25, -0.12,  1.00, -0.22, -0.28],
                [-0.12, 0.42, -0.22,  1.00,  0.35],
                [0.15,  0.22, -0.28,  0.35,  1.00],
            ],
        },
        "gross_output": {
            "source": "StatCan KLEMS 36-10-0217-01 / USDA ERS IAP + Darku et al. (2016) Ontario CEA/Supply-Managed Buffer",
            "mu_tfp": 1.85,
            "mu_tfp_range": (1.21, 2.77),
            "sigma_tfp": 2.90,
            "stds": [6.50, 2.90, 1.20, 16.50, 11.00],
            "corr": [
                [1.00, -0.15,  0.25, -0.12,  0.15],
                [-0.15, 1.00, -0.12,  0.42,  0.22],
                [0.25, -0.12,  1.00, -0.22, -0.28],
                [-0.12, 0.42, -0.22,  1.00,  0.35],
                [0.15,  0.22, -0.28,  0.35,  1.00],
            ],
        },
    },
}

PROVINCIAL_TFP_DECOMPOSITION: List[Dict[str, Any]] = [
    {
        "subsector": "Crops",
        "province": "Ontario",
        "total_tfp_growth_pct": 1.21,
        "technical_change_pp": 0.54,
        "technical_change_share_pct": 44.6,
        "scale_effects_pp": 0.63,
        "scale_effects_share_pct": 52.1,
        "tech_efficiency_pp": 0.08,
        "residual_pp": -0.04,
        "dominant_driver": "Farm consolidation & scale expansion (corn/soy rotations)",
    },
    {
        "subsector": "Crops",
        "province": "Manitoba",
        "total_tfp_growth_pct": 2.03,
        "technical_change_pp": 1.61,
        "technical_change_share_pct": 79.3,
        "scale_effects_pp": 0.34,
        "scale_effects_share_pct": 16.7,
        "tech_efficiency_pp": 0.07,
        "residual_pp": 0.01,
        "dominant_driver": "Technological progress & crop genetics",
    },
    {
        "subsector": "Crops",
        "province": "Saskatchewan",
        "total_tfp_growth_pct": 1.69,
        "technical_change_pp": 1.44,
        "technical_change_share_pct": 85.2,
        "scale_effects_pp": 0.21,
        "scale_effects_share_pct": 12.4,
        "tech_efficiency_pp": 0.05,
        "residual_pp": -0.01,
        "dominant_driver": "Zero-till, herbicide-tolerant canola & GPS auto-steer",
    },
    {
        "subsector": "Crops",
        "province": "Alberta",
        "total_tfp_growth_pct": 1.57,
        "technical_change_pp": 1.39,
        "technical_change_share_pct": 88.5,
        "scale_effects_pp": 0.06,
        "scale_effects_share_pct": 6.4,
        "tech_efficiency_pp": 0.06,
        "residual_pp": 0.02,
        "dominant_driver": "Agronomic & mechanical innovation",
    },
    {
        "subsector": "Livestock",
        "province": "Ontario",
        "total_tfp_growth_pct": 2.77,
        "technical_change_pp": 0.72,
        "technical_change_share_pct": 29.6,
        "scale_effects_pp": 1.92,
        "scale_effects_share_pct": 69.3,
        "tech_efficiency_pp": 0.26,
        "residual_pp": -0.04,
        "dominant_driver": "Intensive confinement scale economies (dairy, poultry, swine)",
    },
    {
        "subsector": "Livestock",
        "province": "Manitoba",
        "total_tfp_growth_pct": 1.08,
        "technical_change_pp": 0.38,
        "technical_change_share_pct": 35.1,
        "scale_effects_pp": 0.56,
        "scale_effects_share_pct": 51.9,
        "tech_efficiency_pp": 0.12,
        "residual_pp": 0.02,
        "dominant_driver": "Hog & dairy scale + genetics",
    },
    {
        "subsector": "Livestock",
        "province": "Saskatchewan",
        "total_tfp_growth_pct": 0.73,
        "technical_change_pp": 0.23,
        "technical_change_share_pct": 31.5,
        "scale_effects_pp": 0.41,
        "scale_effects_share_pct": 56.2,
        "tech_efficiency_pp": 0.08,
        "residual_pp": 0.01,
        "dominant_driver": "Moderate herd scale expansion",
    },
    {
        "subsector": "Livestock",
        "province": "Alberta",
        "total_tfp_growth_pct": 0.61,
        "technical_change_pp": 0.20,
        "technical_change_share_pct": 32.8,
        "scale_effects_pp": 0.31,
        "scale_effects_share_pct": 50.8,
        "tech_efficiency_pp": 0.09,
        "residual_pp": 0.01,
        "dominant_driver": "Extensive beef cow-calf / feedlot scale",
    },
]


def get_mc_covariance_params(region: str = "Ontario", basis: str = "gross_output") -> Dict[str, Any]:
    """Return verified standard deviations, correlation matrix, and covariance matrix."""
    reg_key = "Ontario" if region == "Ontario" else "Canada"
    basis_key = basis if basis in ("value_added", "gross_output") else "gross_output"
    cfg = TFP_AND_MC_REGIMES[reg_key][basis_key]
    stds = np.array(cfg["stds"], dtype=float)
    corr = np.array(cfg["corr"], dtype=float)
    cov = np.diag(stds) @ corr @ np.diag(stds)
    return {
        "region": reg_key,
        "basis": basis_key,
        "source": cfg["source"],
        "mu_tfp": cfg["mu_tfp"],
        "mu_tfp_range": cfg["mu_tfp_range"],
        "sigma_tfp": cfg["sigma_tfp"],
        "stds": stds,
        "corr": corr,
        "cov": cov,
    }


# ==============================================================================
# 2. MASTER CALIBRATION TABLE — ALL 18 PRIMARY AGRICULTURE SECTORS
#    Note: In the simulator, supply contraction under a price drop is written
#    with negative sign (e_sr < 0, e_lr < 0) so that a tariff lowers output,
#    while Nerlove adjustment speed lam = |e_sr| / |e_lr| in (0, 1].
#    NAICS 1114 / HS Ch. 06 & 07 are strictly classified as Primary Agriculture.
# ==============================================================================

PRIMARY_AG_TARIFF_CONFIG: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # PART A: HORTICULTURE & GREENHOUSE PRIMARY AGRICULTURE SECTORS
    # --------------------------------------------------------------------------
    "Floriculture, Nursery & Sod (HS Ch. 06)": {
        "category": "Horticulture & Greenhouse (Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "io_join_code": "1114A",
        "hs_chapters": ["06"],
        "fcr_items": [
            "Floriculture and nursery products, and sod",
            "Christmas trees [115142111]",
        ],
        "e_sr": -0.35, "e_sr_p10_p90": (-0.50, -0.20),
        "e_lr": -1.05, "e_lr_p10_p90": (-1.40, -0.75),
        "lam": 0.333,  "lam_p10_p90": (0.22, 0.45),
        "pass_through": 0.65, "pass_through_p10_p90": (0.48, 0.82),
        "us_price_share": 0.38, "us_price_share_p10_p90": (0.30, 0.45),
        "us_price_share_on": 0.42, "us_price_share_on_p10_p90": (0.34, 0.50),
        "us_lop_spillover": 0.45, "us_lop_spillover_p10_p90": (0.30, 0.60),
        "citations": "AAFC Ornamental Overview (2024); StatCan 32-10-0019/0246/0045; Landscape Ontario/CNLA; Nerlove (1958)",
    },
    "Greenhouse Vegetables": {
        "category": "Horticulture & Greenhouse (Primary Ag)",
        "naics": "Greenhouse, nursery and floriculture production [1114]",
        "io_join_code": "1114A",
        "hs_chapters": ["0702", "0707", "0709"],
        "fcr_items": ["Fresh vegetables - greenhouse"],
        "e_sr": -0.30, "e_sr_p10_p90": (-0.45, -0.18),
        "e_lr": -1.25, "e_lr_p10_p90": (-1.65, -0.90),
        "lam": 0.240,  "lam_p10_p90": (0.16, 0.34),
        "pass_through": 0.78, "pass_through_p10_p90": (0.62, 0.90),
        "us_price_share": 0.76, "us_price_share_p10_p90": (0.70, 0.82),
        "us_price_share_on": 0.84, "us_price_share_on_p10_p90": (0.78, 0.90),
        "us_lop_spillover": 0.85, "us_lop_spillover_p10_p90": (0.72, 0.94),
        "citations": "AAFC GH Veg Overview (2024); USITC Pub 5268/5151; Lopez et al. (2016) SAIDS; CAPI/RBC",
    },
    "Field Vegetables & Melons": {
        "category": "Horticulture & Greenhouse (Primary Ag)",
        "naics": "Other vegetable (except potato) and melon farming [111219]",
        "io_join_code": "111A",
        "hs_chapters": ["07"],
        "fcr_items": ["Fresh vegetables - field"],
        "e_sr": -0.45, "e_sr_p10_p90": (-0.62, -0.28),
        "e_lr": -1.10, "e_lr_p10_p90": (-1.45, -0.80),
        "lam": 0.409,  "lam_p10_p90": (0.30, 0.52),
        "pass_through": 0.55, "pass_through_p10_p90": (0.40, 0.70),
        "us_price_share": 0.22, "us_price_share_p10_p90": (0.16, 0.28),
        "us_price_share_on": 0.18, "us_price_share_on_p10_p90": (0.12, 0.24),
        "us_lop_spillover": 0.70, "us_lop_spillover_p10_p90": (0.55, 0.82),
        "citations": "AAFC Field Veg Overview (2025); Fontagne et al. (2019) / Chepeliev GTAP-HS; OMAFRA",
    },
    "Potatoes": {
        "category": "Horticulture & Greenhouse (Primary Ag)",
        "naics": "Potato farming [111211]",
        "io_join_code": "111A",
        "hs_chapters": ["0701", "2004"],
        "fcr_items": ["Fresh potatoes [114211]", "Seed potatoes [114212]"],
        "e_sr": -0.32, "e_sr_p10_p90": (-0.46, -0.18),
        "e_lr": -0.85, "e_lr_p10_p90": (-1.15, -0.60),
        "lam": 0.376,  "lam_p10_p90": (0.26, 0.48),
        "pass_through": 0.60, "pass_through_p10_p90": (0.45, 0.75),
        "us_price_share": 0.52, "us_price_share_p10_p90": (0.44, 0.60),
        "us_price_share_on": 0.14, "us_price_share_on_p10_p90": (0.09, 0.20),
        "us_lop_spillover": 0.80, "us_lop_spillover_p10_p90": (0.68, 0.90),
        "citations": "AAFC Potato Market Info Review (2024-25); StatCan 32-10-0358; Abizadeh (1988); Pocock",
    },
    "Fresh Fruit & Tree Nuts": {
        "category": "Horticulture & Greenhouse (Primary Ag)",
        "naics": "Fruit and tree nut farming [1113]",
        "io_join_code": "111A",
        "hs_chapters": ["08"],
        "fcr_items": ["Total fresh fruit [11411A]"],
        "e_sr": -0.15, "e_sr_p10_p90": (-0.25, -0.08),
        "e_lr": -0.95, "e_lr_p10_p90": (-1.30, -0.65),
        "lam": 0.158,  "lam_p10_p90": (0.10, 0.24),
        "pass_through": 0.58, "pass_through_p10_p90": (0.42, 0.74),
        "us_price_share": 0.38, "us_price_share_p10_p90": (0.30, 0.46),
        "us_price_share_on": 0.12, "us_price_share_on_p10_p90": (0.07, 0.18),
        "us_lop_spillover": 0.75, "us_lop_spillover_p10_p90": (0.60, 0.86),
        "citations": "AAFC Fruit Overview (2024); Devadoss et al. / French & King (1985); Burlig et al. (2021); FAO",
    },
    # --------------------------------------------------------------------------
    # PART B: FIELD CROPS & LIVESTOCK PRIMARY AGRICULTURE SECTORS
    # --------------------------------------------------------------------------
    "Soybeans": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["1201"],
        "fcr_items": ["Soybeans [1151211]"],
        "e_sr": -0.52, "e_sr_p10_p90": (-0.70, -0.35),
        "e_lr": -1.15, "e_lr_p10_p90": (-1.50, -0.85),
        "lam": 0.452,  "lam_p10_p90": (0.34, 0.56),
        "pass_through": 0.25, "pass_through_p10_p90": (0.15, 0.38),
        "us_price_share": 0.18, "us_price_share_p10_p90": (0.12, 0.24),
        "us_price_share_on": 0.22, "us_price_share_on_p10_p90": (0.16, 0.28),
        "us_lop_spillover": 0.98, "us_lop_spillover_p10_p90": (0.92, 1.00),
        "citations": "Haile et al. (2016); Roberts & Schlenker (2013); FAPRI; Goodwin & Piggott (2001) / Sephton (2003)",
    },
    "Corn (Grain)": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["1005"],
        "fcr_items": ["Corn for grain [1151111]"],
        "e_sr": -0.48, "e_sr_p10_p90": (-0.65, -0.32),
        "e_lr": -1.05, "e_lr_p10_p90": (-1.35, -0.78),
        "lam": 0.457,  "lam_p10_p90": (0.35, 0.56),
        "pass_through": 0.20, "pass_through_p10_p90": (0.10, 0.32),
        "us_price_share": 0.12, "us_price_share_p10_p90": (0.08, 0.18),
        "us_price_share_on": 0.14, "us_price_share_on_p10_p90": (0.09, 0.20),
        "us_lop_spillover": 0.98, "us_lop_spillover_p10_p90": (0.92, 1.00),
        "citations": "Quebec Corn Nerlove Study (1985-2013); Haile et al. (2016); Roberts & Schlenker (2013); Sephton (2003)",
    },
    "Canola": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["1205"],
        "fcr_items": ["Canola (including rapeseed) [113111]", "Flaxseed [115122111]"],
        "e_sr": -0.46, "e_sr_p10_p90": (-0.62, -0.30),
        "e_lr": -1.10, "e_lr_p10_p90": (-1.40, -0.80),
        "lam": 0.418,  "lam_p10_p90": (0.30, 0.52),
        "pass_through": 0.22, "pass_through_p10_p90": (0.12, 0.35),
        "us_price_share": 0.15, "us_price_share_p10_p90": (0.10, 0.22),
        "us_price_share_on": 0.12, "us_price_share_on_p10_p90": (0.06, 0.18),
        "us_lop_spillover": 0.95, "us_lop_spillover_p10_p90": (0.88, 0.99),
        "citations": "UAlberta PR-00-05; US EPA FASOM; Agri-Food Econ (2025); StatCan 32-10-0046",
    },
    "Wheat (Excl. Durum)": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["100199"],
        "fcr_items": ["Wheat (except durum wheat) [1121111]"],
        "e_sr": -0.42, "e_sr_p10_p90": (-0.58, -0.28),
        "e_lr": -0.95, "e_lr_p10_p90": (-1.25, -0.70),
        "lam": 0.442,  "lam_p10_p90": (0.33, 0.55),
        "pass_through": 0.18, "pass_through_p10_p90": (0.10, 0.30),
        "us_price_share": 0.16, "us_price_share_p10_p90": (0.11, 0.22),
        "us_price_share_on": 0.18, "us_price_share_on_p10_p90": (0.12, 0.25),
        "us_lop_spillover": 0.92, "us_lop_spillover_p10_p90": (0.84, 0.97),
        "citations": "CJAE Canadian Wheat Acreage Response; UAlberta PR-00-05; Haile et al.; Goodwin LOP (1992)",
    },
    "Durum Wheat": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["100119"],
        "fcr_items": ["Durum wheat [112111211]"],
        "e_sr": -0.45, "e_sr_p10_p90": (-0.60, -0.30),
        "e_lr": -1.00, "e_lr_p10_p90": (-1.30, -0.75),
        "lam": 0.450,  "lam_p10_p90": (0.34, 0.56),
        "pass_through": 0.22, "pass_through_p10_p90": (0.12, 0.34),
        "us_price_share": 0.20, "us_price_share_p10_p90": (0.14, 0.26),
        "us_price_share_on": 0.00, "us_price_share_on_p10_p90": (0.00, 0.01),
        "us_lop_spillover": 0.90, "us_lop_spillover_p10_p90": (0.80, 0.96),
        "citations": "OECD/FAPRI; PDIC Grain Disputes; USask CDC; StatCan 32-10-0046",
    },
    "Oats": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["1004"],
        "fcr_items": ["Oats [115113111]"],
        "e_sr": -0.55, "e_sr_p10_p90": (-0.72, -0.38),
        "e_lr": -1.15, "e_lr_p10_p90": (-1.45, -0.85),
        "lam": 0.478,  "lam_p10_p90": (0.36, 0.58),
        "pass_through": 0.70, "pass_through_p10_p90": (0.55, 0.84),
        "us_price_share": 0.68, "us_price_share_p10_p90": (0.60, 0.76),
        "us_price_share_on": 0.10, "us_price_share_on_p10_p90": (0.05, 0.16),
        "us_lop_spillover": 0.95, "us_lop_spillover_p10_p90": (0.88, 0.99),
        "citations": "CJAE Ontario Acreage Response; OECD Coarse Grains; USask CDC; PDIC",
    },
    "Barley": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["1003"],
        "fcr_items": ["Barley [1151141]"],
        "e_sr": -0.40, "e_sr_p10_p90": (-0.55, -0.25),
        "e_lr": -0.90, "e_lr_p10_p90": (-1.18, -0.65),
        "lam": 0.444,  "lam_p10_p90": (0.33, 0.55),
        "pass_through": 0.20, "pass_through_p10_p90": (0.12, 0.32),
        "us_price_share": 0.14, "us_price_share_p10_p90": (0.09, 0.20),
        "us_price_share_on": 0.08, "us_price_share_on_p10_p90": (0.04, 0.13),
        "us_lop_spillover": 0.88, "us_lop_spillover_p10_p90": (0.78, 0.95),
        "citations": "UAlberta PR-00-05; SaskBarley / Gray et al.; CAPI; Agri-Food Econ",
    },
    "Dry Peas": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["071310"],
        "fcr_items": ["Dry peas [114314]", "Dry beans [114313]", "Chickpeas [114311]"],
        "e_sr": -0.50, "e_sr_p10_p90": (-0.66, -0.34),
        "e_lr": -1.10, "e_lr_p10_p90": (-1.40, -0.82),
        "lam": 0.455,  "lam_p10_p90": (0.34, 0.56),
        "pass_through": 0.25, "pass_through_p10_p90": (0.14, 0.38),
        "us_price_share": 0.15, "us_price_share_p10_p90": (0.10, 0.21),
        "us_price_share_on": 0.05, "us_price_share_on_p10_p90": (0.02, 0.09),
        "us_lop_spillover": 0.85, "us_lop_spillover_p10_p90": (0.74, 0.93),
        "citations": "USask CDC Plant Breeding Impact (2016); Pulse Canada / AAFC Overview",
    },
    "Lentils": {
        "category": "Field Crops (Primary Ag)",
        "naics": "Oilseed and grain farming [1111]",
        "io_join_code": "111A",
        "hs_chapters": ["071340"],
        "fcr_items": ["Lentils [114312]"],
        "e_sr": -0.48, "e_sr_p10_p90": (-0.64, -0.32),
        "e_lr": -1.05, "e_lr_p10_p90": (-1.35, -0.78),
        "lam": 0.457,  "lam_p10_p90": (0.34, 0.56),
        "pass_through": 0.22, "pass_through_p10_p90": (0.12, 0.34),
        "us_price_share": 0.11, "us_price_share_p10_p90": (0.07, 0.16),
        "us_price_share_on": 0.02, "us_price_share_on_p10_p90": (0.00, 0.05),
        "us_lop_spillover": 0.82, "us_lop_spillover_p10_p90": (0.70, 0.91),
        "citations": "USask CDC Plant Breeding Impact (2016); Ryan; AAFC Overview",
    },
    "Beef Cattle": {
        "category": "Livestock (Primary Ag)",
        "naics": "Beef cattle ranching and farming, including feedlots [112110]",
        "io_join_code": "112A",
        "hs_chapters": ["0102", "0201", "0202"],
        "fcr_items": ["Cattle [1111111]", "Calves [1111112]"],
        "e_sr": -0.15, "e_sr_p10_p90": (-0.28, 0.05),
        "e_lr": -0.85, "e_lr_p10_p90": (-1.15, -0.60),
        "lam": 0.176,  "lam_p10_p90": (0.11, 0.26),
        "pass_through": 0.62, "pass_through_p10_p90": (0.48, 0.76),
        "us_price_share": 0.44, "us_price_share_p10_p90": (0.36, 0.52),
        "us_price_share_on": 0.32, "us_price_share_on_p10_p90": (0.24, 0.40),
        "us_lop_spillover": 0.96, "us_lop_spillover_p10_p90": (0.90, 0.99),
        "citations": "UAlberta PR-00-05 (0.123 SR); Rude (2020) CJAE; Gordon (1990) / Rosen (1974); WTO DS384",
    },
    "Hogs & Pork": {
        "category": "Livestock (Primary Ag)",
        "naics": "Hog and pig farming [1122]",
        "io_join_code": "112A",
        "hs_chapters": ["0103", "0203"],
        "fcr_items": ["Hogs [111121]"],
        "e_sr": -0.32, "e_sr_p10_p90": (-0.45, -0.20),
        "e_lr": -1.00, "e_lr_p10_p90": (-1.30, -0.75),
        "lam": 0.320,  "lam_p10_p90": (0.22, 0.42),
        "pass_through": 0.58, "pass_through_p10_p90": (0.44, 0.72),
        "us_price_share": 0.40, "us_price_share_p10_p90": (0.33, 0.48),
        "us_price_share_on": 0.35, "us_price_share_on_p10_p90": (0.28, 0.43),
        "us_lop_spillover": 0.95, "us_lop_spillover_p10_p90": (0.88, 0.99),
        "citations": "Rude & Surry (2014) CJAE; Holt & Johnson (1988) CJAE; Meilke et al.; FAPRI; WTO DS384",
    },
    "Dairy (Supply Managed)": {
        "category": "Livestock (Primary Ag)",
        "naics": "Dairy cattle and milk production [112120]",
        "io_join_code": "112A",
        "hs_chapters": ["0401", "0402", "0403", "0404", "0405", "0406"],
        "fcr_items": ["Unprocessed milk from bovine [11612111]"],
        "e_sr": -0.02, "e_sr_p10_p90": (-0.05, 0.00),
        "e_lr": -0.08, "e_lr_p10_p90": (-0.15, -0.02),
        "lam": 0.250,  "lam_p10_p90": (0.15, 0.35),
        "pass_through": 0.02, "pass_through_p10_p90": (0.00, 0.05),
        "us_price_share": 0.015, "us_price_share_p10_p90": (0.01, 0.03),
        "us_price_share_on": 0.015, "us_price_share_on_p10_p90": (0.01, 0.03),
        "us_lop_spillover": 0.02, "us_lop_spillover_p10_p90": (0.00, 0.05),
        "citations": "Abbassi et al. (2008) CJAE; Cairns & Meilke (2012) CJAE; Moschini & Meilke (1990)",
    },
    "Poultry & Eggs (Supply Managed)": {
        "category": "Livestock (Primary Ag)",
        "naics": "Poultry and egg production [1123]",
        "io_join_code": "112A",
        "hs_chapters": ["0105", "0207", "0407"],
        "fcr_items": [
            "Chickens for meat [11113131]",
            "Turkeys for meat [111132211]",
            "Eggs in shell [116111]",
            "Hatcheries - chicks and poults [111133]",
        ],
        "e_sr": -0.03, "e_sr_p10_p90": (-0.06, 0.00),
        "e_lr": -0.10, "e_lr_p10_p90": (-0.18, -0.03),
        "lam": 0.300,  "lam_p10_p90": (0.18, 0.40),
        "pass_through": 0.02, "pass_through_p10_p90": (0.00, 0.05),
        "us_price_share": 0.020, "us_price_share_p10_p90": (0.01, 0.04),
        "us_price_share_on": 0.020, "us_price_share_on_p10_p90": (0.01, 0.04),
        "us_lop_spillover": 0.02, "us_lop_spillover_p10_p90": (0.00, 0.05),
        "citations": "Carter & Merel (2016) CJE; StatCan 32-10-0046; AAFC; Broiler USMCA Model",
    },
}


# ==============================================================================
# 3. ANALYSIS-BY-PARTS (ABP) TECHNICAL COEFFICIENTS (a_{farm, proc}^{dom})
#    AND PRIMARY FARM DISPOSITION MATRICES (s_proc, s_intra_ag, s_fresh, s_exp)
#    Source: Statistics Canada SUT 36-10-0478-01 / 36-10-0438-01 & AAEA 36-10-0623-01
# ==============================================================================

ABP_PROCESSOR_TABLE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "3111": {
        "name": "Animal food manufacturing (NAICS 3111)",
        "join_code": "3111",
        "Canada": {
            "int_share": 0.8119,
            "a_111A_dom": 0.2450,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0120,
            "a_farm_proc_dom": 0.2570,
            "m_farm_imp": 0.0480,
            "a_311_intra": 0.2650,
            "psi_pass_through": 0.69,
        },
        "Ontario": {
            "int_share": 0.7702,
            "a_111A_dom": 0.2180,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0100,
            "a_farm_proc_dom": 0.2280,
            "m_farm_imp": 0.0720,
            "a_311_intra": 0.2400,
            "psi_pass_through": 0.69,
        },
    },
    "3112": {
        "name": "Grain and oilseed milling (NAICS 3112)",
        "join_code": "3112",
        "Canada": {
            "int_share": 0.8399,
            "a_111A_dom": 0.5180,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.5180,
            "m_farm_imp": 0.0640,
            "a_311_intra": 0.0580,
            "psi_pass_through": 0.69,
        },
        "Ontario": {
            "int_share": 0.8250,
            "a_111A_dom": 0.3620,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.3620,
            "m_farm_imp": 0.1950,
            "a_311_intra": 0.0680,
            "psi_pass_through": 0.69,
        },
    },
    "3113": {
        "name": "Sugar and confectionery product manufacturing (NAICS 3113)",
        "join_code": "3113",
        "Canada": {
            "int_share": 0.6820,
            "a_111A_dom": 0.0320,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0030,
            "a_farm_proc_dom": 0.0350,
            "m_farm_imp": 0.1850,
            "a_311_intra": 0.2150,
            "psi_pass_through": 0.45,
        },
        "Ontario": {
            "int_share": 0.6750,
            "a_111A_dom": 0.0180,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0030,
            "a_farm_proc_dom": 0.0210,
            "m_farm_imp": 0.2100,
            "a_311_intra": 0.2200,
            "psi_pass_through": 0.45,
        },
    },
    "3114": {
        "name": "Fruit and vegetable preserving and specialty food mfg (NAICS 3114)",
        "join_code": "3114",
        "Canada": {
            "int_share": 0.6908,
            "a_111A_dom": 0.1680,
            "a_1114A_dom": 0.0240,
            "a_112A_dom": 0.0060,
            "a_farm_proc_dom": 0.1980,
            "m_farm_imp": 0.0950,
            "a_311_intra": 0.1550,
            "psi_pass_through": 0.52,
        },
        "Ontario": {
            "int_share": 0.6880,
            "a_111A_dom": 0.1350,
            "a_1114A_dom": 0.0380,
            "a_112A_dom": 0.0060,
            "a_farm_proc_dom": 0.1790,
            "m_farm_imp": 0.1250,
            "a_311_intra": 0.1600,
            "psi_pass_through": 0.52,
        },
    },
    "3115": {
        "name": "Dairy product manufacturing (NAICS 3115)",
        "join_code": "3115",
        "Canada": {
            "int_share": 0.7696,
            "a_111A_dom": 0.0040,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.4680,
            "a_farm_proc_dom": 0.4720,
            "m_farm_imp": 0.0150,
            "a_311_intra": 0.1180,
            "psi_pass_through": 0.75,
        },
        "Ontario": {
            "int_share": 0.7650,
            "a_111A_dom": 0.0040,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.4450,
            "a_farm_proc_dom": 0.4490,
            "m_farm_imp": 0.0320,
            "a_311_intra": 0.1240,
            "psi_pass_through": 0.75,
        },
    },
    "3116": {
        "name": "Meat product manufacturing (NAICS 3116)",
        "join_code": "3116",
        "Canada": {
            "int_share": 0.7652,
            "a_111A_dom": 0.0030,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.4390,
            "a_farm_proc_dom": 0.4420,
            "m_farm_imp": 0.0380,
            "a_311_intra": 0.1420,
            "psi_pass_through": 0.24,
        },
        "Ontario": {
            "int_share": 0.7580,
            "a_111A_dom": 0.0030,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.3480,
            "a_farm_proc_dom": 0.3510,
            "m_farm_imp": 0.1050,
            "a_311_intra": 0.1650,
            "psi_pass_through": 0.24,
        },
    },
    "3117": {
        "name": "Seafood product preparation and packaging (NAICS 3117)",
        "join_code": "3117",
        "Canada": {
            "int_share": 0.7855,
            "a_111A_dom": 0.0010,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0020,
            "a_farm_proc_dom": 0.0030,
            "m_farm_imp": 0.0850,
            "a_311_intra": 0.0650,
            "psi_pass_through": 0.35,
        },
        "Ontario": {
            "int_share": 0.7500,
            "a_111A_dom": 0.0010,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0020,
            "a_farm_proc_dom": 0.0030,
            "m_farm_imp": 0.4200,
            "a_311_intra": 0.0950,
            "psi_pass_through": 0.35,
        },
    },
    "3118": {
        "name": "Bakeries and tortilla manufacturing (NAICS 3118)",
        "join_code": "3118",
        "Canada": {
            "int_share": 0.6125,
            "a_111A_dom": 0.0140,
            "a_1114A_dom": 0.0010,
            "a_112A_dom": 0.0190,
            "a_farm_proc_dom": 0.0340,
            "m_farm_imp": 0.0220,
            "a_311_intra": 0.2850,
            "psi_pass_through": 0.55,
        },
        "Ontario": {
            "int_share": 0.6080,
            "a_111A_dom": 0.0110,
            "a_1114A_dom": 0.0010,
            "a_112A_dom": 0.0180,
            "a_farm_proc_dom": 0.0300,
            "m_farm_imp": 0.0280,
            "a_311_intra": 0.2920,
            "psi_pass_through": 0.55,
        },
    },
    "3119": {
        "name": "Other food manufacturing (NAICS 3119)",
        "join_code": "3119",
        "Canada": {
            "int_share": 0.6622,
            "a_111A_dom": 0.0680,
            "a_1114A_dom": 0.0040,
            "a_112A_dom": 0.0180,
            "a_farm_proc_dom": 0.0900,
            "m_farm_imp": 0.1150,
            "a_311_intra": 0.2100,
            "psi_pass_through": 0.50,
        },
        "Ontario": {
            "int_share": 0.6580,
            "a_111A_dom": 0.0520,
            "a_1114A_dom": 0.0050,
            "a_112A_dom": 0.0160,
            "a_farm_proc_dom": 0.0730,
            "m_farm_imp": 0.1320,
            "a_311_intra": 0.2180,
            "psi_pass_through": 0.50,
        },
    },
    "31211": {
        "name": "Soft drink and ice manufacturing (NAICS 31211)",
        "join_code": "31211",
        "Canada": {
            "int_share": 0.6850,
            "a_111A_dom": 0.0040,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.0040,
            "m_farm_imp": 0.0180,
            "a_311_intra": 0.2450,
            "psi_pass_through": 0.40,
        },
        "Ontario": {
            "int_share": 0.6800,
            "a_111A_dom": 0.0030,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.0030,
            "m_farm_imp": 0.0200,
            "a_311_intra": 0.2500,
            "psi_pass_through": 0.40,
        },
    },
    "31212": {
        "name": "Breweries (NAICS 31212)",
        "join_code": "31212",
        "Canada": {
            "int_share": 0.5450,
            "a_111A_dom": 0.0280,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.0280,
            "m_farm_imp": 0.0190,
            "a_311_intra": 0.1150,
            "psi_pass_through": 0.45,
        },
        "Ontario": {
            "int_share": 0.5400,
            "a_111A_dom": 0.0120,
            "a_1114A_dom": 0.0000,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.0120,
            "m_farm_imp": 0.0350,
            "a_311_intra": 0.1200,
            "psi_pass_through": 0.45,
        },
    },
    "3121A": {
        "name": "Wineries and distilleries (NAICS 3121A)",
        "join_code": "3121A",
        "Canada": {
            "int_share": 0.6150,
            "a_111A_dom": 0.1120,
            "a_1114A_dom": 0.0020,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.1140,
            "m_farm_imp": 0.0880,
            "a_311_intra": 0.0950,
            "psi_pass_through": 0.55,
        },
        "Ontario": {
            "int_share": 0.6100,
            "a_111A_dom": 0.1280,
            "a_1114A_dom": 0.0030,
            "a_112A_dom": 0.0000,
            "a_farm_proc_dom": 0.1310,
            "m_farm_imp": 0.0920,
            "a_311_intra": 0.0980,
            "psi_pass_through": 0.55,
        },
    },
}

# Primary Farm Sub-Sector Disposition Matrix (Table 2 of Deep Research Output #3)
# Channels sum to 1.00: s_proc (sold to domestic NAICS 311/312), s_intra_ag (on-farm/feedlot/seed/landscaping),
# s_fresh (direct fresh household/foodservice consumption), s_exp (direct primary exports + ROC for ON)
PRIMARY_AG_DISPOSITION_MATRIX: Dict[str, Dict[str, Any]] = {
    "All farm types": {
        "Canada": {"s_proc": 0.478, "s_intra_ag": 0.132, "s_fresh": 0.068, "s_exp": 0.322},
        "Ontario": {"s_proc": 0.554, "s_intra_ag": 0.128, "s_fresh": 0.106, "s_exp": 0.212},
        "processor_weights": {"3116": 0.36, "3115": 0.20, "3112": 0.20, "3114": 0.10, "3111": 0.09, "3121A": 0.03, "3118": 0.02},
    },
    "Crop production [111]": {
        "Canada": {"s_proc": 0.295, "s_intra_ag": 0.155, "s_fresh": 0.105, "s_exp": 0.445},
        "Ontario": {"s_proc": 0.375, "s_intra_ag": 0.165, "s_fresh": 0.175, "s_exp": 0.285},
        "processor_weights": {"3112": 0.52, "3114": 0.22, "3111": 0.16, "3121A": 0.07, "3119": 0.03},
    },
    "Animal production [112]": {
        "Canada": {"s_proc": 0.765, "s_intra_ag": 0.105, "s_fresh": 0.030, "s_exp": 0.100},
        "Ontario": {"s_proc": 0.785, "s_intra_ag": 0.080, "s_fresh": 0.035, "s_exp": 0.100},
        "processor_weights": {"3116": 0.62, "3115": 0.35, "3119": 0.02, "3118": 0.01},
    },
    "Oilseed and grain farming [1111]": {
        "Canada": {"s_proc": 0.265, "s_intra_ag": 0.175, "s_fresh": 0.005, "s_exp": 0.555},
        "Ontario": {"s_proc": 0.480, "s_intra_ag": 0.210, "s_fresh": 0.005, "s_exp": 0.305},
        "processor_weights": {"3112": 0.62, "3111": 0.27, "3121A": 0.08, "31212": 0.03},
    },
    "Potato farming [111211]": {
        "Canada": {"s_proc": 0.650, "s_intra_ag": 0.050, "s_fresh": 0.180, "s_exp": 0.120},
        "Ontario": {"s_proc": 0.540, "s_intra_ag": 0.040, "s_fresh": 0.340, "s_exp": 0.080},
        "processor_weights": {"3114": 0.85, "3119": 0.15},
    },
    "Other vegetable (except potato) and melon farming [111219]": {
        "Canada": {"s_proc": 0.310, "s_intra_ag": 0.020, "s_fresh": 0.420, "s_exp": 0.250},
        "Ontario": {"s_proc": 0.360, "s_intra_ag": 0.025, "s_fresh": 0.415, "s_exp": 0.200},
        "processor_weights": {"3114": 0.92, "3119": 0.08},
    },
    "Fruit and tree nut farming [1113]": {
        "Canada": {"s_proc": 0.310, "s_intra_ag": 0.015, "s_fresh": 0.445, "s_exp": 0.230},
        "Ontario": {"s_proc": 0.380, "s_intra_ag": 0.015, "s_fresh": 0.485, "s_exp": 0.120},
        "processor_weights": {"3121A": 0.55, "3114": 0.37, "3118": 0.05, "3115": 0.03},
    },
    "Greenhouse, nursery and floriculture production [1114]": {
        "Canada": {"s_proc": 0.045, "s_intra_ag": 0.215, "s_fresh": 0.360, "s_exp": 0.380},
        "Ontario": {"s_proc": 0.050, "s_intra_ag": 0.160, "s_fresh": 0.270, "s_exp": 0.520},
        "processor_weights": {"3114": 0.90, "3119": 0.10},
    },
    "Other crop farming [1119]": {
        "Canada": {"s_proc": 0.250, "s_intra_ag": 0.300, "s_fresh": 0.150, "s_exp": 0.300},
        "Ontario": {"s_proc": 0.300, "s_intra_ag": 0.280, "s_fresh": 0.180, "s_exp": 0.240},
        "processor_weights": {"3111": 0.45, "3113": 0.25, "3119": 0.30},
    },
    "Beef cattle ranching and farming, including feedlots [112110]": {
        "Canada": {"s_proc": 0.620, "s_intra_ag": 0.215, "s_fresh": 0.005, "s_exp": 0.160},
        "Ontario": {"s_proc": 0.690, "s_intra_ag": 0.175, "s_fresh": 0.010, "s_exp": 0.125},
        "processor_weights": {"3116": 0.98, "3111": 0.02},
    },
    "Beef cattle ranching and farming, including feedlots [112111]": {
        "Canada": {"s_proc": 0.620, "s_intra_ag": 0.215, "s_fresh": 0.005, "s_exp": 0.160},
        "Ontario": {"s_proc": 0.690, "s_intra_ag": 0.175, "s_fresh": 0.010, "s_exp": 0.125},
        "processor_weights": {"3116": 0.98, "3111": 0.02},
    },
    "Dairy cattle and milk production [112120]": {
        "Canada": {"s_proc": 0.985, "s_intra_ag": 0.010, "s_fresh": 0.000, "s_exp": 0.005},
        "Ontario": {"s_proc": 0.960, "s_intra_ag": 0.010, "s_fresh": 0.000, "s_exp": 0.030},
        "processor_weights": {"3115": 0.98, "3118": 0.01, "3113": 0.01},
    },
    "Dairy cattle and milk production [11212]": {
        "Canada": {"s_proc": 0.985, "s_intra_ag": 0.010, "s_fresh": 0.000, "s_exp": 0.005},
        "Ontario": {"s_proc": 0.960, "s_intra_ag": 0.010, "s_fresh": 0.000, "s_exp": 0.030},
        "processor_weights": {"3115": 0.98, "3118": 0.01, "3113": 0.01},
    },
    "Hog and pig farming [1122]": {
        "Canada": {"s_proc": 0.785, "s_intra_ag": 0.065, "s_fresh": 0.005, "s_exp": 0.145},
        "Ontario": {"s_proc": 0.680, "s_intra_ag": 0.060, "s_fresh": 0.005, "s_exp": 0.255},
        "processor_weights": {"3116": 0.99, "3111": 0.01},
    },
    "Poultry and egg production [1123]": {
        "Canada": {"s_proc": 0.745, "s_intra_ag": 0.035, "s_fresh": 0.205, "s_exp": 0.015},
        "Ontario": {"s_proc": 0.735, "s_intra_ag": 0.035, "s_fresh": 0.210, "s_exp": 0.020},
        "processor_weights": {"3116": 0.89, "3119": 0.07, "3118": 0.04},
    },
    "Sheep and goat farming [1124]": {
        "Canada": {"s_proc": 0.650, "s_intra_ag": 0.100, "s_fresh": 0.200, "s_exp": 0.050},
        "Ontario": {"s_proc": 0.680, "s_intra_ag": 0.090, "s_fresh": 0.190, "s_exp": 0.040},
        "processor_weights": {"3116": 0.75, "3115": 0.25},
    },
    "Other animal production [112A]": {
        "Canada": {"s_proc": 0.500, "s_intra_ag": 0.150, "s_fresh": 0.150, "s_exp": 0.200},
        "Ontario": {"s_proc": 0.520, "s_intra_ag": 0.140, "s_fresh": 0.160, "s_exp": 0.180},
        "processor_weights": {"3116": 0.60, "3119": 0.30, "3111": 0.10},
    },
    "Other animal production [1129]": {
        "Canada": {"s_proc": 0.500, "s_intra_ag": 0.150, "s_fresh": 0.150, "s_exp": 0.200},
        "Ontario": {"s_proc": 0.520, "s_intra_ag": 0.140, "s_fresh": 0.160, "s_exp": 0.180},
        "processor_weights": {"3116": 0.60, "3119": 0.30, "3111": 0.10},
    },
}

# Mutually Exclusive Leaf NAICS Sub-Sectors in StatCan Table 32-10-0136-01
# (Separating leaf sub-sectors from 3-digit parent aggregates [111] and [112] — Fixes Bug #3)
MUTUALLY_EXCLUSIVE_LEAF_NAICS: List[str] = [
    "Oilseed and grain farming [1111]",
    "Potato farming [111211]",
    "Other vegetable (except potato) and melon farming [111219]",
    "Fruit and tree nut farming [1113]",
    "Greenhouse, nursery and floriculture production [1114]",
    "Other crop farming [1119]",
    "Beef cattle ranching and farming, including feedlots [112110]",
    "Beef cattle ranching and farming, including feedlots [112111]",
    "Dairy cattle and milk production [112120]",
    "Dairy cattle and milk production [11212]",
    "Hog and pig farming [1122]",
    "Poultry and egg production [1123]",
    "Sheep and goat farming [1124]",
    "Other animal production [112A]",
    "Other animal production [1129]",
]

PARENT_AGGREGATE_NAICS: List[str] = [
    "Crop production [111]",
    "Animal production [112]",
]

# Empirical Incremental Capital-Output Ratios (ICORs) from StatCan Tables 36-10-0096-01 & 36-10-0488-01
EMPIRICAL_ICOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "Primary Farming & CEA Greenhouse (NAICS 111-112 / 1114)": {
        "icor": 0.72,
        "output_per_dollar_capex": 1.0 / 0.72,
        "gestation_years": 2,
        "ramp_schedule": [0.45, 0.85, 1.00],
        "source": "StatCan Tables 36-10-0096-01 & 36-10-0488-01 (Primary Ag & CEA Capital Stock / Value-Added)",
    },
    "Food & Beverage Processing (NAICS 311-312)": {
        "icor": 2.59,
        "output_per_dollar_capex": 1.0 / 2.59,
        "gestation_years": 3,
        "ramp_schedule": [0.25, 0.65, 0.90, 1.00],
        "source": "StatCan Tables 36-10-0096-01 & 36-10-0488-01 (Food Manufacturing Fixed Capital Formation)",
    },
    "Cold-Chain, Grain Handling & Trade Logistics (NAICS 411/493)": {
        "icor": 0.32,
        "output_per_dollar_capex": 1.0 / 0.32,
        "gestation_years": 2,
        "ramp_schedule": [0.55, 0.90, 1.00],
        "source": "StatCan Tables 36-10-0096-01 & 36-10-0488-01 (Agri-Wholesale & Refrigerated Storage)",
    },
}


# ==============================================================================
# 4. FARM DEBT MATURITY STRUCTURE & 5-YEAR INTEREST RATE PASS-THROUGH SCHEDULE
#    Source: Statistics Canada Tables 32-10-0051-01 & 32-10-0052-01; FCC Portfolio
#    Reports (2022–2025); Deep Research Output #5 (Block 1)
#    Note: sum_{k=1..5} theta_yr[k] = 1.000
# ==============================================================================

DEBT_REPRICING_SCHEDULE: Dict[str, Dict[str, Any]] = {
    "Canada|All farm types": {
        "working_capital_share": 0.165,
        "variable_term_share": 0.140,
        "yr1_maturing_fixed_share": 0.115,
        "theta_schedule": [0.420, 0.160, 0.150, 0.140, 0.130],
    },
    "Ontario|All farm types": {
        "working_capital_share": 0.175,
        "variable_term_share": 0.145,
        "yr1_maturing_fixed_share": 0.120,
        "theta_schedule": [0.440, 0.160, 0.150, 0.130, 0.120],
    },
    "Crop production [111]": {
        "working_capital_share": 0.165,
        "variable_term_share": 0.132,
        "yr1_maturing_fixed_share": 0.118,
        "theta_schedule": [0.415, 0.163, 0.152, 0.140, 0.130],
    },
    "Animal production [112]": {
        "working_capital_share": 0.210,
        "variable_term_share": 0.135,
        "yr1_maturing_fixed_share": 0.105,
        "theta_schedule": [0.450, 0.150, 0.144, 0.134, 0.122],
    },
    "Oilseed and grain farming [1111]": {
        "working_capital_share": 0.150,
        "variable_term_share": 0.125,
        "yr1_maturing_fixed_share": 0.115,
        "theta_schedule": [0.390, 0.165, 0.155, 0.150, 0.140],
    },
    "Greenhouse, nursery and floriculture production [1114]": {
        "working_capital_share": 0.185,
        "variable_term_share": 0.145,
        "yr1_maturing_fixed_share": 0.120,
        "theta_schedule": [0.450, 0.165, 0.150, 0.125, 0.110],
    },
    "Potato farming [111211]": {
        "working_capital_share": 0.200,
        "variable_term_share": 0.140,
        "yr1_maturing_fixed_share": 0.120,
        "theta_schedule": [0.460, 0.160, 0.145, 0.125, 0.110],
    },
    "Other vegetable (except potato) and melon farming [111219]": {
        "working_capital_share": 0.200,
        "variable_term_share": 0.140,
        "yr1_maturing_fixed_share": 0.120,
        "theta_schedule": [0.460, 0.160, 0.145, 0.125, 0.110],
    },
    "Fruit and tree nut farming [1113]": {
        "working_capital_share": 0.200,
        "variable_term_share": 0.140,
        "yr1_maturing_fixed_share": 0.120,
        "theta_schedule": [0.460, 0.160, 0.145, 0.125, 0.110],
    },
    "Other crop farming [1119]": {
        "working_capital_share": 0.175,
        "variable_term_share": 0.135,
        "yr1_maturing_fixed_share": 0.115,
        "theta_schedule": [0.425, 0.160, 0.150, 0.140, 0.125],
    },
    "Dairy cattle and milk production [112120]": {
        "working_capital_share": 0.085,
        "variable_term_share": 0.125,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.320, 0.175, 0.175, 0.170, 0.160],
    },
    "Dairy cattle and milk production [11212]": {
        "working_capital_share": 0.085,
        "variable_term_share": 0.125,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.320, 0.175, 0.175, 0.170, 0.160],
    },
    "Poultry and egg production [1123]": {
        "working_capital_share": 0.100,
        "variable_term_share": 0.130,
        "yr1_maturing_fixed_share": 0.115,
        "theta_schedule": [0.345, 0.170, 0.170, 0.160, 0.155],
    },
    "Beef cattle ranching and farming, including feedlots [112110]": {
        "working_capital_share": 0.380,
        "variable_term_share": 0.135,
        "yr1_maturing_fixed_share": 0.095,
        "theta_schedule": [0.610, 0.115, 0.105, 0.090, 0.080],
    },
    "Beef cattle ranching and farming, including feedlots [112111]": {
        "working_capital_share": 0.380,
        "variable_term_share": 0.135,
        "yr1_maturing_fixed_share": 0.095,
        "theta_schedule": [0.610, 0.115, 0.105, 0.090, 0.080],
    },
    "Hog and pig farming [1122]": {
        "working_capital_share": 0.260,
        "variable_term_share": 0.150,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.520, 0.140, 0.125, 0.115, 0.100],
    },
    "Sheep and goat farming [1124]": {
        "working_capital_share": 0.260,
        "variable_term_share": 0.150,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.520, 0.140, 0.125, 0.115, 0.100],
    },
    "Other animal production [112A]": {
        "working_capital_share": 0.260,
        "variable_term_share": 0.150,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.520, 0.140, 0.125, 0.115, 0.100],
    },
    "Other animal production [1129]": {
        "working_capital_share": 0.260,
        "variable_term_share": 0.150,
        "yr1_maturing_fixed_share": 0.110,
        "theta_schedule": [0.520, 0.140, 0.125, 0.115, 0.100],
    },
}


# ==============================================================================
# 5. JORGENSON USER-COST-OF-CAPITAL & PECKING-ORDER CASH-FLOW CAPEX ELASTICITIES
#    Source: Weersink & Tauer (1989) AJAE; Turvey, Weersink & Godah (CJAE);
#    Bierlen & Featherstone (1998); Barry, Bierlen & Sotomayor (2000) AJAE;
#    Deep Research Output #5 (Block 2)
# ==============================================================================

JORGENSON_CAPEX_ELASTICITIES: Dict[str, Any] = {
    "lambda_partial_adjustment": 0.620,
    "lambda_range": (0.550, 0.680),
    "speed_of_adjustment": 0.380,  # 1 - lambda
    "eps_I_c_sr": -0.220,
    "eps_I_c_sr_range": (-0.280, -0.170),
    "eps_I_c_lr": -0.550,
    "eps_I_c_lr_range": (-0.650, -0.450),
    # Semi-elasticity d(ln I)/dr per +100 bps (+1.0 percentage point) interest shock
    "semi_elast_composite_per_100bps_sr": -0.0147,
    "semi_elast_composite_per_100bps_lr": -0.0367,
    "semi_elast_machinery_per_100bps_sr": -0.0126,
    "semi_elast_machinery_per_100bps_lr": -0.0314,
    "semi_elast_structures_gh_per_100bps_sr": -0.0211,
    "semi_elast_structures_gh_per_100bps_lr": -0.0526,
    # Pecking-Order Internal Liquidity / Cash-Flow Elasticities (d ln I / d ln NCI)
    "eps_I_nci_healthy_sr": 0.240,    # DSCR >= 1.50
    "eps_I_nci_healthy_lr": 0.480,
    "eps_I_nci_normal_sr": 0.320,     # 1.15 <= DSCR < 1.50
    "eps_I_nci_normal_lr": 0.640,
    "eps_I_nci_stressed_sr": 0.580,   # DSCR < 1.15 (covenant-binding credit constraint)
    "eps_I_nci_stressed_lr": 0.850,
    # Direct Year-1 Operating-Credit Rationing Effect on Physical Output per +100 bps
    # Near-zero due to APP ($100k-$500k interest-free) and FCC Alliance working capital buffers
    "yr1_output_semi_elast_per_100bps": -0.0015,  # -0.15% per +100 bps
    "lr_output_semi_elast_per_100bps": -0.0115,   # -1.15% per +100 bps (via K_t erosion)
    "yr1_acreage_semi_elast_per_100bps": -0.0004, # -0.04% per +100 bps
    "lr_acreage_semi_elast_per_100bps": -0.0025,  # -0.25% per +100 bps
    # Economic Depreciation Rates delta (Deep Research Output #5 Block 2 & StatCan 36-10-0096-01):
    # delta_mach = 12.0%/yr (farm machinery & equipment), delta_struct = 4.0%/yr (barns & greenhouses),
    # delta_composite = 9.5%/yr (weighted primary agriculture capital stock)
    "delta_machinery": 0.120,
    "delta_structures_gh": 0.040,
    "delta_composite": 0.095,
}


# ==============================================================================
# 6. HOTELLING'S LEMMA DUALITY OUTPUT SUPPLY & INPUT DEMAND ELASTICITIES
#    WITH RESPECT TO INPUT PRICES (eps_{Y, w_i}^{SR/LR} & eps_{X_i, w_i}^{SR})
#    Source: Lopez (1984) AJAE; Moschini (1988) CJAE/AJAE; Chambers (1988);
#    Fox & Weersink (1995) AJAE; StatCan 32-10-0456-01; Deep Research Output #5 (Block 3)
# ==============================================================================

HOTELLING_INPUT_SUPPLY_ELASTICITIES: Dict[str, Dict[str, Any]] = {
    "field_crops_1111": {
        "label": "Field Crops / Grains & Oilseeds (NAICS 1111)",
        "total_input_rev_share": 0.435,
        "sum_eps_Y_w_sr": -0.115,
        "sum_eps_Y_w_lr": -0.365,
        "yr1_margin_share": 0.872,
        "yr1_output_share": 0.128,
        "lr_margin_share": 0.615,
        "lr_output_share": 0.385,
        "inputs": {
            "fertilizer": {"s_i": 0.170, "eps_X_w_sr": -0.260, "eps_Y_w_sr": -0.048, "eps_Y_w_lr": -0.145, "yr1_margin_share": 0.865, "lr_margin_share": 0.600},
            "fuel":       {"s_i": 0.075, "eps_X_w_sr": -0.140, "eps_Y_w_sr": -0.020, "eps_Y_w_lr": -0.062, "yr1_margin_share": 0.882, "lr_margin_share": 0.625},
            "pesticides": {"s_i": 0.075, "eps_X_w_sr": -0.150, "eps_Y_w_sr": -0.016, "eps_Y_w_lr": -0.052, "yr1_margin_share": 0.895, "lr_margin_share": 0.640},
            "seed":       {"s_i": 0.085, "eps_X_w_sr": -0.110, "eps_Y_w_sr": -0.019, "eps_Y_w_lr": -0.064, "yr1_margin_share": 0.888, "lr_margin_share": 0.610},
            "feed":       {"s_i": 0.000, "eps_X_w_sr":  0.000, "eps_Y_w_sr":  0.000, "eps_Y_w_lr":  0.000, "yr1_margin_share": 1.000, "lr_margin_share": 1.000},
            "labor":      {"s_i": 0.045, "eps_X_w_sr": -0.280, "eps_Y_w_sr": -0.012, "eps_Y_w_lr": -0.042, "yr1_margin_share": 0.860, "lr_margin_share": 0.580},
        },
    },
    "greenhouse_nursery_1114": {
        "label": "Greenhouse, Nursery, Floriculture & Sod (NAICS 1114 — Primary Ag)",
        "total_input_rev_share": 0.625,
        "sum_eps_Y_w_sr": -0.106,
        "sum_eps_Y_w_lr": -0.415,
        "yr1_margin_share": 0.905,
        "yr1_output_share": 0.095,
        "lr_margin_share": 0.580,
        "lr_output_share": 0.420,
        "inputs": {
            "fertilizer": {"s_i": 0.060, "eps_X_w_sr": -0.180, "eps_Y_w_sr": -0.010, "eps_Y_w_lr": -0.036, "yr1_margin_share": 0.908, "lr_margin_share": 0.620},
            "fuel":       {"s_i": 0.145, "eps_X_w_sr": -0.120, "eps_Y_w_sr": -0.028, "eps_Y_w_lr": -0.115, "yr1_margin_share": 0.896, "lr_margin_share": 0.550},
            "pesticides": {"s_i": 0.025, "eps_X_w_sr": -0.140, "eps_Y_w_sr": -0.004, "eps_Y_w_lr": -0.014, "yr1_margin_share": 0.915, "lr_margin_share": 0.650},
            "seed":       {"s_i": 0.115, "eps_X_w_sr": -0.100, "eps_Y_w_sr": -0.019, "eps_Y_w_lr": -0.070, "yr1_margin_share": 0.912, "lr_margin_share": 0.605},
            "feed":       {"s_i": 0.000, "eps_X_w_sr":  0.000, "eps_Y_w_sr":  0.000, "eps_Y_w_lr":  0.000, "yr1_margin_share": 1.000, "lr_margin_share": 1.000},
            "labor":      {"s_i": 0.280, "eps_X_w_sr": -0.220, "eps_Y_w_sr": -0.045, "eps_Y_w_lr": -0.180, "yr1_margin_share": 0.902, "lr_margin_share": 0.565},
        },
    },
    "horticulture_1112_1113": {
        "label": "Field Vegetables, Potatoes & Fruit (NAICS 1112 / 1113)",
        "total_input_rev_share": 0.550,
        "sum_eps_Y_w_sr": -0.112,
        "sum_eps_Y_w_lr": -0.388,
        "yr1_margin_share": 0.888,
        "yr1_output_share": 0.112,
        "lr_margin_share": 0.602,
        "lr_output_share": 0.398,
        "inputs": {
            "fertilizer": {"s_i": 0.090, "eps_X_w_sr": -0.220, "eps_Y_w_sr": -0.020, "eps_Y_w_lr": -0.065, "yr1_margin_share": 0.880, "lr_margin_share": 0.610},
            "fuel":       {"s_i": 0.055, "eps_X_w_sr": -0.130, "eps_Y_w_sr": -0.011, "eps_Y_w_lr": -0.038, "yr1_margin_share": 0.892, "lr_margin_share": 0.620},
            "pesticides": {"s_i": 0.085, "eps_X_w_sr": -0.130, "eps_Y_w_sr": -0.017, "eps_Y_w_lr": -0.058, "yr1_margin_share": 0.894, "lr_margin_share": 0.615},
            "seed":       {"s_i": 0.080, "eps_X_w_sr": -0.100, "eps_Y_w_sr": -0.014, "eps_Y_w_lr": -0.052, "yr1_margin_share": 0.905, "lr_margin_share": 0.615},
            "feed":       {"s_i": 0.000, "eps_X_w_sr":  0.000, "eps_Y_w_sr":  0.000, "eps_Y_w_lr":  0.000, "yr1_margin_share": 1.000, "lr_margin_share": 1.000},
            "labor":      {"s_i": 0.260, "eps_X_w_sr": -0.240, "eps_Y_w_sr": -0.050, "eps_Y_w_lr": -0.175, "yr1_margin_share": 0.882, "lr_margin_share": 0.585},
        },
    },
    "market_livestock_112110_1122": {
        "label": "Market Livestock: Beef & Hogs (NAICS 112110 / 1122)",
        "total_input_rev_share": 0.560,
        "sum_eps_Y_w_sr": -0.118,
        "sum_eps_Y_w_lr": -0.395,
        "yr1_margin_share": 0.885,
        "yr1_output_share": 0.115,
        "lr_margin_share": 0.595,
        "lr_output_share": 0.405,
        "inputs": {
            "fertilizer": {"s_i": 0.035, "eps_X_w_sr": -0.250, "eps_Y_w_sr": -0.007, "eps_Y_w_lr": -0.022, "yr1_margin_share": 0.885, "lr_margin_share": 0.620},
            "fuel":       {"s_i": 0.045, "eps_X_w_sr": -0.140, "eps_Y_w_sr": -0.009, "eps_Y_w_lr": -0.030, "yr1_margin_share": 0.890, "lr_margin_share": 0.615},
            "pesticides": {"s_i": 0.020, "eps_X_w_sr": -0.120, "eps_Y_w_sr": -0.003, "eps_Y_w_lr": -0.011, "yr1_margin_share": 0.910, "lr_margin_share": 0.650},
            "seed":       {"s_i": 0.020, "eps_X_w_sr": -0.120, "eps_Y_w_sr": -0.003, "eps_Y_w_lr": -0.012, "yr1_margin_share": 0.910, "lr_margin_share": 0.640},
            "feed":       {"s_i": 0.370, "eps_X_w_sr": -0.280, "eps_Y_w_sr": -0.082, "eps_Y_w_lr": -0.270, "yr1_margin_share": 0.880, "lr_margin_share": 0.580},
            "labor":      {"s_i": 0.070, "eps_X_w_sr": -0.250, "eps_Y_w_sr": -0.014, "eps_Y_w_lr": -0.050, "yr1_margin_share": 0.882, "lr_margin_share": 0.590},
        },
    },
    "supply_managed_112120_1123": {
        "label": "Supply-Managed: Dairy & Poultry (NAICS 112120 / 1123)",
        "total_input_rev_share": 0.455,
        "sum_eps_Y_w_sr": -0.039,
        "sum_eps_Y_w_lr": -0.095,
        "yr1_margin_share": 0.948,
        "yr1_output_share": 0.052,
        "lr_margin_share": 0.900,
        "lr_output_share": 0.100,
        "inputs": {
            "fertilizer": {"s_i": 0.040, "eps_X_w_sr": -0.220, "eps_Y_w_sr": -0.004, "eps_Y_w_lr": -0.010, "yr1_margin_share": 0.945, "lr_margin_share": 0.900},
            "fuel":       {"s_i": 0.045, "eps_X_w_sr": -0.120, "eps_Y_w_sr": -0.004, "eps_Y_w_lr": -0.010, "yr1_margin_share": 0.950, "lr_margin_share": 0.900},
            "pesticides": {"s_i": 0.015, "eps_X_w_sr": -0.120, "eps_Y_w_sr": -0.001, "eps_Y_w_lr": -0.003, "yr1_margin_share": 0.960, "lr_margin_share": 0.920},
            "seed":       {"s_i": 0.025, "eps_X_w_sr": -0.110, "eps_Y_w_sr": -0.002, "eps_Y_w_lr": -0.005, "yr1_margin_share": 0.955, "lr_margin_share": 0.910},
            "feed":       {"s_i": 0.245, "eps_X_w_sr": -0.220, "eps_Y_w_sr": -0.021, "eps_Y_w_lr": -0.048, "yr1_margin_share": 0.945, "lr_margin_share": 0.890},
            "labor":      {"s_i": 0.085, "eps_X_w_sr": -0.200, "eps_Y_w_sr": -0.007, "eps_Y_w_lr": -0.019, "yr1_margin_share": 0.950, "lr_margin_share": 0.900},
        },
    },
    "all_farm_types_composite": {
        "label": "Composite Primary Agriculture (NAICS 111 & 112 Weighted)",
        "total_input_rev_share": 0.505,
        "sum_eps_Y_w_sr": -0.098,
        "sum_eps_Y_w_lr": -0.325,
        "yr1_margin_share": 0.895,
        "yr1_output_share": 0.105,
        "lr_margin_share": 0.640,
        "lr_output_share": 0.360,
        "inputs": {
            "fertilizer": {"s_i": 0.095, "eps_X_w_sr": -0.240, "eps_Y_w_sr": -0.024, "eps_Y_w_lr": -0.075, "yr1_margin_share": 0.885, "lr_margin_share": 0.620},
            "fuel":       {"s_i": 0.068, "eps_X_w_sr": -0.135, "eps_Y_w_sr": -0.015, "eps_Y_w_lr": -0.050, "yr1_margin_share": 0.895, "lr_margin_share": 0.625},
            "pesticides": {"s_i": 0.045, "eps_X_w_sr": -0.140, "eps_Y_w_sr": -0.009, "eps_Y_w_lr": -0.030, "yr1_margin_share": 0.905, "lr_margin_share": 0.650},
            "seed":       {"s_i": 0.060, "eps_X_w_sr": -0.110, "eps_Y_w_sr": -0.012, "eps_Y_w_lr": -0.042, "yr1_margin_share": 0.900, "lr_margin_share": 0.630},
            "feed":       {"s_i": 0.142, "eps_X_w_sr": -0.250, "eps_Y_w_sr": -0.024, "eps_Y_w_lr": -0.080, "yr1_margin_share": 0.900, "lr_margin_share": 0.650},
            "labor":      {"s_i": 0.095, "eps_X_w_sr": -0.235, "eps_Y_w_sr": -0.014, "eps_Y_w_lr": -0.048, "yr1_margin_share": 0.895, "lr_margin_share": 0.620},
        },
    },
}


# ==============================================================================
# 7. INDUCED FARM-HOUSEHOLD CONSUMPTION MULTIPLIERS (MPC_NCI & CSNA I-O)
#    Source: Langemeier & Patrick (1990) AJAE; Carriker, Langemeier & Schroeder
#    (1993); Loyns & Ahmadi-Esfahani; Jette-Nantel et al. (2011); StatCan Table
#    36-10-0594-01; Deep Research Output #5 (Block 4)
# ==============================================================================

FARM_HOUSEHOLD_MPC_MULTIPLIERS: Dict[str, Any] = {
    "mpc_nci_sr": 0.120,
    "mpc_nci_sr_range": (0.080, 0.150),
    "mpc_nci_sustained": 0.220,
    "mpc_nci_sustained_range": (0.180, 0.280),
    "mpc_non_farm_wage_reference": 0.700,
    "Ontario": {
        "gdp_basic_prices_mult": 0.720,
        "fte_per_million_consumption": 8.10,
        "eff_induced_gdp_per_dollar_nci_sr": 0.0864,         # 0.120 * 0.72
        "eff_induced_gdp_per_dollar_nci_sustained": 0.1584,  # 0.220 * 0.72
        "eff_induced_fte_per_million_nci_sr": 0.972,         # 0.120 * 8.1
        "eff_induced_fte_per_million_nci_sustained": 1.782,  # 0.220 * 8.1
    },
    "Canada": {
        "gdp_basic_prices_mult": 0.880,
        "fte_per_million_consumption": 9.50,
        "eff_induced_gdp_per_dollar_nci_sr": 0.1056,         # 0.120 * 0.88
        "eff_induced_gdp_per_dollar_nci_sustained": 0.1936,  # 0.220 * 0.88
        "eff_induced_fte_per_million_nci_sr": 1.140,         # 0.120 * 9.5
        "eff_induced_fte_per_million_nci_sustained": 2.090,  # 0.220 * 9.5
    },
}


def get_debt_repricing_schedule(region: str = "Ontario", farm_type: str = "All farm types") -> List[float]:
    """Return the 5-year contractual debt repricing schedule [theta_yr1 .. theta_yr5]."""
    if farm_type in DEBT_REPRICING_SCHEDULE:
        return list(DEBT_REPRICING_SCHEDULE[farm_type]["theta_schedule"])
    comp_key = f"{region}|{farm_type}"
    if comp_key in DEBT_REPRICING_SCHEDULE:
        return list(DEBT_REPRICING_SCHEDULE[comp_key]["theta_schedule"])
    if region == "Ontario":
        return list(DEBT_REPRICING_SCHEDULE["Ontario|All farm types"]["theta_schedule"])
    return list(DEBT_REPRICING_SCHEDULE["Canada|All farm types"]["theta_schedule"])


def get_economic_depreciation_rate(region: str = "Ontario", farm_type: str = "All farm types") -> Dict[str, Any]:
    """
    Return the empirical geometric economic depreciation rate delta (% per year)
    for depreciable non-residential farm capital (StatCan Table 36-10-0096-01 &
    Deep Research Output #5 Block 2: delta_mach = 12.0%, delta_struct = 4.0%,
    delta_composite = 9.5%).
    """
    delta_mach = float(JORGENSON_CAPEX_ELASTICITIES["delta_machinery"])       # 0.120 (12.0%/yr)
    delta_struct = float(JORGENSON_CAPEX_ELASTICITIES["delta_structures_gh"]) # 0.040 (4.0%/yr)
    ft = str(farm_type)

    if "1114" in ft or "Greenhouse" in ft or "nursery" in ft.lower():
        # Controlled-environment greenhouse/nursery: ~65% glass/poly structures + 35% HVAC/LED/machinery
        w_mach, w_struct = 0.35, 0.65
        label = "NAICS 1114 CEA Structures (65% @ 4.0%) + HVAC/Machinery (35% @ 12.0%)"
    elif any(k in ft for k in ["11212", "1123", "Dairy", "Poultry"]):
        # Supply-managed livestock: ~60% barns/parlours + 40% robotic milkers/feeding equipment
        w_mach, w_struct = 0.40, 0.60
        label = "Dairy/Poultry Barns & Storage (60% @ 4.0%) + Barn Equipment (40% @ 12.0%)"
    elif any(k in ft for k in ["1111", "Oilseed", "grain", "Potato", "1112"]):
        # Field crops & potatoes: ~75% tractors/combines/planters + 25% grain bins/tile/sheds
        w_mach, w_struct = 0.75, 0.25
        label = "Row-Crop Tractors/Combines (75% @ 12.0%) + Grain Storage/Tile (25% @ 4.0%)"
    else:
        # Composite Primary Agriculture (Deep Research Output #5 Block 2 benchmark: delta = 9.5%)
        w_mach, w_struct = 0.6875, 0.3125
        label = "Composite Primary Ag Benchmark (68.8% Machinery @ 12.0% + 31.2% Structures @ 4.0%)"

    blended_pct = round((w_mach * delta_mach + w_struct * delta_struct) * 100.0, 1)
    return {
        "delta_pct": blended_pct,
        "delta_dec": blended_pct / 100.0,
        "delta_mach_pct": delta_mach * 100.0,
        "delta_struct_pct": delta_struct * 100.0,
        "w_mach": w_mach,
        "w_struct": w_struct,
        "basis_label": label,
    }


def get_hotelling_sector_profile(farm_type: str) -> Dict[str, Any]:
    """Map any StatCan NAICS farm_type string to its Hotelling Duality elasticity profile."""
    ft = str(farm_type)
    if "1114" in ft or "Greenhouse" in ft or "nursery" in ft.lower():
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["greenhouse_nursery_1114"]
    if "1111" in ft or "Oilseed" in ft or "grain" in ft.lower():
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["field_crops_1111"]
    if any(code in ft for code in ["1112", "1113", "Potato", "vegetable", "Fruit"]):
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["horticulture_1112_1113"]
    if any(code in ft for code in ["11212", "1123", "Dairy", "Poultry"]):
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["supply_managed_112120_1123"]
    if any(code in ft for code in ["11211", "1122", "1124", "112A", "1129", "Beef", "Hog", "Animal production [112]"]):
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["market_livestock_112110_1122"]
    if "Crop production [111]" in ft:
        return HOTELLING_INPUT_SUPPLY_ELASTICITIES["field_crops_1111"]
    return HOTELLING_INPUT_SUPPLY_ELASTICITIES["all_farm_types_composite"]


# ==============================================================================
# 8. FARM DEBT PRINCIPAL AMORTIZATION (gamma_principal), DISTRESS WORKING-CAPITAL
#    BORROWING (phi_debt), AND FFSC / FCC SOLVENCY CREDIT-SCORING WEIGHTS
#    Source: StatCan 32-10-0051-01 & FFS; FCC Corporate Plan & Senate Testimony;
#    Splett et al. / Barry, Escalante & Ellinger (AFR); Deep Research Output #6 (Section 1)
# ==============================================================================

DEBT_AMORTIZATION_AND_DISTRESS_PARAMS: Dict[str, Any] = {
    "Canada": {
        "w_op": 0.175,
        "w_int": 0.235,
        "w_lt": 0.590,
        "gamma_op": 0.0180,
        "gamma_int": 0.1480,
        "gamma_lt": 0.0565,
        "gamma_principal": 0.0713,  # 7.13%/yr blended principal repayment
        "gamma_principal_ci95": (0.0660, 0.0765),
    },
    "Ontario": {
        "w_op": 0.155,
        "w_int": 0.205,
        "w_lt": 0.640,
        "gamma_op": 0.0180,
        "gamma_int": 0.1480,
        "gamma_lt": 0.0510,
        "gamma_principal": 0.0658,  # 6.58%/yr blended principal repayment
        "gamma_principal_ci95": (0.0610, 0.0710),
    },
    # Distress Working-Capital Borrowing vs. Retained-Earnings Drawdown (phi_debt = -d(Delta Debt_{t+1})/d(NCI_t))
    "phi_debt_shallow": 0.580,      # DSCR_{t-1} >= 1.15 (58% operating credit / 42% liquid buffer drawdown)
    "phi_equity_shallow": 0.420,
    "phi_debt_acute": 0.820,        # DSCR_{t-1} < 1.15 or consecutive deficit (82% FCC Alliance / 10-yr term-out)
    "phi_equity_acute": 0.180,
    "phi_debt_blended": 0.680,      # Unconditional blended parameter (60% shallow / 40% acute)
    "phi_equity_blended": 0.320,
    # FFSC / Barry-Escalante-Ellinger / FDMA Normalized Two-Factor Cash-Flow Resilience Index Weights
    "lrf_w_dscr": 0.620,            # 62% weight on Repayment Capacity (DSCR)
    "lrf_w_fsf": 0.380,             # 38% weight on Interest-to-Revenue Efficiency (FSF)
    # Structural Logit Probability of Farm Liquidity Distress: Pr(Distress_t = 1)
    "distress_logit": {
        "beta_0": -2.40,            # Baseline log-odds (P_0 = 8.3% migration at benchmark)
        "beta_dscr": -3.15,         # Marginal effect: -0.240 per +1.0 DSCR
        "dscr_benchmark": 1.10,
        "beta_fsf": 14.80,          # Marginal effect: +0.113 per +0.01 (+1%) FSF
        "fsf_benchmark": 0.10,      # 10% Interest-to-Revenue benchmark
        "beta_dar": 4.20,           # Marginal effect: +0.032 per +0.01 (+1%) DAR
        "dar_benchmark": 0.17,      # 17% Canadian sector average Debt-to-Asset ratio
    },
}


# ==============================================================================
# 9. FORWARD COST-PUSH PRICE TRANSMISSION (psi_cost) FROM PRIMARY AGRICULTURE
#    TO DOMESTIC FOOD PROCESSORS (NAICS 311/312) BY MARKET REGIME
#    Source: CDC National Pricing Formula (2024-2027); CFO 8-Week ON COPF;
#    OPVG Arbitration Records; Deep Research Output #6 (Section 2)
# ==============================================================================

PROCESSOR_FORWARD_COST_PASS_THROUGH: Dict[str, Dict[str, Any]] = {
    "grains_oilseeds_1111": {
        "label": "Grains & Oilseeds (NAICS 1111 -> 3111/3112)",
        "regime": "Export-Traded Price-Taking (CBOT/ICE Parity)",
        "psi_sr": 0.020,
        "psi_total": 0.020,
        "ci95": (0.000, 0.050),
        "lag_months": 0.0,
    },
    "beef_cattle_112110": {
        "label": "Beef Cattle & Feedlots (NAICS 112110 -> 311611)",
        "regime": "Export-Traded Price-Taking (CME Parity + Herd Cycle)",
        "psi_sr": 0.000,
        "psi_total": 0.060,
        "ci95": (0.030, 0.090),
        "lag_months": 30.0,
    },
    "hogs_pork_1122": {
        "label": "Hogs & Pig Farming (NAICS 1122 -> 311614)",
        "regime": "Export-Traded Price-Taking (CME Lean Hogs Parity)",
        "psi_sr": 0.000,
        "psi_total": 0.050,
        "ci95": (0.020, 0.080),
        "lag_months": 21.0,
    },
    "dairy_112120": {
        "label": "Dairy Cattle & Milk (NAICS 112120 -> 3115)",
        "regime": "Supply-Managed CDC NPF (50% iCOP + 50% CPI, s_NPF=0.865)",
        "psi_sr": 0.200,            # 0.45 weight in year t * 0.433 isolated COP pass-through
        "psi_total": 0.433,         # 0.433 isolated COP (0.680 under correlated macro CPI)
        "psi_macro_cpi": 0.680,
        "ci95": (0.400, 0.680),
        "lag_months": 7.5,          # 6-9 months (Aug cutoff -> Feb 1 pricing)
    },
    "poultry_eggs_1123": {
        "label": "Poultry & Eggs (NAICS 1123 -> 311615)",
        "regime": "Supply-Managed 100% COP Formula (CFO 8-Week A-Period / EFC)",
        "psi_sr": 0.790,            # 88% realized within year t * 0.90
        "psi_total": 0.900,
        "ci95": (0.850, 0.940),
        "lag_months": 1.8,
    },
    "proc_vegetables_1112": {
        "label": "Processing Vegetables & Potatoes (NAICS 1112 -> 3114)",
        "regime": "OPVG Pre-Plant Bilateral Negotiation / Final-Offer Arbitration",
        "psi_sr": 0.100,
        "psi_total": 0.380,
        "ci95": (0.300, 0.450),
        "lag_months": 8.0,
    },
    "fruit_tree_nut_1113": {
        "label": "Fruit & Tree Nut Farming (NAICS 1113 -> 3114/3121)",
        "regime": "Import-Competing Wholesale & Processor Contracts",
        "psi_sr": 0.100,
        "psi_total": 0.250,
        "ci95": (0.180, 0.320),
        "lag_months": 9.0,
    },
    "greenhouse_nursery_1114": {
        "label": "Greenhouse, Nursery & Floriculture (NAICS 1114 -> 3114/4131)",
        "regime": "Seasonal/Annual Retail & Packer Contracts vs. US/MX",
        "psi_sr": 0.150,
        "psi_total": 0.320,
        "ci95": (0.250, 0.400),
        "lag_months": 6.0,
    },
}


# ==============================================================================
# 10. ANNUAL TIME-SERIES PERSISTENCE (AR(1) / MA(1)) OF AGRICULTURAL MACRO SHOCKS
#     FOR 10-YEAR MONTE CARLO SIMULATION (1990-2024)
#     Source: StatCan CPA, 32-10-0049-01, 32-10-0051-01, 18-10-0258-01;
#     Deep Research Output #6 (Section 3)
# ==============================================================================

ARMA_MONTE_CARLO_PERSISTENCE: Dict[str, Dict[str, Any]] = {
    "Canada": {
        "theta_tfp_ma1": -0.680,            # MA(1) negative bounce-back after drought/weather shock
        "theta_tfp_ci95": (-0.750, -0.600),
        "rho_r_portfolio_ar1": 0.870,       # AR(1) effective portfolio interest rate persistence
        "rho_r_marginal_ar1": 0.790,        # AR(1) marginal new-loan interest rate persistence
        "rho_fipi_ar1": 0.450,              # AR(1) FIPI input inflation persistence
        "rho_fipi_ci95": (0.380, 0.520),
        "rho_nci_ar1": 0.540,               # AR(1) real Net Cash Income ratio persistence
        "rho_capex_ar1": 0.360,             # AR(1) real Farm CapEx growth persistence
    },
    "Ontario": {
        "theta_tfp_ma1": -0.760,            # Stronger 1-yr weather bounce-back in Great Lakes basin
        "theta_tfp_ci95": (-0.830, -0.680),
        "rho_r_portfolio_ar1": 0.870,
        "rho_r_marginal_ar1": 0.790,
        "rho_fipi_ar1": 0.480,
        "rho_fipi_ci95": (0.410, 0.550),
        "rho_nci_ar1": 0.580,
        "rho_capex_ar1": 0.390,
    },
}


# ==============================================================================
# 11. AGRISTABILITY & ALL-BRM EFFECTIVE MARGINAL PAYOUT RATES (beta_BRM)
#     Source: StatCan 32-10-0045-01 & 32-10-0106-01; Sustainable CAP (2023-2027)
#     Handbook; Ontario Agricorp RMP & SDRM; Deep Research Output #6 (Section 4)
# ==============================================================================

BRM_EFFECTIVE_PAYOUT_PARAMETERS: Dict[str, Any] = {
    "c_stat_scap": 0.800,           # 80% statutory compensation beyond 30% deductible (2023-2027)
    "c_stat_2025_adhoc": 0.900,     # 90% temporary 2025 enhancement ($6M cap)
    "deductible_d": 0.300,          # 30% deductible (70% reference margin trigger)
    "lambda_allowable": 0.780,      # 78% of StatCan NCI expenses are AgriStability-allowable (interest, rent, repairs excluded)
    "sigma_idiosyncratic": 0.250,   # Cross-farm margin dispersion sigma_id = 0.25
    # Cash-Flow Timing Split in StatCan 32-10-0045-01 (w_t in shock year, w_{t+1} in following tax-filing year)
    "timing_split": {
        "shallow":  {"w_t": 0.40, "w_t1": 0.60},
        "moderate": {"w_t": 0.45, "w_t1": 0.55},
        "severe":   {"w_t": 0.55, "w_t1": 0.45},
    },
    # Effective Marginal Payout Rate beta_BRM = -Delta(Direct Payments) / Delta(Pre-Program Margin)
    "regime_rates": {
        "agristability_standalone": {"shallow": 0.044, "moderate": 0.092, "severe": 0.158},
        "Canada_price_cost":        {"shallow": 0.065, "moderate": 0.125, "severe": 0.185},
        "Ontario_price_cost_rmp":   {"shallow": 0.210, "moderate": 0.225, "severe": 0.245},
        "Canada_yield_weather":     {"shallow": 0.280, "moderate": 0.430, "severe": 0.520},
        "Ontario_yield_weather":    {"shallow": 0.300, "moderate": 0.460, "severe": 0.540},
    },
    # Sub-Sector Participation & Program Eligibility Multipliers
    "subsector_rates": {
        "grains_1111": {
            "p_agristab": 0.34, "mult": 1.03,
            "can_price": 0.130, "ont_price_rmp": 0.240, "yield_ins": 0.480,
        },
        "hogs_hort_1122_1112_1114": {
            "p_agristab": 0.56, "mult": 1.70,
            "can_price": 0.195, "ont_price_rmp": 0.285, "yield_ins": 0.310,
        },
        "beef_112110": {
            "p_agristab": 0.28, "mult": 0.85,
            "can_price": 0.105, "ont_price_rmp": 0.215, "yield_ins": 0.290,
        },
        "supply_managed_112120_1123": {
            "p_agristab": 0.04, "mult": 0.15,
            "can_price": 0.015, "ont_price_rmp": 0.015, "yield_ins": 0.080,
        },
    },
}


def get_debt_amortization_params(region: str = "Ontario") -> Dict[str, Any]:
    """Return maturity-weighted principal amortization and distress borrowing parameters."""
    reg_key = "Ontario" if region == "Ontario" else "Canada"
    logit = DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["distress_logit"]
    return {
        **DEBT_AMORTIZATION_AND_DISTRESS_PARAMS[reg_key],
        "phi_debt_shallow": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["phi_debt_shallow"],
        "phi_debt_normal": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["phi_debt_shallow"],
        "phi_debt_acute": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["phi_debt_acute"],
        "phi_debt_distress": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["phi_debt_acute"],
        "phi_debt_blended": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["phi_debt_blended"],
        "dscr_distress_trigger": 1.150,
        "lrf_w_dscr": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["lrf_w_dscr"],
        "w_dscr": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["lrf_w_dscr"],
        "lrf_w_fsf": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["lrf_w_fsf"],
        "w_fsf": DEBT_AMORTIZATION_AND_DISTRESS_PARAMS["lrf_w_fsf"],
        "distress_logit": logit,
        "logit_intercept": logit["beta_0"],
        "logit_beta_dscr": logit["beta_dscr"],
        "logit_dscr_anchor": logit["dscr_benchmark"],
        "logit_beta_fsf": logit["beta_fsf"],
        "logit_fsf_anchor": logit["fsf_benchmark"],
        "logit_beta_dar": logit["beta_dar"],
        "logit_dar_anchor": logit["dar_benchmark"],
    }


def get_processor_cost_pass_through(
    proc_code: Optional[str] = None,
    farm_type: str = "All farm types",
    is_long_run: bool = False,
    p_code: Optional[str] = None,
) -> float:
    """
    Return the empirical forward cost-push pass-through coefficient psi_cost
    from primary agriculture to downstream food processor `proc_code` (Section 2).
    """
    code = str(proc_code if proc_code is not None else (p_code or ""))
    ft = str(farm_type)
    key = "psi_total" if is_long_run else "psi_sr"
    if code == "3115" or "11212" in ft or "Dairy" in ft:
        return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["dairy_112120"][key])
    if "1123" in ft or "Poultry" in ft:
        return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["poultry_eggs_1123"][key])
    if code == "3114":
        if "1114" in ft or "Greenhouse" in ft:
            return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["greenhouse_nursery_1114"][key])
        if "1113" in ft or "Fruit" in ft:
            return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["fruit_tree_nut_1113"][key])
        return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["proc_vegetables_1112"][key])
    if code in ("3111", "3112") or "1111" in ft:
        return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["grains_oilseeds_1111"][key])
    if code == "3116":
        if "11211" in ft or "Beef" in ft:
            return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["beef_cattle_112110"][key])
        if "1122" in ft or "Hog" in ft:
            return float(PROCESSOR_FORWARD_COST_PASS_THROUGH["hogs_pork_1122"][key])
        # Composite 3116 (Beef + Pork + Poultry): Poultry accounts for ~26% of 3116 value
        return 0.26 * float(PROCESSOR_FORWARD_COST_PASS_THROUGH["poultry_eggs_1123"][key]) + 0.74 * (
            0.055 if is_long_run else 0.000
        )
    return 0.020


def compute_effective_brm_marginal_payout(
    region: str = "Ontario",
    pre_program_margin_drop_dollars: Optional[float] = None,
    baseline_nci_dollars: float = 1.0,
    allowable_shock_share: float = 0.78,
    is_yield_shock: bool = False,
    farm_type: str = "All farm types",
    allowable_margin_shortfall_dollars: Optional[float] = None,
    interest_shortfall_dollars: float = 0.0,
    labor_shortfall_dollars: float = 0.0,
    shock_type: Optional[str] = None,
    brm_regime: str = "statutory_scap_80pct",
    olympic_ref_ratio: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute empirical All-BRM (AgriStability + AgriInsurance + Ontario Agricorp RMP/SDRM)
    marginal compensation payout for a given pre-program margin drop (Section 4).
    Note: Interest expense shocks are non-allowable under AgriStability (allowable_shock_share=0).
    Supports:
      - `brm_regime="statutory_scap_80pct"`: Sustainable CAP 80% compensation beyond 30% deductible
      - `brm_regime="adhoc_2025_90pct"`: 2025/2026 temporary 90% compensation enhancement ($6M cap)
      - `olympic_ref_ratio`: 5-year Olympic average reference margin retention factor (1.0 in Yrs 1-2)
    """
    if allowable_margin_shortfall_dollars is not None:
        eff_shortfall = max(0.0, float(allowable_margin_shortfall_dollars) + 0.78 * float(labor_shortfall_dollars))
    else:
        eff_shortfall = max(0.0, float(pre_program_margin_drop_dollars or 0.0))

    if eff_shortfall <= 0.0 or baseline_nci_dollars <= 0.0:
        return {
            "total_brm_payout_dollars": 0.0,
            "total_brm_entitlement": 0.0,
            "yr_t_cash_payout_dollars": 0.0,
            "yr1_cash_payout": 0.0,
            "yr_t1_lagged_payout_dollars": 0.0,
            "yr2_lagged_payout": 0.0,
            "effective_beta_brm": 0.0,
            "beta_brm_effective": 0.0,
            "olympic_ref_ratio": float(olympic_ref_ratio),
            "severity_tier": "none",
        }

    if shock_type is not None:
        is_yield_shock = ("yield" in str(shock_type).lower()) or ("weather" in str(shock_type).lower())

    drop_pct = eff_shortfall / max(baseline_nci_dollars, 1.0)
    tier = "shallow" if drop_pct <= 0.15 else ("moderate" if drop_pct <= 0.30 else "severe")
    tier_label = (
        "shallow_under_15pct"
        if tier == "shallow"
        else ("moderate_15_to_30pct" if tier == "moderate" else "severe_over_30pct")
    )

    reg_rates = BRM_EFFECTIVE_PAYOUT_PARAMETERS["regime_rates"]
    is_ont = (region == "Ontario")
    if is_yield_shock:
        regime_key = "Ontario_yield_weather" if is_ont else "Canada_yield_weather"
    else:
        regime_key = "Ontario_price_cost_rmp" if is_ont else "Canada_price_cost"

    raw_regime_beta = float(reg_rates[regime_key][tier])
    agristab_base_beta = float(reg_rates["agristability_standalone"][tier])
    non_agristab_beta = max(0.0, raw_regime_beta - agristab_base_beta)

    # Scale AgriStability tranche for 90% ad-hoc enhancement (0.90 / 0.80 = 1.125x)
    # and 5-year Olympic reference margin ratcheting in Years 3+ (while RMP/SDRM/Insurance remain COP/yield-indexed)
    comp_rate_mult = (0.900 / 0.800) if brm_regime == "adhoc_2025_90pct" else 1.0
    ref_retention = float(np.clip(olympic_ref_ratio, 0.25, 1.0))
    base_beta = (agristab_base_beta * comp_rate_mult * ref_retention) + non_agristab_beta

    # Sub-sector participation scaling
    ft = str(farm_type)
    sub_mult = 1.0
    if any(k in ft for k in ["11212", "1123", "Dairy", "Poultry"]):
        sub_mult = 0.15
    elif any(k in ft for k in ["1122", "1112", "1113", "1114", "Hog", "Potato", "vegetable", "Fruit", "Greenhouse"]):
        sub_mult = 1.25 if is_ont else 1.50
    elif any(k in ft for k in ["11211", "Beef"]):
        sub_mult = 0.90

    if allowable_margin_shortfall_dollars is not None:
        allow_scalar = 1.0
    else:
        allow_scalar = float(np.clip(allowable_shock_share / 0.78, 0.0, 1.25)) if not is_yield_shock else 1.0
    eff_beta = float(np.clip(base_beta * sub_mult * allow_scalar, 0.0, 0.65))

    total_payout = eff_shortfall * eff_beta
    timing = BRM_EFFECTIVE_PAYOUT_PARAMETERS["timing_split"][tier]
    yr_t_payout = total_payout * float(timing["w_t"])
    yr_t1_payout = total_payout * float(timing["w_t1"])

    return {
        "total_brm_payout_dollars": total_payout,
        "total_brm_entitlement": total_payout,
        "yr_t_cash_payout_dollars": yr_t_payout,
        "yr1_cash_payout": yr_t_payout,
        "yr_t1_lagged_payout_dollars": yr_t1_payout,
        "yr2_lagged_payout": yr_t1_payout,
        "effective_beta_brm": eff_beta,
        "beta_brm_effective": eff_beta,
        "olympic_ref_ratio": ref_retention,
        "severity_tier": tier_label,
    }


