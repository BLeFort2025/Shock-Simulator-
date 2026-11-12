"""
Reproducible Empirical Audit & Update for `us_section338_tariffs` in `data/trade_app/trade_exposure.db`.

Reconciles 6-digit CIMT trade statistics against 8-digit HTSUS statutory scope across:
  - Proclamations 11046, 11047, 11048 (Aug 22, 2026: +50% ad valorem Section 338 duty)
  - Proclamations 11064 & 11065 (Sept 15, 2026: +50% cheese escalation across 24 subheadings in 0406, incl. goat cheese 0406.90.99)
  - Proclamations 11061 & 11062 (Sept 29, 2026: U.S. import prohibition on packaged beverage alcohol 2203-2208 and whey/molasses 0404.10, 1703)

Fixes two major 6-digit over-aggregations identified in the Deep Research audit:
  1. HS Chapter 06 (Procl. 11048 / HTSUS 9903.03.14): Exempts soil-grown potted bedding plants,
     perennials, woody nursery trees/shrubs, roses, rhododendrons, and fresh Christmas foliage
     (0602.20, 0602.30, 0602.40, 0604.20 = 0.0 midpoint; 0602.90 = 0.09 midpoint for live orchids .20
     & mushroom spawn .50; 0601.20 = 0.12 midpoint for hyacinths without soil .10). Leaves ~$175M CAD
     of Ontario's $660.2M HS Ch. 06 U.S. exports targeted rather than 100%.
  2. HS Chapter 04 (Procl. 11047 / 11064): Exempts retail flavored yogurt (0403.20, $104.9M ON) and
     butter (0405.10, $10.0M ON) outside the statutory annexes, while tagging 0406.* cheese under
     Sept 15 Procl. 11064/11065 and 0404.10 whey + 1703 molasses + packaged 2203-2208 alcohol under
     the Sept 29 Procl. 11061/11062 U.S. import prohibition.
"""

from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "trade_app" / "trade_exposure.db"


def _norm_hs6(val) -> str:
    """Normalize float or string hs6_code (e.g., 602.9 or '0602.90') to 6-digit string '060290'."""
    s = str(val).strip().replace(".", "")
    try:
        num = int(round(float(val) * 100))
        return f"{num:06d}"
    except Exception:
        return s.zfill(6)[:6]


def update_section338_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(us_section338_tariffs)").fetchall()}
    if "effective_date" not in existing_cols:
        cur.execute("ALTER TABLE us_section338_tariffs ADD COLUMN effective_date TEXT DEFAULT '2026-08-22'")
    if "statutory_regime" not in existing_cols:
        cur.execute(
            "ALTER TABLE us_section338_tariffs ADD COLUMN statutory_regime TEXT "
            "DEFAULT '+50% Section 338 Duty (Aug 22, 2026)'"
        )

    rows = cur.execute("SELECT rowid, proclamation, htsus_8digit, hs6_code, hs_chapter, description FROM us_section338_tariffs").fetchall()

    for rowid, procl, hts8, hs6_raw, chap, desc in rows:
        hs6 = _norm_hs6(hs6_raw)
        new_procl = str(procl)
        eff_date = "2026-08-22"
        regime = "+50% Section 338 Duty (Aug 22, 2026)"
        cons_f, mid_f, up_f = None, None, None
        new_desc = desc

        # ------------------------------------------------------------------
        # 1. HS Chapter 06 — Procl. 11048 (HTSUS 9903.03.14) 8-digit filter
        # ------------------------------------------------------------------
        if hs6 in {"060220", "060230", "060240", "060420"}:
            cons_f, mid_f, up_f = 0.00, 0.00, 0.15
            eff_date = "Exempt"
            regime = "Exempt Soil-Grown / Fresh Foliage (Outside HTSUS 9903.03.14)"
        elif hs6 == "060290":
            # Only live orchids (0602.90.20) & mushroom spawn (0602.90.50) targeted;
            # potted greenhouse bedding/herbaceous perennials in soil (.30/.40/.60/.90) exempt!
            cons_f, mid_f, up_f = 0.05, 0.065, 0.12
            eff_date = "2026-08-22"
            regime = "Partial 8-Digit Scope: Live Orchids (.20) & Spawn (.50) Only (+50% Duty)"
            new_desc = "Live plants nesoi: ONLY orchids (.20) & mushroom spawn (.50) targeted; soil potted plants exempt"
        elif hs6 == "060120":
            # Only hyacinth bulbs without soil (0601.20.10) targeted; potted bulbs in soil exempt
            cons_f, mid_f, up_f = 0.08, 0.12, 0.20
            eff_date = "2026-08-22"
            regime = "Partial 8-Digit Scope: Hyacinth Bulbs w/o Soil (.10) Only (+50% Duty)"
            new_desc = "Bulbs/tubers in growth: ONLY hyacinths w/o soil (0601.20.10) targeted; soil-grown exempt"
        elif hs6 in {"060110", "060210", "060311", "060313", "060314", "060315", "060319", "060390", "060490"}:
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            eff_date = "2026-08-22"
            regime = "+50% Section 338 Duty (Aug 22, 2026 — Procl. 11048)"

        # ------------------------------------------------------------------
        # 2. HS Chapter 04 — Dairy, Honey, Cheese Escalation & Whey Ban
        # ------------------------------------------------------------------
        elif hs6 in {"040320", "040510"}:
            cons_f, mid_f, up_f = 0.00, 0.00, 0.10
            eff_date = "Exempt"
            regime = "Exempt Retail Dairy Line (Outside Procl. 11047 / 11064)"
        elif hs6.startswith("0406"):
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            new_procl = "11064/11065"
            eff_date = "2026-09-15"
            regime = "+50% Cheese Escalation (Sept 15, 2026 — Procl. 11064/11065, Incl. Goat Cheese 0406.90.99)"
            if hs6 == "040690":
                new_desc = "Other cheeses (cheddar, gouda, specialty & unpooled Ontario goat cheese 0406.90.99)"
        elif hs6 == "040410":
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            new_procl = "11047/11062"
            eff_date = "2026-09-29"
            regime = "U.S. Import Prohibition (Sept 29, 2026 — Procl. 11062; +50% Aug 22)"
        elif hs6 == "040900":
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            eff_date = "2026-08-22"
            regime = "+50% Section 338 Duty (Aug 22, 2026 — Procl. 11048 Primary Honey)"
        elif hs6.startswith("040"):
            eff_date = "2026-08-22"
            regime = "+50% Section 338 Duty (Aug 22, 2026 — Procl. 11047)"

        # ------------------------------------------------------------------
        # 3. HS Chapters 17, 19, 35, 12, 33 — Ingredients, Mixes, Molasses Ban
        # ------------------------------------------------------------------
        elif hs6 in {"170310", "170390"}:
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            new_procl = "11047/11062"
            eff_date = "2026-09-29"
            regime = "U.S. Import Prohibition (Sept 29, 2026 — Procl. 11062; +50% Aug 22)"
        elif hs6 == "190120":
            # Dairy-containing baker mixes (>10% milk solids under U.S. dairy Chapter 19 TRQ lines)
            cons_f, mid_f, up_f = 0.05, 0.08, 0.16
            eff_date = "2026-08-22"
            regime = "Partial 8-Digit Scope: Dairy-Solids Mixes Only (+50% Duty — Procl. 11047)"
        elif hs6 == "170260":
            cons_f, mid_f, up_f = 0.50, 0.80, 1.00
            eff_date = "2026-08-22"
            regime = "+50% Section 338 Duty (Aug 22, 2026 — Procl. 11047)"

        # ------------------------------------------------------------------
        # 4. HS Chapter 22 — Beverage Alcohol (+50% Aug 22 -> Sept 29 Ban on Packaged)
        # ------------------------------------------------------------------
        elif hs6 == "220429":
            # Bulk wine >10L remains subject to +50% duty rather than packaged retail container ban
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            new_procl = "11046"
            eff_date = "2026-08-22"
            regime = "+50% Bulk Wine (>10L) Duty (Aug 22, 2026 — Procl. 11046; Outside Packaged Ban)"
        elif hs6 == "220710":
            # Only beverage-grade undenatured ethyl alcohol (2207.10.30) targeted; industrial fuel ethanol exempt
            cons_f, mid_f, up_f = 0.01, 0.01, 0.03
            new_procl = "11046/11061"
            eff_date = "2026-09-29"
            regime = "Beverage Ethyl Only (2207.10.30 — Sept 29 Ban; Fuel Ethanol Exempt)"
        elif hs6.startswith("220"):
            cons_f, mid_f, up_f = 1.00, 1.00, 1.00
            new_procl = "11046/11061"
            eff_date = "2026-09-29"
            regime = "U.S. Import Prohibition on Packaged Alcohol (Sept 29, 2026 — Procl. 11061; +50% Aug 22)"

        if cons_f is not None:
            cur.execute(
                """
                UPDATE us_section338_tariffs
                SET proclamation = ?,
                    description = ?,
                    conservative_fraction = ?,
                    midpoint_fraction = ?,
                    upper_bound_fraction = ?,
                    effective_date = ?,
                    statutory_regime = ?
                WHERE rowid = ?
                """,
                (new_procl, new_desc, cons_f, mid_f, up_f, eff_date, regime, rowid),
            )
        else:
            cur.execute(
                """
                UPDATE us_section338_tariffs
                SET proclamation = ?,
                    effective_date = ?,
                    statutory_regime = ?
                WHERE rowid = ?
                """,
                (new_procl, eff_date, regime, rowid),
            )

    conn.commit()

    # Audit query joining Ontario 2025 HS6 exports against updated midpoint_fraction
    sht_cols = {row[1] for row in cur.execute("PRAGMA table_info(state_hs_trade)").fetchall()}
    country_clause = "AND country = 'United States'" if "country" in sht_cols else ""
    audit_q = f"""
    WITH on_2025 AS (
        SELECT hs6_clean, hs2_chapter, SUM(value_cad)/1e6 AS on_2025_exp_m
        FROM state_hs_trade
        WHERE province = 'Ontario'
          {country_clause}
          AND trade_type LIKE '%export%'
          AND substr(ref_date, 1, 4) = '2025'
        GROUP BY hs6_clean, hs2_chapter
    )
    SELECT
        t.hs_chapter,
        t.proclamation,
        t.effective_date,
        t.statutory_regime,
        COUNT(*) AS lines,
        ROUND(SUM(COALESCE(o.on_2025_exp_m, 0)), 2) AS gross_hs6_exp_m,
        ROUND(SUM(COALESCE(o.on_2025_exp_m, 0) * t.conservative_fraction), 2) AS targeted_cons_m,
        ROUND(SUM(COALESCE(o.on_2025_exp_m, 0) * t.midpoint_fraction), 2) AS targeted_mid_m,
        ROUND(SUM(COALESCE(o.on_2025_exp_m, 0) * t.upper_bound_fraction), 2) AS targeted_upper_m
    FROM us_section338_tariffs t
    LEFT JOIN on_2025 o
      ON printf('%06d', CAST(ROUND(t.hs6_code * 100) AS INTEGER)) = o.hs6_clean
    GROUP BY t.hs_chapter, t.proclamation, t.effective_date, t.statutory_regime
    ORDER BY t.hs_chapter, targeted_mid_m DESC
    """
    summary_df = pd.read_sql_query(audit_q, conn)
    conn.close()
    return summary_df


if __name__ == "__main__":
    df_sum = update_section338_table()
    print("=== UPDATED ONTARIO 2025 TARGETED SECTION 338 EXPOSURE SUMMARY ($M CAD) ===")
    print(df_sum.to_string(index=False))
    print("\n=== TOTAL BY CHAPTER ($M CAD) ===")
    by_chap = df_sum.groupby("hs_chapter")[["gross_hs6_exp_m", "targeted_cons_m", "targeted_mid_m", "targeted_upper_m"]].sum()
    print(by_chap.round(2).to_string())
    print(f"\nGrand Total Ontario Targeted Midpoint Exposure: ${by_chap['targeted_mid_m'].sum():,.2f}M CAD")
