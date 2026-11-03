"""Farm Input Cost Engine — powers Page 15 (Farm Input Cost Tracker).

Loads StatCan Farm Input Price Index (FIPI, table 18-10-0258-01) and
provides clean, analysis-ready methods for trend display, snapshots,
and year-over-year comparisons.

FIPI data is a quarterly index (2021=100) covering 31 input categories
across Canada and provinces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.smart_read import smart_read


# ── Category Groupings ──────────────────────────────────────────────────────
# Human-friendly groups mapping to the raw "Price index" column values

CATEGORY_GROUPS = {
    "Fuel & Energy": ["Machinery fuel"],
    "Fertilizer": ["Fertilizer", "Nitrogen fertilizers", "Other fertilizers"],
    "Crop Protection": ["Pesticides"],
    "Seed & Plant": ["Commercial seed and plant"],
    "Machinery & Equipment": [
        "Machinery and motor vehicles",
        "Machine repairs",
        "Depreciation on machinery and motor vehicles",
    ],
    "Labour": ["Cash wages in crop production", "Cash wages in animal production"],
    "Feed": ["Commercial feed", "Grain feed", "Prepared feed"],
    "Livestock Inputs": [
        "Livestock purchases",
        "Veterinary fees and drugs",
    ],
    "General Business": [
        "General business costs",
        "Twine, wire and containers",
        "Production insurance",
        "Stabilization premiums",
    ],
}

# Flat list of all leaf categories for validation
ALL_CATEGORIES = sorted({
    cat for cats in CATEGORY_GROUPS.values() for cat in cats
} | {
    "Farm input total",
    "Crop production",
    "Animal production",
    "Crop-related custom work",
    "Animal-related custom work",
    "Buildings",
    "Cattle",
    "Hogs",
    "Poultry",
    "Depreciation on machinery",
    "Depreciation on motor vehicles",
})

# Headline categories shown in KPI cards
HEADLINE_CATEGORIES = [
    "Fertilizer",
    "Machinery fuel",
    "Pesticides",
    "Commercial seed and plant",
    "Farm input total",
]

VALID_GEOS = [
    "Canada",
    "Ontario",
    "Alberta",
    "Saskatchewan",
    "Quebec",
    "Manitoba",
    "British Columbia",
    "New Brunswick",
    "Nova Scotia",
    "Prince Edward Island",
    "Newfoundland and Labrador",
    "Eastern Canada",
    "Western Canada",
]

# Live market tickers for real-time commodity signals (via yfinance)
LIVE_MARKET_TICKERS = {
    "NG=F": {
        "name": "Natural Gas",
        "unit": "USD/MMBtu",
        "group": "Energy",
        "impact": "Fertilizer",
        "exchange": "NYMEX",
        "source_url": "https://finance.yahoo.com/quote/NG%3DF/",
        "description": "Primary feedstock for ammonia/urea production — "
                       "nat gas spikes directly drive nitrogen fertilizer costs.",
    },
    "CL=F": {
        "name": "Crude Oil (WTI)",
        "unit": "USD/bbl",
        "group": "Energy",
        "impact": "Fuel",
        "exchange": "NYMEX",
        "source_url": "https://finance.yahoo.com/quote/CL%3DF/",
        "description": "Drives diesel and gasoline prices — directly impacts "
                       "machinery fuel and transportation costs.",
    },
    "BZ=F": {
        "name": "Brent Crude",
        "unit": "USD/bbl",
        "group": "Energy",
        "impact": "Fuel",
        "exchange": "ICE",
        "source_url": "https://finance.yahoo.com/quote/BZ%3DF/",
        "description": "International oil benchmark — affects global "
                       "freight and input logistics costs.",
    },
    "ZC=F": {
        "name": "Corn Futures",
        "unit": "USD/bu",
        "group": "Grain",
        "impact": "Feed",
        "exchange": "CBOT",
        "source_url": "https://finance.yahoo.com/quote/ZC%3DF/",
        "description": "Key feed input for livestock — rising corn prices "
                       "increase animal production costs.",
    },
    "ZS=F": {
        "name": "Soybean Futures",
        "unit": "USD/bu",
        "group": "Grain",
        "impact": "Feed",
        "exchange": "CBOT",
        "source_url": "https://finance.yahoo.com/quote/ZS%3DF/",
        "description": "Soybean meal is a critical protein feed ingredient — "
                       "affects livestock and dairy operations.",
    },
    "ZW=F": {
        "name": "Wheat Futures",
        "unit": "USD/bu",
        "group": "Grain",
        "impact": "Feed",
        "exchange": "CBOT",
        "source_url": "https://finance.yahoo.com/quote/ZW%3DF/",
        "description": "Feed wheat alternative — price spikes cascade through "
                       "grain feed markets.",
    },
    "CAD=X": {
        "name": "USD/CAD Rate",
        "unit": "CAD per USD",
        "group": "Currency",
        "impact": "All Imports",
        "exchange": "FOREX",
        "source_url": "https://finance.yahoo.com/quote/CAD%3DX/",
        "description": "Weaker CAD makes all USD-priced imports (fertilizer, fuel, "
                       "machinery, chemicals) more expensive for Canadian farmers.",
    },
}


# ── Helper ──────────────────────────────────────────────────────────────────

def _ref_date_to_quarter_label(ref_date: str) -> str:
    """Convert StatCan REF_DATE (e.g. '2024-07') to 'Q3 2024'."""
    try:
        parts = str(ref_date).split("-")
        year = parts[0]
        month = int(parts[1])
        q = (month - 1) // 3 + 1
        return f"Q{q} {year}"
    except (IndexError, ValueError):
        return str(ref_date)


def _ref_date_to_datetime(ref_date: str) -> pd.Timestamp:
    """Convert StatCan REF_DATE (e.g. '2024-07') to a Timestamp."""
    try:
        return pd.Timestamp(str(ref_date) + "-01")
    except Exception:
        return pd.NaT


# ── Engine ──────────────────────────────────────────────────────────────────

class InputCostEngine:
    """Loads and queries FIPI data for the Farm Input Cost Tracker page."""

    FIPI_TABLE = "18-10-0258-01"

    def __init__(self):
        csv_path = Path(f"data/latest/{self.FIPI_TABLE}.csv")
        raw = smart_read(csv_path)

        # Keep only rows with valid numeric VALUE
        raw["VALUE"] = pd.to_numeric(raw["VALUE"], errors="coerce")
        self._df = raw.dropna(subset=["VALUE"]).copy()

        # Parse dates (convert REF_DATE to str first so Categorical columns produce datetime64[ns])
        ref_str = self._df["REF_DATE"].astype(str)
        self._df["date"] = pd.to_datetime(ref_str + "-01", errors="coerce")
        self._df["quarter_label"] = ref_str.apply(
            _ref_date_to_quarter_label
        )
        self._df = self._df.sort_values("date")

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def available_categories(self) -> list[str]:
        """Return sorted list of all available 'Price index' values."""
        return sorted(self._df["Price index"].unique())

    @property
    def available_geos(self) -> list[str]:
        """Return sorted list of all available geographies."""
        return sorted(self._df["GEO"].unique())

    def get_category_trend(
        self, category: str, geo: str = "Ontario"
    ) -> pd.DataFrame:
        """Time series for a single input category in one geography.

        Returns DataFrame with columns: date, quarter_label, value
        """
        self._validate(category, geo)
        mask = (self._df["Price index"] == category) & (self._df["GEO"] == geo)
        result = (
            self._df.loc[mask, ["date", "quarter_label", "VALUE"]]
            .rename(columns={"VALUE": "value"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        return result

    def get_latest_snapshot(self, geo: str = "Ontario") -> pd.DataFrame:
        """Latest quarter values for all categories in a geography.

        Returns DataFrame with columns: category, value, quarter_label
        """
        geo_df = self._df[self._df["GEO"] == geo]
        if geo_df.empty:
            raise ValueError(f"No data for geography: {geo}")

        latest_date = geo_df["date"].max()
        snap = (
            geo_df[geo_df["date"] == latest_date]
            [["Price index", "VALUE", "quarter_label"]]
            .rename(columns={"Price index": "category", "VALUE": "value"})
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )
        return snap

    def get_yoy_changes(self, geo: str = "Ontario") -> pd.DataFrame:
        """Year-over-year percentage change for each category.

        Compares the latest quarter to the same quarter one year prior.

        Returns DataFrame with columns: category, current, prior, yoy_pct, quarter_label
        """
        geo_df = self._df[self._df["GEO"] == geo].copy()
        if geo_df.empty:
            raise ValueError(f"No data for geography: {geo}")

        latest_date = geo_df["date"].max()
        prior_date = latest_date - pd.DateOffset(years=1)

        latest = (
            geo_df[geo_df["date"] == latest_date]
            .set_index("Price index")["VALUE"]
        )
        prior = (
            geo_df[geo_df["date"] == prior_date]
            .set_index("Price index")["VALUE"]
        )

        both = pd.DataFrame({"current": latest, "prior": prior}).dropna()
        both["yoy_pct"] = ((both["current"] - both["prior"]) / both["prior"]) * 100
        both = both.reset_index().rename(columns={"Price index": "category"})
        both["quarter_label"] = _ref_date_to_quarter_label(
            latest_date.strftime("%Y-%m")
        )
        return both.sort_values("yoy_pct", ascending=False).reset_index(drop=True)

    def get_multi_category_comparison(
        self,
        categories: list[str],
        geo: str = "Ontario",
    ) -> pd.DataFrame:
        """Multiple category trends for overlay comparison.

        Returns DataFrame with columns: date, quarter_label, category, value
        """
        for cat in categories:
            self._validate(cat, geo)

        mask = (
            self._df["Price index"].isin(categories) & (self._df["GEO"] == geo)
        )
        result = (
            self._df.loc[mask, ["date", "quarter_label", "Price index", "VALUE"]]
            .rename(columns={"Price index": "category", "VALUE": "value"})
            .sort_values(["date", "category"])
            .reset_index(drop=True)
        )
        return result

    def get_headline_kpis(self, geo: str = "Ontario") -> list[dict]:
        """Return KPI data for the headline categories.

        Each dict has: category, value, qoq_change, yoy_change, quarter_label
        """
        geo_df = self._df[self._df["GEO"] == geo]
        if geo_df.empty:
            return []

        latest_date = geo_df["date"].max()

        # Find prior quarter and prior year dates
        all_dates = sorted(geo_df["date"].dropna().unique())
        latest_idx = list(all_dates).index(latest_date) if latest_date in all_dates else -1
        prior_q_date = all_dates[latest_idx - 1] if latest_idx > 0 else None
        prior_y_date = latest_date - pd.DateOffset(years=1)

        kpis = []
        for cat in HEADLINE_CATEGORIES:
            cat_df = geo_df[geo_df["Price index"] == cat]
            if cat_df.empty:
                continue

            latest_val = cat_df[cat_df["date"] == latest_date]["VALUE"]
            val = float(latest_val.iloc[0]) if not latest_val.empty else None
            if val is None:
                continue

            # Quarter-over-quarter
            qoq = None
            if prior_q_date is not None:
                prior_q_val = cat_df[cat_df["date"] == prior_q_date]["VALUE"]
                if not prior_q_val.empty:
                    pv = float(prior_q_val.iloc[0])
                    qoq = ((val - pv) / pv) * 100 if pv != 0 else None

            # Year-over-year
            yoy = None
            prior_y_val = cat_df[cat_df["date"] == prior_y_date]["VALUE"]
            if not prior_y_val.empty:
                py = float(prior_y_val.iloc[0])
                yoy = ((val - py) / py) * 100 if py != 0 else None

            kpis.append({
                "category": cat,
                "value": val,
                "qoq_change": qoq,
                "yoy_change": yoy,
                "quarter_label": _ref_date_to_quarter_label(
                    latest_date.strftime("%Y-%m")
                ),
            })

        return kpis

    # ── Fertilizer Benchmarks (World Bank Pink Sheet) ──────────────────

    def get_fertilizer_benchmarks(
        self, start_year: int = 2010
    ) -> pd.DataFrame | None:
        """Load World Bank Pink Sheet fertilizer benchmark prices.

        Returns DataFrame with columns: date, urea_usd, dap_usd, tsp_usd,
        potash_usd, phosphate_rock_usd.  Returns None if data not fetched.
        """
        try:
            csv_path = Path("data/latest/fertilizer_benchmarks.csv")
            df = smart_read(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"].dt.year >= start_year].sort_values("date")
            return df.reset_index(drop=True)
        except FileNotFoundError:
            return None

    def get_fertilizer_vs_fipi(
        self, geo: str = "Ontario", start_year: int = 2010
    ) -> pd.DataFrame | None:
        """Merge global benchmark prices with FIPI Fertilizer index.

        Returns monthly DataFrame with: date, urea_usd, dap_usd, potash_usd,
        fipi_fertilizer (quarterly, forward-filled to monthly).
        Returns None if benchmark data unavailable.
        """
        benchmarks = self.get_fertilizer_benchmarks(start_year)
        if benchmarks is None:
            return None

        # Get FIPI Fertilizer trend for the selected geo
        try:
            fipi = self.get_category_trend("Fertilizer", geo)
        except ValueError:
            return None

        if fipi.empty:
            return None

        # Merge on date — FIPI is quarterly, benchmarks are monthly,
        # so we do an outer merge then forward-fill the FIPI values
        merged = pd.merge(
            benchmarks,
            fipi[["date", "value"]].rename(columns={"value": "fipi_fertilizer"}),
            on="date",
            how="left",
        )
        merged["fipi_fertilizer"] = merged["fipi_fertilizer"].ffill()
        merged = merged.dropna(subset=["fipi_fertilizer"])
        return merged.reset_index(drop=True)

    def get_fertilizer_kpis(self) -> list[dict] | None:
        """Latest month benchmark prices with month-over-month change.

        Returns list of dicts with: commodity, price, mom_change, date_label
        """
        benchmarks = self.get_fertilizer_benchmarks(start_year=2020)
        if benchmarks is None or len(benchmarks) < 2:
            return None

        from scripts.fetch_fertilizer_benchmarks import FERTILIZER_LABELS

        latest = benchmarks.iloc[-1]
        prior = benchmarks.iloc[-2]
        date_label = latest["date"].strftime("%b %Y")

        kpis = []
        price_cols = [c for c in benchmarks.columns if c.endswith("_usd")]
        for col in price_cols:
            val = latest[col]
            prev = prior[col]
            if pd.notna(val) and pd.notna(prev) and prev != 0:
                mom = ((val - prev) / prev) * 100
            else:
                mom = None

            kpis.append({
                "commodity": FERTILIZER_LABELS.get(col, col),
                "price": float(val) if pd.notna(val) else None,
                "mom_change": mom,
                "date_label": date_label,
            })

        return kpis

    # ── Internals ───────────────────────────────────────────────────────

    def _validate(self, category: str, geo: str):
        available_cats = self._df["Price index"].unique()
        if category not in available_cats:
            raise ValueError(
                f"Unknown category: '{category}'. "
                f"Available: {sorted(available_cats)}"
            )
        available_geos = self._df["GEO"].unique()
        if geo not in available_geos:
            raise ValueError(
                f"Unknown geography: '{geo}'. "
                f"Available: {sorted(available_geos)}"
            )
