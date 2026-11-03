"""
Smart data reader with Parquet-first loading and CSV fallback.

When a Parquet file exists alongside a CSV, it is loaded instead for
faster I/O and smaller memory footprint.  If only the CSV exists, it
is loaded transparently.  All downstream code sees the same DataFrame.

Usage:
    from app.smart_read import smart_read
    df = smart_read(Path("data/latest/my_table.csv"))
"""

from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def smart_read(csv_path: Path, **csv_kwargs) -> pd.DataFrame:
    """Read data from Parquet if available, otherwise fall back to CSV.

    Parameters
    ----------
    csv_path : Path
        The *CSV* file path.  The function checks for a ``.parquet`` file
        with the same stem in the same directory first.
    **csv_kwargs
        Extra keyword arguments forwarded to ``pd.read_csv`` when the CSV
        fallback is used (e.g. ``low_memory=False``).

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    if csv_path.suffix == ".gz":
        parquet_path = csv_path.with_suffix("").with_suffix(".parquet")
    else:
        parquet_path = csv_path.with_suffix(".parquet")

    df = None
    if parquet_path.exists():
        try:
            logger.debug("Loading Parquet: %s", parquet_path)
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning("Parquet read failed for %s (%s); falling back to CSV", parquet_path, e)
            df = None

    if df is None and csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            logger.error(f"OOM Guard: Attempted to load {csv_path} ({size_mb:.1f} MB)")
            raise RuntimeError(
                f"OOM Guard: Refusing to load {csv_path.name} ({size_mb:.1f} MB) into memory. "
                "Files > 100MB must use a _dashboard subset or Parquet conversion."
            )
        logger.debug("Loading CSV (fallback or no Parquet found): %s", csv_path)
        df = pd.read_csv(csv_path, **csv_kwargs)
    
    if df is not None:
        # Memory Optimization: Convert low-cardinality object columns to category
        if len(df) > 100:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # If unique values < 50% of length, convert to category
                    if df[col].nunique() / len(df) < 0.5:
                        df[col] = df[col].astype('category')
        return df

    raise FileNotFoundError(
        f"Neither Parquet ({parquet_path}) nor CSV ({csv_path}) found."
    )
