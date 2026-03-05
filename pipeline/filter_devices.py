"""
Loads the raw FDA 510(k) PMN flat file and filters it down to:
  - Neurostimulation product codes defined in config
  - Cleared decisions only
  - Submissions on or after MIN_SUBMISSION_YEAR

Writes the result to DEVICES_FILTERED_PATH.
"""

import pandas as pd

from config import (
    CLEARED_DECISION_CODES,
    DEVICES_FILTERED_PATH,
    MIN_SUBMISSION_YEAR,
    NEUROSTIMULATION_PRODUCT_CODES,
    PMN_RAW_PATH,
    PMN_RELAT_PATH,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column constants
# The FDA flat file uses fixed column names; define them once here.
# ---------------------------------------------------------------------------

COL_KNUMBER = "KNUMBER"
COL_APPLICANT = "APPLICANT"
COL_DEVICENAME = "DEVICENAME"
COL_PRODUCTCODE = "PRODUCTCODE"
COL_DECISION = "DECISION"
COL_DECISIONDATE = "DECISIONDATE"
COL_PREDICATENUMBER = "PREDICATENUMBER"

# PREDICATENUMBER comes from pmnrelat.zip, not the main PMN flat file
REQUIRED_COLUMNS = [
    COL_KNUMBER,
    COL_APPLICANT,
    COL_DEVICENAME,
    COL_PRODUCTCODE,
    COL_DECISION,
    COL_DECISIONDATE,
]


def load_pmn_records() -> pd.DataFrame:
    """
    Load the raw PMN flat file into a DataFrame.

    Returns:
        Raw DataFrame with all records.

    Raises:
        FileNotFoundError: If the raw file does not exist.
        ValueError: If expected columns are missing.
    """
    if not PMN_RAW_PATH.exists():
        raise FileNotFoundError(
            f"PMN records not found at {PMN_RAW_PATH}. "
            "Run download_data.download_pmn_records() first."
        )

    logger.info("Loading PMN records from %s", PMN_RAW_PATH)

    df = pd.read_csv(
        PMN_RAW_PATH,
        sep="|",              # FDA flat files use pipe delimiter
        encoding="latin-1",  # Handles non-UTF-8 characters common in older records
        low_memory=False,
        dtype=str,            # Load everything as string; we parse types explicitly
    )

    logger.info("Loaded %d raw records with %d columns", len(df), len(df.columns))
    logger.debug("Columns found: %s", list(df.columns))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in PMN file: {missing}")

    return df



def _load_predicate_relations() -> pd.DataFrame:
    """
    Load the FDA predicate relationship file and return a DataFrame
    mapping each K-number to its predicate K-number(s).

    The pmnrelat file has two columns: KNUMBER and PREDICATENUMBER.
    Multiple rows per K-number are possible when a submission lists
    more than one predicate.

    Returns:
        DataFrame with columns [KNUMBER, PREDICATENUMBER], or an empty
        DataFrame if the file does not exist.
    """
    if not PMN_RELAT_PATH.exists():
        logger.warning(
            "Predicate relations file not found at %s — "
            "predicate data will be absent from filtered records",
            PMN_RELAT_PATH,
        )
        return pd.DataFrame(columns=[COL_KNUMBER, COL_PREDICATENUMBER])

    logger.info("Loading predicate relations from %s", PMN_RELAT_PATH)

    df = pd.read_csv(
        PMN_RELAT_PATH,
        sep="|",
        encoding="latin-1",
        low_memory=False,
        dtype=str,
    )

    logger.debug("Predicate relation columns: %s", list(df.columns))

    # Normalise column names — FDA may use different capitalisation
    df.columns = [c.strip().upper() for c in df.columns]

    if COL_KNUMBER not in df.columns:
        logger.warning(
            "KNUMBER column not found in predicate relations file. "
            "Columns present: %s",
            list(df.columns),
        )
        return pd.DataFrame(columns=[COL_KNUMBER, COL_PREDICATENUMBER])

    # Identify the predicate column — FDA uses various names
    predicate_col = next(
        (c for c in df.columns if "PREDICATE" in c),
        None,
    )
    if predicate_col is None:
        logger.warning(
            "No predicate column found in relations file. "
            "Columns present: %s",
            list(df.columns),
        )
        return pd.DataFrame(columns=[COL_KNUMBER, COL_PREDICATENUMBER])

    df = df[[COL_KNUMBER, predicate_col]].copy()
    df.columns = [COL_KNUMBER, COL_PREDICATENUMBER]
    df[COL_KNUMBER] = df[COL_KNUMBER].str.strip().str.upper()
    df[COL_PREDICATENUMBER] = df[COL_PREDICATENUMBER].str.strip().str.upper()

    logger.info(
        "Loaded %d predicate relationships covering %d unique K-numbers",
        len(df),
        df[COL_KNUMBER].nunique(),
    )
    return df


def _join_predicate_relations(
    devices_df: pd.DataFrame,
    relat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join predicate K-numbers onto the filtered devices DataFrame.

    Where a device has multiple predicates, they are concatenated into
    a single semicolon-separated string to match the expected format
    in load_graph.py.

    Args:
        devices_df: Filtered devices DataFrame.
        relat_df: Predicate relations DataFrame.

    Returns:
        devices_df with a PREDICATENUMBER column added.
    """
    if relat_df.empty:
        devices_df = devices_df.copy()
        devices_df[COL_PREDICATENUMBER] = ""
        return devices_df

    # Aggregate multiple predicates per K-number into one semicolon string
    aggregated = (
        relat_df.groupby(COL_KNUMBER)[COL_PREDICATENUMBER]
        .apply(lambda x: ";".join(x.dropna().unique()))
        .reset_index()
    )

    merged = devices_df.merge(aggregated, on=COL_KNUMBER, how="left")
    merged[COL_PREDICATENUMBER] = merged[COL_PREDICATENUMBER].fillna("")

    has_predicate = (merged[COL_PREDICATENUMBER] != "").sum()
    logger.info(
        "Predicate join: %d/%d filtered devices have at least one predicate",
        has_predicate,
        len(merged),
    )
    return merged


def _filter_by_product_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose PRODUCTCODE is in the neurostimulation target set.

    Args:
        df: Raw PMN DataFrame.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    mask = df[COL_PRODUCTCODE].str.strip().isin(NEUROSTIMULATION_PRODUCT_CODES)
    filtered = df[mask].copy()
    after = len(filtered)

    logger.info(
        "Product code filter: %d → %d records (kept codes: %s)",
        before,
        after,
        NEUROSTIMULATION_PRODUCT_CODES,
    )

    code_counts = (
        filtered[COL_PRODUCTCODE]
        .value_counts()
        .to_dict()
    )
    logger.info("Records per product code: %s", code_counts)

    return filtered


def _filter_by_decision(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only cleared submissions (SESE or SE decision codes).

    Args:
        df: DataFrame already filtered by product code.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    mask = df[COL_DECISION].str.strip().isin(CLEARED_DECISION_CODES)
    filtered = df[mask].copy()
    after = len(filtered)

    logger.info(
        "Decision filter: %d → %d records (kept decisions: %s)",
        before,
        after,
        CLEARED_DECISION_CODES,
    )
    return filtered


def _parse_decision_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse DECISIONDATE into a datetime column and add a YEAR column.

    Args:
        df: DataFrame with DECISIONDATE as string.

    Returns:
        DataFrame with parsed dates and a new YEAR column.
    """
    df = df.copy()
    df[COL_DECISIONDATE] = pd.to_datetime(
        df[COL_DECISIONDATE], errors="coerce"
    )
    df["YEAR"] = df[COL_DECISIONDATE].dt.year
    unparseable = df[COL_DECISIONDATE].isna().sum()
    if unparseable:
        logger.warning(
            "%d records had unparseable DECISIONDATE values", unparseable
        )
    return df


def _filter_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only submissions on or after MIN_SUBMISSION_YEAR.

    Args:
        df: DataFrame with a YEAR column.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    filtered = df[df["YEAR"] >= MIN_SUBMISSION_YEAR].copy()
    after = len(filtered)
    logger.info(
        "Year filter (>= %d): %d → %d records",
        MIN_SUBMISSION_YEAR,
        before,
        after,
    )
    return filtered


def _clean_k_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise K-numbers to uppercase with no surrounding whitespace.

    Args:
        df: DataFrame with KNUMBER column.

    Returns:
        DataFrame with cleaned KNUMBER values.
    """
    df = df.copy()
    df[COL_KNUMBER] = df[COL_KNUMBER].str.strip().str.upper()
    return df


def _log_predicate_coverage(df: pd.DataFrame) -> None:
    """
    Log how many records have a non-null predicate number.

    Args:
        df: Filtered DataFrame.
    """
    total = len(df)
    has_predicate = df[COL_PREDICATENUMBER].notna().sum()
    logger.info(
        "Predicate coverage: %d/%d records have a PREDICATENUMBER (%.1f%%)",
        has_predicate,
        total,
        100 * has_predicate / total if total else 0,
    )


def filter_devices() -> pd.DataFrame:
    """
    Full filter pipeline: load raw records, apply all filters, write output.

    Returns:
        Filtered DataFrame written to DEVICES_FILTERED_PATH.
    """
    df = load_pmn_records()
    df = _filter_by_product_code(df)
    df = _filter_by_decision(df)
    df = _parse_decision_date(df)
    df = _filter_by_year(df)
    df = _clean_k_numbers(df)

    relat_df = _load_predicate_relations()
    df = _join_predicate_relations(df, relat_df)

    _log_predicate_coverage(df)

    df.to_csv(DEVICES_FILTERED_PATH, index=False)
    logger.info(
        "Filtered dataset (%d records) written to %s",
        len(df),
        DEVICES_FILTERED_PATH,
    )

    return df
