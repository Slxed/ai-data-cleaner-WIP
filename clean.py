import json
import sys

import numpy as np
import pandas as pd

from config import CLEAN_CSV, CLEAN_DIR, CLEANING_LOG, RAW_CSV

SPARSE_THRESHOLD = 0.95
PLACEHOLDERS = ["", "N/A", "Unspecified", "n/a", "unspecified", "NAN", "nan"]
TEXT_COLUMNS = ["City", "Borough", "Agency"]
DATE_COLUMNS = ["Created Date", "Closed Date", "Due Date"]


def load(path):
    return pd.read_csv(path, low_memory=False)


def drop_sparse_columns(df, threshold=SPARSE_THRESHOLD):
    sparse_cols = df.columns[df.isna().mean() >= threshold]
    return df.drop(columns=sparse_cols)


def standardize_text(df):
    df = df.replace(PLACEHOLDERS, np.nan)
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    return df


def fix_types(df):
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Closed Date" in df.columns and "Created Date" in df.columns:
        df["resolution_hours"] = (df["Closed Date"] - df["Created Date"]).dt.total_seconds() / 3600
    if "Incident Zip" in df.columns:
        def clean_zip(val):
            if pd.isna(val):
                return np.nan
            s = str(val).split(".")[0].strip()
            if s.isdigit() and len(s) <= 5:
                return s.zfill(5)
            return np.nan
        df["Incident Zip"] = df["Incident Zip"].apply(clean_zip)
    return df


def drop_bad_rows(df):
    df = df.drop_duplicates()
    lat_lon_cols = [col for col in ["Latitude", "Longitude"] if col in df.columns]
    if lat_lon_cols:
        df = df.dropna(subset=lat_lon_cols)
    return df


def missing_counts(df):
    counts = df.isna().sum()
    return {col: int(counts[col]) for col in df.columns}


def text_examples(df):
    """A few real values from the text columns, skipping placeholders."""
    skip = {item.lower() for item in PLACEHOLDERS}
    examples = {}
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            continue
        values = []
        for value in df[col].dropna().astype(str).unique():
            if value.strip().lower() in skip:
                continue
            values.append(str(value))
            if len(values) == 3:
                break
        examples[col] = values
    return examples


def detail_before(step_name, df):
    detail = {}
    if step_name == "drop_sparse_columns":
        detail["sparse_threshold"] = SPARSE_THRESHOLD
    if step_name == "standardize_text":
        detail["placeholders_treated_as_missing"] = PLACEHOLDERS
        detail["title_cased_columns"] = [col for col in TEXT_COLUMNS if col in df.columns]
        detail["examples_before"] = text_examples(df)
    if step_name == "drop_bad_rows":
        detail["duplicate_rows"] = int(df.duplicated().sum())
        lat_lon = [col for col in ("Latitude", "Longitude") if col in df.columns]
        if lat_lon:
            detail["rows_missing_coordinates"] = int(df[lat_lon].isna().any(axis=1).sum())
    return detail


def clean_dataframe(df):
    """Run the four cleaning steps and return (cleaned frame, log dict)."""
    df = df.copy()
    log = {
        "dataset": "311 Service Requests from 2020 to Present",
        "rows_before": int(len(df)),
        "columns_before": int(df.shape[1]),
        "steps": [],
    }
    for step in (drop_sparse_columns, standardize_text, fix_types, drop_bad_rows):
        rows_before = int(len(df))
        cols_before = list(df.columns)
        missing_before = missing_counts(df)
        detail = detail_before(step.__name__, df)
        df = step(df)
        if step.__name__ == "standardize_text":
            detail["examples_after"] = text_examples(df)
        cols_after = list(df.columns)
        entry = {
            "name": step.__name__,
            "rows_before": rows_before,
            "rows_after": int(len(df)),
            "columns_before": len(cols_before),
            "columns_after": len(cols_after),
            "rows_removed": rows_before - int(len(df)),
            "columns_dropped": [col for col in cols_before if col not in cols_after],
            "columns_added": [col for col in cols_after if col not in cols_before],
            "missing_counts_before": missing_before,
            "missing_counts_after": missing_counts(df),
            "detail": detail,
        }
        log["steps"].append(entry)
    log["rows_after"] = int(len(df))
    log["columns_after"] = int(df.shape[1])
    log["columns_after_names"] = [str(col) for col in df.columns]
    return df, log


def print_step(entry):
    print(f"{entry['name']}: {entry['rows_removed']} rows removed, {entry['columns_after']} columns left")
    if entry["columns_dropped"]:
        print("  columns dropped:", ", ".join(entry["columns_dropped"]))
    if entry["columns_added"]:
        print("  columns added:", ", ".join(entry["columns_added"]))


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else RAW_CSV
    try:
        df = load(source)
    except FileNotFoundError:
        print(f"Could not find {source}")
        print('Pass a path if you want the full file: python clean.py "raw data\\file.csv"')
        sys.exit(1)

    cleaned, log = clean_dataframe(df)
    log["source"] = source
    log["output_csv"] = CLEAN_CSV
    for entry in log["steps"]:
        print_step(entry)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    with open(CLEANING_LOG, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    print(f"saved {CLEAN_CSV}")
    print(f"saved {CLEANING_LOG}")


if __name__ == "__main__":
    main()
