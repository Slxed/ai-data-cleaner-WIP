import pandas as pd
import numpy as np

RAW = "raw data/example_data.csv"
OUT = "clean data/nyc311_clean.csv"

def log(step, df, before):
    print(f"{step}: {before - len(df)} rows removed, {df.shape[1]} columns left")

def load(path):
    return pd.read_csv(path, low_memory=False)

def drop_sparse_columns(df, threshold=0.95):
    sparse_cols = df.columns[df.isna().mean() >= threshold]
    return df.drop(columns=sparse_cols)

def standardize_text(df):
    missing_vals = ["", "N/A", "Unspecified", "n/a", "unspecified", "NAN", "nan"]
    df = df.replace(missing_vals, np.nan)
    for col in ["City", "Borough", "Agency"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    return df

def fix_types(df):
    date_cols = ["Created Date", "Closed Date", "Due Date"]
    for col in date_cols:
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

def main():
    df = load(RAW)
    for step in (drop_sparse_columns, standardize_text, fix_types, drop_bad_rows):
        before = len(df)
        df = step(df)
        log(step.__name__, df, before)
    df.to_csv(OUT, index=False)

if __name__ == "__main__":
    main()