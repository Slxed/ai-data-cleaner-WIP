import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean import (
    clean_dataframe,
    drop_bad_rows,
    drop_sparse_columns,
    fix_types,
    standardize_text,
)
from config import RAW_CSV


class CleanTests(unittest.TestCase):
    def test_drop_sparse_columns_uses_95_percent(self):
        frame = pd.DataFrame(
            {
                "kept": [1] * 20,
                "half_empty": [None] * 10 + [1] * 10,
                "mostly_empty": [None] * 19 + [1],
            }
        )
        cleaned = drop_sparse_columns(frame)
        self.assertIn("kept", cleaned.columns)
        self.assertIn("half_empty", cleaned.columns)
        self.assertNotIn("mostly_empty", cleaned.columns)

    def test_standardize_text_fills_placeholders_and_title_cases(self):
        frame = pd.DataFrame(
            {
                "Borough": ["  BROOKLYN ", "Unspecified", "N/A"],
                "Agency": ["NYPD", "n/a", "HPD"],
                "City": [" QUEENS", "unspecified", "Bronx"],
            }
        )
        cleaned = standardize_text(frame)
        self.assertEqual(cleaned["Borough"].iloc[0], "Brooklyn")
        self.assertTrue(pd.isna(cleaned["Borough"].iloc[1]))
        self.assertTrue(pd.isna(cleaned["Borough"].iloc[2]))
        self.assertEqual(cleaned["Agency"].iloc[0], "Nypd")
        self.assertTrue(pd.isna(cleaned["Agency"].iloc[1]))

    def test_fix_types_dates_zip_and_resolution_hours(self):
        frame = pd.DataFrame(
            {
                "Created Date": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "Closed Date": ["2024-01-01 03:30:00", "not a date"],
                "Incident Zip": [10001.0, "7302"],
            }
        )
        cleaned = fix_types(frame)
        self.assertEqual(cleaned["resolution_hours"].iloc[0], 3.5)
        self.assertTrue(pd.isna(cleaned["resolution_hours"].iloc[1]))
        self.assertEqual(cleaned["Incident Zip"].iloc[0], "10001")
        self.assertEqual(cleaned["Incident Zip"].iloc[1], "07302")

    def test_fix_types_rejects_zip_codes_that_are_not_plain_digits(self):
        frame = pd.DataFrame({"Incident Zip": ["ABCDE", "10001-1234", None]})
        cleaned = fix_types(frame)
        self.assertTrue(cleaned["Incident Zip"].isna().all())

    def test_drop_bad_rows_removes_duplicates_and_missing_coordinates(self):
        frame = pd.DataFrame(
            {
                "Borough": ["A", "A", "B", "C"],
                "Latitude": [1.0, 1.0, None, 3.0],
                "Longitude": [2.0, 2.0, 4.0, 5.0],
            }
        )
        cleaned = drop_bad_rows(frame)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(list(cleaned["Borough"]), ["A", "C"])

    def test_clean_dataframe_log_records_each_step(self):
        frame = pd.DataFrame(
            {
                "Borough": ["BROOKLYN", "BROOKLYN", "QUEENS"],
                "Created Date": ["2024-01-01 00:00:00", "2024-01-01 00:00:00", "2024-01-01 00:00:00"],
                "Closed Date": ["2024-01-01 02:00:00", "2024-01-01 02:00:00", "2024-01-01 04:00:00"],
                "Latitude": [1.0, 1.0, None],
                "Longitude": [2.0, 2.0, 3.0],
                "Almost Empty": [None, None, None],
            }
        )
        cleaned, log = clean_dataframe(frame)
        names = [step["name"] for step in log["steps"]]
        self.assertEqual(
            names,
            ["drop_sparse_columns", "standardize_text", "fix_types", "drop_bad_rows"],
        )
        self.assertIn("Almost Empty", log["steps"][0]["columns_dropped"])
        self.assertIn("resolution_hours", cleaned.columns)
        self.assertEqual(log["rows_before"], 3)
        self.assertEqual(log["rows_after"], len(cleaned))
        self.assertLess(log["rows_after"], log["rows_before"])
        self.assertIn("missing_counts_before", log["steps"][0])
        self.assertIn("missing_counts_after", log["steps"][0])

    def test_sample_file_cleans(self):
        frame = pd.read_csv(RAW_CSV, low_memory=False)
        self.assertGreater(len(frame), 1000)
        self.assertEqual(frame["Unique Key"].nunique(), len(frame))
        for column in ("Created Date", "Borough", "Latitude", "Longitude", "Incident Zip"):
            self.assertIn(column, frame.columns)
        cleaned, log = clean_dataframe(frame)
        dropped = log["steps"][0]["columns_dropped"]
        self.assertIn("Vehicle Type", dropped)
        self.assertNotIn("Borough", dropped)
        self.assertIn("resolution_hours", cleaned.columns)
        self.assertGreater(log["steps"][3]["detail"]["rows_missing_coordinates"], 0)
        self.assertEqual(log["rows_after"], len(cleaned))


if __name__ == "__main__":
    unittest.main()
