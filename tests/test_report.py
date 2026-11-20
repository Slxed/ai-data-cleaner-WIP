import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report import fallback_report, numbers_not_in_source


class ReportTests(unittest.TestCase):
    def test_number_check_allows_source_numbers_only(self):
        source = '{"rows_before": 3000, "rows_after": 2963}'
        self.assertEqual(numbers_not_in_source("The file started with 3000 rows.", source), [])
        self.assertEqual(numbers_not_in_source("The file started with 3,000 rows.", source), [])
        self.assertEqual(numbers_not_in_source("Step 1 removed some rows.", source), [])
        self.assertEqual(numbers_not_in_source("The file started with 99999 rows.", source), ["99999"])

    def test_fallback_report_uses_the_log(self):
        log = {
            "rows_before": 10,
            "columns_before": 2,
            "rows_after": 8,
            "columns_after": 2,
            "steps": [
                {
                    "name": "drop_bad_rows",
                    "rows_removed": 2,
                    "columns_before": 2,
                    "columns_after": 2,
                    "columns_dropped": [],
                    "columns_added": [],
                    "detail": {"duplicate_rows": 0, "rows_missing_coordinates": 2},
                }
            ],
        }
        text = fallback_report(log, {"rows": 8}, "Test reason.")
        self.assertIn("Test reason.", text)
        self.assertIn("started with 10 rows", text)
        self.assertIn("8 rows", text)


if __name__ == "__main__":
    unittest.main()
