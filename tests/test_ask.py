import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask import (
    fallback_spec,
    looks_like_code,
    parse_tool_call,
    run_query,
    validate_query,
)
from clean import clean_dataframe
from config import RAW_CSV


class AskTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "Borough": ["Brooklyn", "Brooklyn", "Queens"],
                "Agency": ["Nypd", "Hpd", "Nypd"],
                "resolution_hours": [2.0, 4.0, 6.0],
            }
        )

    def test_count_by_accepts_a_real_column_and_fixes_case(self):
        cleaned, errors = validate_query(
            {"op": "count_by", "column": "borough"},
            self.frame(),
        )
        self.assertEqual(errors, [])
        self.assertEqual(cleaned["column"], "Borough")
        self.assertEqual(cleaned["limit"], 10)
        text = run_query(self.frame(), cleaned)
        self.assertIn("Brooklyn has the most rows (2).", text)

    def test_unknown_column_is_rejected(self):
        cleaned, errors = validate_query(
            {"op": "count_by", "column": "Favorite Color"},
            self.frame(),
        )
        self.assertIsNone(cleaned)
        self.assertTrue(any("Unknown column" in err for err in errors))

    def test_disallowed_operation_is_rejected(self):
        cleaned, errors = validate_query(
            {"op": "eval", "column": "Borough"},
            self.frame(),
        )
        self.assertIsNone(cleaned)
        self.assertTrue(any("not allowed" in err for err in errors))

    def test_code_like_text_is_rejected(self):
        self.assertFalse(looks_like_code("Important"))
        self.assertTrue(looks_like_code("import os"))
        cleaned, errors = validate_query(
            {"op": "count_by", "column": "__import__('os')"},
            self.frame(),
        )
        self.assertIsNone(cleaned)
        self.assertTrue(any("looks like code" in err for err in errors))

    def test_unexpected_field_is_rejected(self):
        cleaned, errors = validate_query(
            {"op": "count_by", "column": "Borough", "code": "print(1)"},
            self.frame(),
        )
        self.assertIsNone(cleaned)
        self.assertTrue(any("Unexpected fields" in err for err in errors))

    def test_mean_of_a_text_column_is_rejected(self):
        cleaned, errors = validate_query(
            {"op": "mean_by", "column": "Borough", "by": "Agency"},
            self.frame(),
        )
        self.assertIsNone(cleaned)
        self.assertTrue(any("not numeric" in err for err in errors))

    def test_text_reply_is_not_a_tool_call(self):
        spec, error = parse_tool_call(
            {"role": "assistant", "content": "import os\nos.system('echo hi')"}
        )
        self.assertIsNone(spec)
        self.assertIn("did not return a tool call", error)

    def test_bad_json_arguments_are_rejected(self):
        spec, error = parse_tool_call(
            {
                "tool_calls": [
                    {"function": {"name": "run_table_op", "arguments": "Borough.head()"}}
                ]
            }
        )
        self.assertIsNone(spec)
        self.assertIn("not valid JSON", error)

    def test_valid_tool_call_parses(self):
        spec, error = parse_tool_call(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_table_op",
                            "arguments": json.dumps(
                                {"op": "count_by", "column": "Borough", "limit": 5}
                            ),
                        }
                    }
                ]
            }
        )
        self.assertIsNone(error)
        self.assertEqual(spec["op"], "count_by")

    def test_keyword_fallback_for_the_demo_question(self):
        spec = fallback_spec("which borough has the most complaints?", self.frame().columns)
        self.assertEqual(spec["op"], "count_by")
        self.assertEqual(spec["column"], "Borough")

    def test_sample_borough_question(self):
        raw = pd.read_csv(RAW_CSV, low_memory=False)
        cleaned_frame, _log = clean_dataframe(raw)
        spec = fallback_spec("which borough has the most complaints?", cleaned_frame.columns)
        checked, errors = validate_query(spec, cleaned_frame)
        self.assertEqual(errors, [])
        text = run_query(cleaned_frame, checked)
        self.assertTrue(text.startswith("Brooklyn:"))
        self.assertIn("Brooklyn has the most rows", text)


if __name__ == "__main__":
    unittest.main()
