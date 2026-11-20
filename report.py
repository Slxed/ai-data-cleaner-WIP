"""Write a plain-English data quality report from the cleaning log.

Gemini drafts the report. Before that draft is saved, numbers with 3 or more
digits are checked against the log and the summary. If one does not match,
the draft is thrown out and the script-written summary is saved instead.
"""

import json
import re
import sys

import pandas as pd

from config import CLEAN_CSV, CLEANING_LOG, LLM_API_KEY_ENV, LLM_MODEL, REPORT_PATH, llm_api_key
from llm import LLMError, chat, message_text

BIG_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d{3,}")


def big_numbers(text):
    return [match.replace(",", "") for match in BIG_NUMBER.findall(text)]


def numbers_not_in_source(report_text, source_text):
    """Numbers of 3 or more digits that appear in the report but not in the source."""
    allowed = set(big_numbers(source_text))
    extra = []
    for number in big_numbers(report_text):
        if number not in allowed and number not in extra:
            extra.append(number)
    return extra


def find_column(columns, needle):
    needle = needle.lower()
    for col in columns:
        if needle in str(col).lower():
            return str(col)
    return None


def top_counts(series, limit=10):
    counts = series.fillna("(missing)").astype(str).value_counts().head(limit)
    return {str(label): int(count) for label, count in counts.items()}


def round_or_none(value):
    if pd.isna(value):
        return None
    # Two decimals keeps a tiny negative hour from looking like zero.
    number = round(float(value), 2)
    if number == 0:
        return 0.0
    return number


def summarize(df):
    """A small JSON summary. Raw rows are not sent to the model."""
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(col) for col in df.columns],
    }
    if "Borough" in df.columns:
        summary["borough_counts"] = top_counts(df["Borough"])
    complaint = find_column(df.columns, "complaint")
    if complaint:
        summary["complaint_column"] = complaint
        summary["top_complaint_types"] = top_counts(df[complaint], limit=8)
    if "Agency" in df.columns:
        summary["top_agencies"] = top_counts(df["Agency"], limit=8)
    if "resolution_hours" in df.columns:
        hours = pd.to_numeric(df["resolution_hours"], errors="coerce").dropna()
        summary["resolution_hours"] = {
            "non_null": int(hours.shape[0]),
            "missing": int(df["resolution_hours"].isna().sum()),
            "negative": int((hours < 0).sum()),
            "min": round_or_none(hours.min()) if len(hours) else None,
            "median": round_or_none(hours.median()) if len(hours) else None,
            "max": round_or_none(hours.max()) if len(hours) else None,
        }
    if "Created Date" in df.columns:
        created = pd.to_datetime(df["Created Date"], errors="coerce")
        if created.notna().any():
            summary["created_date_min"] = str(created.min())
            summary["created_date_max"] = str(created.max())
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(12)
    summary["highest_missing_shares"] = {
        str(col): round(float(share), 3) for col, share in missing.items()
    }
    return summary


def format_counts(counts):
    return ", ".join(f"{label} {count}" for label, count in counts.items())


def fallback_report(log, summary, reason):
    lines = [
        "# Data quality report",
        "",
        reason,
        "",
        "## Size",
        (
            f"The file started with {log['rows_before']} rows and {log['columns_before']} columns. "
            f"After cleaning it has {log['rows_after']} rows and {log['columns_after']} columns."
        ),
        "",
        "## What each step did",
    ]
    for step in log["steps"]:
        lines.append(
            f"- {step['name']}: removed {step['rows_removed']} rows, "
            f"columns {step['columns_before']} to {step['columns_after']}."
        )
        if step["columns_dropped"]:
            lines.append("  Dropped columns: " + ", ".join(step["columns_dropped"]) + ".")
        if step["columns_added"]:
            lines.append("  Added columns: " + ", ".join(step["columns_added"]) + ".")
        detail = step.get("detail") or {}
        if "duplicate_rows" in detail:
            lines.append(f"  Duplicate rows before this step: {detail['duplicate_rows']}.")
        if "rows_missing_coordinates" in detail:
            lines.append(
                "  Rows missing latitude or longitude before this step: "
                f"{detail['rows_missing_coordinates']}."
            )
        before = detail.get("examples_before") or {}
        after = detail.get("examples_after") or {}
        for col in before:
            lines.append(f"  {col} examples before: {before[col]}.")
            lines.append(f"  {col} examples after: {after.get(col)}.")
    lines.append("")
    lines.append("## Snapshot of the cleaned file")
    if "borough_counts" in summary:
        lines.append("Borough counts: " + format_counts(summary["borough_counts"]) + ".")
    if "top_complaint_types" in summary:
        lines.append(
            f"Top values in {summary['complaint_column']}: "
            + format_counts(summary["top_complaint_types"])
            + "."
        )
    if "top_agencies" in summary:
        lines.append("Top agencies: " + format_counts(summary["top_agencies"]) + ".")
    hours = summary.get("resolution_hours")
    if hours:
        lines.append(
            f"resolution_hours: {hours['non_null']} non-null, {hours['missing']} missing, "
            f"min {hours['min']}, median {hours['median']}, max {hours['max']}."
        )
        if hours.get("negative"):
            lines.append(
                f"{hours['negative']} of those values are negative, so the closed time is before the created time."
            )
    if "created_date_min" in summary:
        lines.append(
            f"Created Date ranges from {summary['created_date_min']} to {summary['created_date_max']}."
        )
    lines.extend(
        [
            "",
            "## Be careful",
            "- Rows without coordinates were dropped, so this file is only complaints that had a map point.",
            "- resolution_hours is missing when the closed date was missing or could not be parsed.",
            "- Columns that were 95% or more empty were dropped. The word Unspecified is turned into missing in the text step, which runs after that, so a column full of Unspecified can stay in the file.",
            "- City, Borough, and Agency are title-cased. That makes BROOKLYN into Brooklyn, and it also makes NYPD into Nypd. The examples in the log show the before and after.",
            "",
        ]
    )
    return "\n".join(lines)


def strip_fence(text):
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def report_messages(payload):
    system = (
        "You write a short data quality report for a student who cleaned NYC 311 complaint data. "
        "Use only the JSON in the user message. Do not invent counts, columns, or dates. "
        "When you mention a count, copy the digits exactly from the JSON. Do not round. "
        "Write plain English markdown with these sections: "
        "What this file is, What cleaning changed, What is still messy, What to be careful about. "
        "Do not wrap the report in a code fence. Keep it under 400 words."
    )
    user = json.dumps(payload, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def number_check_note():
    return (
        "\n\n## Number check\n\n"
        "Every number with 3 or more digits in this report also appears in the cleaning log "
        "or the summary that was sent to the model.\n"
    )


def write_report(text):
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def load_inputs():
    try:
        with open(CLEANING_LOG, encoding="utf-8") as handle:
            log = json.load(handle)
    except FileNotFoundError:
        print(f"Could not find {CLEANING_LOG}")
        print("Run python clean.py first.")
        sys.exit(1)
    try:
        df = pd.read_csv(CLEAN_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"Could not find {CLEAN_CSV}")
        print("Run python clean.py first.")
        sys.exit(1)
    return log, df


def main():
    log, df = load_inputs()
    summary = summarize(df)
    payload = {"cleaning_log": log, "cleaned_summary": summary}
    source_text = json.dumps(payload)

    if not llm_api_key():
        print(f"No API key in {LLM_API_KEY_ENV}. Writing a report from the log instead of calling Gemini.")
        text = fallback_report(
            log,
            summary,
            "No API key was set, so this report was written by the script from the cleaning log. It is not a Gemini writeup.",
        )
        write_report(text)
        print(text)
        print(f"saved {REPORT_PATH}")
        return

    print(f"Asking {LLM_MODEL} for a data quality report.")
    try:
        message = chat(report_messages(payload))
    except LLMError as err:
        print(f"The API call failed: {err}")
        print("Writing a report from the log instead.")
        text = fallback_report(
            log,
            summary,
            "The Gemini request failed, so this report was written by the script from the cleaning log. It is not a Gemini writeup.",
        )
        write_report(text)
        print(f"saved {REPORT_PATH}")
        return

    draft = strip_fence(message_text(message))
    if not draft:
        print("Guardrail: the model returned an empty report. Nothing from the model was saved.")
        text = fallback_report(
            log,
            summary,
            "Gemini returned an empty report, so this file was written by the script from the cleaning log.",
        )
        write_report(text)
        print(f"saved {REPORT_PATH}")
        return

    unknown = numbers_not_in_source(draft, source_text)
    if unknown:
        shown = ", ".join(unknown)
        print("Guardrail: rejected the model report. These numbers are not in the log or the summary:")
        print(shown)
        print("The model draft was not saved.")
        text = fallback_report(
            log,
            summary,
            "The number check rejected the Gemini draft, so this file was written by the script from the cleaning log. "
            f"The draft was not saved. Unknown numbers: {shown}.",
        )
        write_report(text)
        print(f"saved {REPORT_PATH}")
        return

    text = draft + number_check_note()
    write_report(text)
    print(text)
    print(f"saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
