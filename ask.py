"""Ask a plain-English question about the cleaned 311 file.

The model may only call one tool, run_table_op. validate_query checks that
call before pandas runs it. Free-written code is never executed.
"""

import json
import sys

import pandas as pd

from config import CLEAN_CSV, LLM_API_KEY_ENV, LLM_MODEL, llm_api_key
from llm import LLMError, chat

TOOL_NAME = "run_table_op"
ALLOWED_OPS = ("count_by", "mean_by", "filter_count", "column_summary", "cannot_answer")
ALLOWED_KEYS = {"op", "column", "by", "value", "limit"}
CODE_WORDS = {"eval", "exec", "import", "lambda", "pandas"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Run one safe aggregation on the cleaned 311 table. Do not compute the numbers yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": list(ALLOWED_OPS),
                        "description": "count_by, mean_by, filter_count, column_summary, or cannot_answer.",
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to count, average, filter, or summarize. Use an exact column name.",
                    },
                    "by": {
                        "type": "string",
                        "description": "Group-by column, only for mean_by. Use an exact column name.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to match, only for filter_count. Send it as text.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many groups to return, from 1 to 20.",
                    },
                },
                "required": ["op"],
            },
        },
    }
]


def looks_like_code(text):
    """True when a string looks like code rather than a column name or a value."""
    lowered = str(text).lower()
    if any(piece in lowered for piece in ("__", ";", "`", "\n", "\r", "os.", "subprocess")):
        return True
    words = set("".join(ch if ch.isalnum() else " " for ch in lowered).split())
    return bool(words & CODE_WORDS)


def resolve_column(name, columns):
    if name in columns:
        return name
    lookup = {col.lower(): col for col in columns}
    return lookup.get(name.lower())


def column_containing(columns, needle):
    needle = needle.lower()
    for col in columns:
        if col.lower() == needle:
            return col
    for col in columns:
        if needle in col.lower():
            return col
    return None


def parse_limit(value):
    if value is None:
        return 10
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= 20:
        return value
    if isinstance(value, float) and value.is_integer() and 1 <= int(value) <= 20:
        return int(value)
    if isinstance(value, str) and value.isdigit() and 1 <= int(value) <= 20:
        return int(value)
    return None


def _blank(value):
    return value is None or value == ""


def validate_query(spec, df):
    """Return (cleaned spec, errors). An empty error list means it is safe to run.

    The schema we send the model is only a hint. This function is the check.
    """
    if not isinstance(spec, dict):
        return None, ["The tool call was not a JSON object."]

    errors = []
    extra = sorted(key for key in spec.keys() if key not in ALLOWED_KEYS)
    if extra:
        errors.append("Unexpected fields: " + ", ".join(extra) + ".")

    op = spec.get("op")
    if op not in ALLOWED_OPS:
        errors.append(
            f"Operation {op!r} is not allowed. Allowed operations: {', '.join(ALLOWED_OPS)}."
        )

    limit = parse_limit(spec.get("limit"))
    if limit is None:
        errors.append("limit must be a whole number from 1 to 20.")

    cleaned = {"op": op, "limit": limit if limit is not None else 10}
    columns = [str(col) for col in df.columns]

    for key in ("column", "by"):
        if _blank(spec.get(key)):
            continue
        raw = spec.get(key)
        if not isinstance(raw, str):
            errors.append(f"{key} must be a column name.")
            continue
        if looks_like_code(raw):
            errors.append(f"{key} looks like code, so I rejected it.")
            continue
        resolved = resolve_column(raw, columns)
        if resolved is None:
            errors.append(f"Unknown column {raw!r}.")
            continue
        cleaned[key] = resolved

    if not _blank(spec.get("value")):
        raw_value = spec.get("value")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int, float)):
            errors.append("value must be text or a number.")
        elif looks_like_code(raw_value):
            errors.append("value looks like code, so I rejected it.")
        else:
            cleaned["value"] = str(raw_value)

    if op == "count_by" and "column" not in cleaned:
        errors.append("count_by needs a column.")
    if op == "mean_by":
        if "column" not in cleaned or "by" not in cleaned:
            errors.append("mean_by needs column and by.")
        elif not column_is_numeric(df, cleaned["column"]):
            errors.append(f"{cleaned['column']} is not numeric, so I will not average it.")
    if op == "filter_count":
        if "column" not in cleaned or "value" not in cleaned:
            errors.append("filter_count needs a column and a value.")
    if op == "column_summary":
        if "column" not in cleaned:
            errors.append("column_summary needs a column.")
        elif not column_is_numeric(df, cleaned["column"]):
            errors.append(f"{cleaned['column']} is not numeric, so I will not summarize it.")

    if errors:
        return None, errors
    return cleaned, []


def column_is_numeric(df, name):
    series = df[name]
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().any())


def run_query(df, spec):
    """Run a spec that validate_query already accepted. No eval."""
    op = spec["op"]
    if op == "cannot_answer":
        return (
            "That question does not fit the operations I am allowed to run. "
            "I can count by a column, average a numeric column by another column, "
            "count rows that match a value, or summarize one numeric column."
        )
    if op == "count_by":
        counts = (
            df[spec["column"]]
            .fillna("(missing)")
            .astype(str)
            .value_counts()
            .head(spec["limit"])
        )
        if counts.empty:
            return f"No rows to count for {spec['column']}."
        lines = [f"{label}: {int(count)}" for label, count in counts.items()]
        top_label = counts.index[0]
        top_n = int(counts.iloc[0])
        lines.append("")
        lines.append(f"{top_label} has the most rows ({top_n}).")
        return "\n".join(lines)
    if op == "mean_by":
        frame = df[[spec["by"], spec["column"]]].copy()
        frame[spec["column"]] = pd.to_numeric(frame[spec["column"]], errors="coerce")
        grouped = frame.groupby(spec["by"], dropna=False)[spec["column"]].mean()
        grouped = grouped.sort_values(ascending=False).head(spec["limit"])
        if grouped.empty:
            return f"No rows to average for {spec['column']}."
        lines = []
        for label, value in grouped.items():
            shown = "(missing)" if pd.isna(label) else label
            number = "n/a" if pd.isna(value) else f"{float(value):.1f}"
            lines.append(f"{shown}: {number}")
        return "\n".join(lines)
    if op == "filter_count":
        series = df[spec["column"]]
        value = spec["value"]
        if pd.api.types.is_numeric_dtype(series):
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            matched = 0 if pd.isna(number) else int((series == number).sum())
        else:
            matched = int(
                series.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(str(value).strip().lower())
                .sum()
            )
        return f"{matched} rows where {spec['column']} is {value}."
    if op == "column_summary":
        numbers = pd.to_numeric(df[spec["column"]], errors="coerce").dropna()
        if numbers.empty:
            return f"No numeric values in {spec['column']}."
        return (
            f"{spec['column']}: count {int(numbers.shape[0])}, "
            f"min {float(numbers.min()):.1f}, median {float(numbers.median()):.1f}, "
            f"max {float(numbers.max()):.1f}."
        )
    return "That operation is not allowed."


def fallback_spec(question, columns):
    """A small keyword matcher used only when Gemini is not available."""
    q = " ".join(question.lower().split())
    if not q:
        return None
    columns = [str(col) for col in columns]
    borough = column_containing(columns, "borough")
    agency = column_containing(columns, "agency")
    complaint = column_containing(columns, "complaint")
    zip_col = column_containing(columns, "zip")
    hours = column_containing(columns, "hour")
    wants_count = any(word in q for word in ("most", "common", "how many", "count", "which"))

    if wants_count and "borough" in q and borough:
        return {"op": "count_by", "column": borough, "limit": 10}
    if wants_count and "agency" in q and agency:
        return {"op": "count_by", "column": agency, "limit": 10}
    if wants_count and ("complaint" in q or "problem" in q) and complaint:
        return {"op": "count_by", "column": complaint, "limit": 10}
    if wants_count and "zip" in q and zip_col:
        return {"op": "count_by", "column": zip_col, "limit": 10}
    if hours and any(word in q for word in ("hour", "how long", "resolution")):
        if borough and "borough" in q:
            return {"op": "mean_by", "column": hours, "by": borough, "limit": 10}
        return {"op": "column_summary", "column": hours, "limit": 10}
    return None


def parse_tool_call(message):
    """Pull run_table_op arguments out of an assistant message.

    Returns (spec, error). If error is set, spec is None and nothing should run.
    """
    if not isinstance(message, dict):
        return None, "The model reply was not a message object."
    calls = message.get("tool_calls") or []
    if not calls:
        return None, "The model did not return a tool call."
    function = (calls[0] or {}).get("function") or {}
    name = function.get("name")
    if name != TOOL_NAME:
        return None, f"The model called {name!r} instead of {TOOL_NAME}."
    raw = function.get("arguments")
    if isinstance(raw, str):
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError:
            return None, "The tool call arguments were not valid JSON."
    elif isinstance(raw, dict):
        spec = raw
    else:
        return None, "The tool call did not include arguments."
    if not isinstance(spec, dict):
        return None, "The tool call arguments were not a JSON object."
    return spec, None


def messages_for(question, columns):
    column_lines = "\n".join(f"- {col}" for col in columns)
    example = json.dumps({"op": "count_by", "column": "Borough", "limit": 10})
    system = (
        "You turn a question about a table into one tool call. "
        "Do not answer the question yourself and do not invent numbers. "
        "The program will run the tool and compute the numbers. "
        "Only use column names from the list. "
        "If nothing fits, call the tool with op cannot_answer."
    )
    user = (
        f"Question: {question}\n\n"
        f"Columns:\n{column_lines}\n\n"
        f"Call {TOOL_NAME} once. "
        f"Example for 'which borough has the most complaints?': {example}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def question_from_argv(argv):
    return " ".join(argv[1:]).strip()


def main():
    question = question_from_argv(sys.argv)
    if not question and sys.stdin.isatty():
        try:
            question = input("Question: ").strip()
        except EOFError:
            question = ""
    if not question:
        print('Type a question. Example: python ask.py "which borough has the most complaints?"')
        return

    try:
        df = pd.read_csv(CLEAN_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"Could not find {CLEAN_CSV}")
        print("Run python clean.py first.")
        sys.exit(1)

    print(f"Question: {question}")
    spec = None
    source = None

    if not llm_api_key():
        print(f"No API key in {LLM_API_KEY_ENV}. Using the keyword fallback instead of Gemini.")
        spec = fallback_spec(question, df.columns)
        source = "keyword fallback"
        if spec is None:
            print("This question did not match the keyword fallback.")
            print("The fallback can count by borough, agency, complaint type, or ZIP, or summarize resolution hours.")
            print("Add a key to .env (see .env.example) to ask Gemini.")
            return
    else:
        print(f"Asking {LLM_MODEL} for a {TOOL_NAME} tool call.")
        try:
            message = chat(messages_for(question, df.columns), tools=TOOLS, tool_name=TOOL_NAME)
        except LLMError as err:
            print(f"The API call failed: {err}")
            spec = fallback_spec(question, df.columns)
            source = "keyword fallback after the API error"
            if spec is None:
                print("The keyword fallback did not match this question either.")
                return
            print("Using the keyword fallback instead.")
        else:
            spec, error = parse_tool_call(message)
            source = "model tool call"
            if error:
                print(f"Guardrail: {error} Nothing was run.")
                return

    print()
    print(f"Proposed operation ({source}):")
    print(json.dumps(spec, indent=2))
    cleaned, errors = validate_query(spec, df)
    if errors:
        print("Validator rejected this operation, so nothing was run:")
        for err in errors:
            print(f"- {err}")
        return

    print("Validator: ok")
    print()
    print(run_query(df, cleaned))
    if cleaned["op"] != "cannot_answer":
        print()
        print("Those numbers were computed by pandas from the cleaned CSV. The model did not supply them.")


if __name__ == "__main__":
    main()
