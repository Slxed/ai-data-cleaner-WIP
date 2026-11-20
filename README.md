# AI Data Cleaner (WIP)

(new, please review) A Python pipeline that cleans the NYC 311 Service Requests dataset, asks an LLM for a plain-English data quality report, and answers plain-English questions about the cleaned data.

(new, please review) Anything in this README that starts with "(new, please review)" is draft text. I still need to rewrite those parts in my own words.

## Why I built this

I wanted more experience building data cleaning pipelines, similar to the work I was doing before at my past internship. I wanted to replicate that process with a bit of AI and then feed the result into an LLM for a cleaning report. I used AI coding tools to help write parts of the code.

## Data

NYC 311 Service Requests (NYC Open Data), about 316,000 rows and 44 columns. The raw file isn't in this repo because of its size.

(new, please review) For a short demo I committed a real 3,000 row sample at `sample/nyc311_sample.csv`. I pulled it from the NYC Open Data API (dataset `erm2-nwe9`) on November 20, 2026. The values are the API's values. `sample/SOURCE.txt` has the exact request. Created dates in the sample run from the evening of November 18, 2026 into the early morning of November 19.

## How to run

1. Clone the repo and create a virtual environment (I Used Python 3.13):
   ```
   git clone https://github.com/Slxed/ai-data-cleaner-WIP.git
   cd ai-data-cleaner-WIP
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. (new, please review) The scripts default to the sample file, so I can skip the full download when I am demoing. `clean.py` creates the `clean data` folder itself.
3. (new, please review) To clean the full file instead, download the NYC 311 Service Requests CSV from NYC Open Data (https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data), save it under `raw data/`, and run:
   ```
   python clean.py "raw data\your-file.csv"
   ```
   The default path lives in `config.py` as `RAW_CSV`. A path on the command line overrides it.
4. Run the scripts from the repo folder:
   ```
   python explore.py
   python clean.py
   python report.py
   python ask.py "which borough has the most complaints?"
   ```
   (new, please review) `clean.py` writes `clean data/nyc311_clean.csv` and `clean data/cleaning_log.json`. `report.py` writes `clean data/data_quality_report.md`.

For Mac or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

## Gemini key

(new, please review) `report.py` and `ask.py` call Google Gemini with a free key from Google AI Studio. Gemini writes the report and picks a table operation. pandas computes the numbers.

(new, please review) How I set the key in Windows PowerShell:

1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click Create API key and copy the key.
3. From this folder:
   ```
   copy .env.example .env
   notepad .env
   ```
4. Paste the key after `GEMINI_API_KEY=` and save. No quotes. `.env.example` shows the shape of the file. `.env` is listed in `.gitignore`, so the key stays on my machine.

(new, please review) The default model is `gemini-3.8-flash` (a free-tier Flash model). The default base URL is Gemini's OpenAI-compatible endpoint, `https://generativelanguage.googleapis.com/v1beta/openai/`. In `.env` I can set `LLM_MODEL`, `LLM_BASE_URL`, or `LLM_API_KEY_ENV` if I want a different model, a different endpoint, or a key stored under another variable name.

(new, please review) If `.env` has no key, the scripts still finish. `report.py` writes a short report straight from the cleaning log. `ask.py` uses a small keyword fallback for a few questions, including which borough has the most complaints. If the key is set but the request fails, they do the same thing and print the error.

## What it does

1. `explore.py` shows the shape, column types, missing values, and duplicates.
2. `clean.py` runs four steps and logs what each one removes:
   - drops columns that are 95% or more empty
   - standardizes text and turns 'No Data' placeholders like "N/A" and "Unspecified" into missing values
   - converts dates, fixes ZIP codes, and adds a `resolution_hours` column
   - drops duplicates and rows without coordinates
3. (new, please review) The same `clean.py` run writes `clean data/cleaning_log.json`. Each step has the row and column counts before and after, the columns dropped or added, missing-value counts, and a few extra notes (the 95% threshold, placeholder list, duplicate count, missing coordinates, and before/after text examples).
4. (new, please review) `report.py` sends that log plus a small summary of the cleaned file to Gemini. The summary is counts, not the raw rows. Gemini is asked for a markdown report. Before the draft is saved, the script pulls out every number with 3 or more digits. If one of those numbers is not in the log or the summary, the draft is not saved, and the file gets the script-written summary instead. A report that passes the check gets a short "Number check" note at the bottom.
5. (new, please review) `ask.py` sends the question, the real column names, and one tool, `run_table_op`, to Gemini. The tool arguments are a small operation: `count_by`, `mean_by`, `filter_count`, `column_summary`, or `cannot_answer`. The schema is only a hint. `validate_query` is the check that decides whether anything runs:
   - the operation has to be one of those five
   - column names have to match the cleaned file
   - the text cannot look like code (`eval`, `import`, `__`, and similar)
   - unexpected fields are rejected
   If the check fails, the script prints the reason and does not touch the table. If it passes, pandas runs that one operation. There is no `eval` of model text. A reply that is not a tool call is also rejected.
6. (new, please review) The keyword fallback in `ask.py` only handles a few phrases: which borough, agency, complaint type, or ZIP shows up most, and questions about resolution hours. Anything else needs Gemini.

## Tools

Python, pandas, NumPy, Matplotlib

(new, please review) The LLM calls use `urllib` from the standard library against a chat completions URL. `python-dotenv` loads `.env`. No LangChain.

## Tests

(new, please review) From the repo folder, with the venv on:

```
python -m unittest -v
```

(new, please review) `tests/test_clean.py` checks the four cleaning steps and the log. `tests/test_ask.py` checks the query validator, including an unknown column and a tool call that looks like code. `tests/test_report.py` checks that a made-up count fails the number check. The ask tests are an easy way to show the bad-output check on camera without waiting for the model to mess up.

## Next steps

(new, please review) The report and the plain-English questions are in `report.py` and `ask.py` now. Next I might run the same scripts on the full 311 download and add a couple of charts in `explore.py`.
