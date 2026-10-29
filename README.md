# ai-data-cleaner-WIP
Work in progress*** ai-data-cleaner


# AI Data Cleaner

A Python pipeline that cleans messy public data and, in the next step, uses an LLM to write a plain-English data quality report.

## Why I built this

I wanted more experience building data cleaning pipelines, similar to the work I was doing before at my past internship. I wanted to replicate that process with a bit of AI and then feed the result into an LLM for a cleaning report. Note that I did use an LLM to generate code in this project.

## Data
NYC 311 Service Requests (NYC Open Data), about 316,000 rows and 44 columns. The raw file isn't in this repo because of its size.

## What it does
1. `explore.py` shows the shape, column types, missing values, and duplicates.
2. `clean.py` runs four steps and logs what each one removes:
   - drops columns that are 95% or more empty
   - standardizes text and turns placeholders like "N/A" and "Unspecified" into missing values
   - converts dates, fixes ZIP codes, and adds a `resolution_hours` column
   - drops duplicates and rows with no coordinates

Put the raw CSV in `raw data/` first and update the path in `clean.py`.

## Tools
Python, pandas, NumPy, Matplotlib

## Next steps
- AI layer: send the cleaning log to an LLM for a data quality report
- Plain-English questions turned into read-only queries, with checks
