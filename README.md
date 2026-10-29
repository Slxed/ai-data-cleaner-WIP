# AI Data Cleaner (WIP)

A Python pipeline that cleans a particular NY dataset and, in the next step, uses an LLM to write a plain-English data quality report which can also be queried for different questions relating to the cleaned data.

## Why I built this

I wanted more experience building data cleaning pipelines, similar to the work I was doing before at my past internship. I wanted to replicate that process with a bit of AI and then feed the result into an LLM for a cleaning report. I used AI coding tools to help write parts of the code.

## Data
NYC 311 Service Requests (NYC Open Data), about 316,000 rows and 44 columns. The raw file isn't in this repo because of its size.

## How to run
1. Clone the repo and create a virtual environment (I Used Python 3.13):
   ```
   git clone https://github.com/Slxed/ai-data-cleaner-WIP.git
   cd ai-data-cleaner-WIP
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Download the NYC 311 Service Requests CSV from NYC Open Data (https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data) and save it in a folder named `raw data/`. 
3. Update the `RAW` path at the top of `clean.py` to match your file name.
4. Create a folder named `clean data/` for the output.
5. Run the scripts:
   ```
   python explore.py
   python clean.py
   ```
   The cleaned file is saved to `clean data/nyc311_clean.csv`.

For Mac or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

## What it does
1. `explore.py` shows the shape, column types, missing values, and duplicates.
2. `clean.py` runs four steps and logs what each one removes:
   - drops columns that are 95% or more empty
   - standardizes text and turns 'No Data' placeholders like "N/A" and "Unspecified" into missing values
   - converts dates, fixes ZIP codes, and adds a `resolution_hours` column
   - drops duplicates and rows without coordinates

Put the raw CSV in `raw data/` first and update the path in `clean.py`.

## Tools
Python, pandas, NumPy, Matplotlib

## Next steps
- AI layer: send the cleaning log to an LLM for a data quality report
- Plain-English questions turned into queries via LLM.
