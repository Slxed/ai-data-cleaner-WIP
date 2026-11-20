"""File paths and LLM settings.

The defaults call Gemini through its OpenAI-compatible endpoint.
python-dotenv loads a .env file in this folder if you have one.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RAW_CSV = os.environ.get("RAW_CSV", str(ROOT / "sample" / "nyc311_sample.csv"))
CLEAN_DIR = ROOT / "clean data"
CLEAN_CSV = str(CLEAN_DIR / "nyc311_clean.csv")
CLEANING_LOG = str(CLEAN_DIR / "cleaning_log.json")
REPORT_PATH = str(CLEAN_DIR / "data_quality_report.md")

# Chat completions URL, without the /chat/completions suffix.
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Free-tier Flash model. Change LLM_MODEL in .env if you want a different one.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
# Which environment variable holds the key. The key itself is not stored here.
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "GEMINI_API_KEY")


def llm_api_key():
    return os.environ.get(LLM_API_KEY_ENV, "").strip()
