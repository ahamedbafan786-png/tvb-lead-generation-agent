"""
core/config.py
Production configuration and expanded search matrix for TVB lead generation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS = 15
TAVILY_MAX_CALLS_PER_RUN = int(os.getenv("TAVILY_MAX_CALLS_PER_RUN", "60"))

DEFAULT_MAX_SEARCH_QUERIES = 35
MAX_SEARCH_QUERIES_CEILING = 50
DEFAULT_TIMEOUT_SECONDS = 720
SCRAPER_TIMEOUT_SECONDS = 8
MAX_CANDIDATES_PER_RUN = int(os.getenv("MAX_CANDIDATES_PER_RUN", "150"))

MIN_FINANCIAL_USD = 1_000_000
MAX_FINANCIAL_USD = 5_000_000

NON_US_REGIONS = [
    "UK", "London", "Germany", "Munich", "Berlin", "France", "Paris",
    "Sweden", "Stockholm", "Norway", "Oslo", "Switzerland", "Zurich",
    "Netherlands", "Amsterdam", "Estonia", "Tallinn", "Czechia", "Prague",
    "Singapore", "Australia", "Sydney", "Canada", "Montreal", "Quebec",
    "India", "Bengaluru", "Brazil", "São Paulo", "Poland", "Warsaw"
]

TECH_SECTORS = [
    "B2B SaaS", "DevTools platform", "API infrastructure", "logistics software",
    "cybersecurity platform", "DevSecOps tools", "compliance automation",
    "AI workflow software", "legaltech software", "data engineering platform",
    "usage-based billing SaaS", "developer localization tools"
]

REVENUE_QUERY_TEMPLATES = [
    '{sector} startup {region} ARR 1M to 5M revenue -fund -consulting -report',
    '{sector} software startup {region} 1M to 5M annual recurring revenue -agency -recruiter',
    'bootstrapped {sector} startup {region} ARR revenue -consulting -directory',
    'profitable {sector} platform {region} revenue 1M to 5M -agency -report',
    '{sector} software startup {region} reached 1M to 4M ARR -fund -report'
]

FUNDING_QUERY_TEMPLATES = [
    '{sector} startup {region} "seed round" OR "seed funding" -fund -vc -consulting',
    '{sector} software startup {region} secures 1M to 5M seed funding -agency -recruiter',
    '{sector} platform {region} closes 1M to 5M seed funding round -fund -report',
    '{sector} startup {region} announces seed funding 2025 2026 -fund -directory',
    '{sector} tech startup {region} raised seed or Series A -fund -venture'
]

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
