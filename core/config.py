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
    '{sector} startup {region} "$1M" OR "$2M" "ARR" founder -fund -consulting -report',
    '{sector} software {region} "$2M" OR "$3M" "annual recurring revenue" CEO -agency -recruiter',
    '{sector} platform {region} "$1M" OR "$4M" "ARR" co-founder -fund -directory',
    '{sector} startup {region} "reached $1M ARR" OR "reached $2M ARR" founder -consulting -venture',
    '{sector} software startup {region} "$1.5M" OR "$3M" "ARR" CEO 2024 2025 -report -vc',
    '{sector} SaaS {region} "€1M" OR "€2M" "ARR" founder -fund -agency',
    '{sector} platform {region} "profitable" "$1M" OR "$2.5M" revenue CEO -agency -report',
    '{sector} software {region} "bootstrapped" "$1M" OR "$3M" "ARR" founder -consulting -directory',
    '{sector} startup {region} "£1M" OR "£2M" "ARR" founder -fund -venture',
    '{sector} platform {region} "$1M" OR "$5M" "annual recurring revenue" co-founder -report -recruiter'
]

FUNDING_QUERY_TEMPLATES = [
    '{sector} startup {region} "raised $1M" OR "raised $2M" seed founder -fund -vc',
    '{sector} software {region} "closes $2M" OR "closes $3M" seed round CEO -agency -report',
    '{sector} platform {region} "secures $1.5M" OR "secures $2.5M" seed co-founder -fund -consulting',
    '{sector} tech startup {region} "seed round" "$1M" OR "$2M" founder 2024 2025 -venture -directory',
    '{sector} software startup {region} "raised $3M" OR "raised $4M" seed CEO -agency -recruiter',
    '{sector} platform {region} "raised €1M" OR "raised €2M" seed founder -fund -report',
    '{sector} startup {region} "secures $3M" OR "secures $5M" seed round CEO -fund -vc',
    '{sector} software {region} "raised £1M" OR "raised £2M" seed co-founder -consulting -directory',
    '{sector} platform {region} "closes $1M" OR "closes $2.5M" seed round founder 2024 2025 -fund -recruiter',
    '{sector} tech startup {region} "pre-seed" OR "seed" "$1M" OR "$3M" CEO -agency -venture'
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
