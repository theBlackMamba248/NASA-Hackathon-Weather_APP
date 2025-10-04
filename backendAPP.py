
#!pip install fastapi uvicorn pydantic openai python-dotenv pydantic-settings

#export OPENAI_API_KEY=sk-xxxx        # your key (llm_side.py reads API_KEY env)
#uvicorn app:app --port 8080 --reload

#app.py

from __future__ import annotations
import os, re, datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from functools import lru_cache

import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, validator

# ---------------- Settings ----------------
class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY",""))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS","*")  # "*" or "https://site1,https://site2"
    YEARS_BACK: int = 20                               # target history length
    DOY_WINDOW_DAYS: int = 7                           # +/- days around target DOY
    POWER_COMMUNITY: str = "RE"                        # POWER community (RE/AG/...)
    WHY_WORDS_MAX: int = 12
    ONE_LINER_WORDS_MAX: int = 22
    MAX_ALT: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache
def get_settings() -> Settings:
    return Settings()

# ------------ Minimal POWER client (climatology) ------------
POWER_BASE = "https://power.larc.nasa.gov/api"

DEFAULT_PARAMS = ["T2M_MAX","T2M_MIN","WS10M","PRECTOTCORR","RH2M","T2M"]

def _get_power_climatology_point(
    lat: float,
    lon: float,
    start: str,
    end: str,
    parameters: List[str],
    community: str = "RE",
) -> pd.DataFrame:
    """Return DOY-indexed climatology frame: rows=doy(1..366), cols=parameters."""
    url = f"{POWER_BASE}/temporal/climatology/point"
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start,   # e.g., "20050101"
        "end": end,       # e.g., "20251004"
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=40)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"POWER error: {r.status_code} {r.text[:200]}")
    data = r.json().get("properties", {}).get("parameter", {})
    if not data:
        raise HTTPException(status_code=502, detail="POWER: empty response")
    frames = []
    for p in parameters:
        if p in data:
            # keys are DOY strings "1".."366"
            ser = pd.Series(data[p], dtype="float64")
            ser.index = ser.index.astype(int)
            frames.append(ser.rename(p))
    if not frames:
        raise HTTPException(status_code=502, detail="POWER: requested parameters not present")
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "doy"
    return df

def _window_on_doy(df: pd.DataFrame, date: dt.date, window_days: int) -> pd.DataFrame:
    """Slice rows with DOY within +/- window_days; handles wrap-around."""
    center = date.timetuple().tm_yday
    idx = df.index.to_series()
    mask = (idx - center).abs() <= window_days
    # Wrap-around at year ends
    mask |= (idx + 366 - center).abs() <= window_days
    mask |= (idx - (center + 366)).abs() <= window_days
    return df.loc[mask]

def build_weather_summary(
    lat: float, lon: float, date: dt.date, years_back: int, window_days: int, community: str
) -> Dict[str, Any]:
    """
    Build the backend "weather" JSON that your LLM consumes.
    Uses POWER climatology over a bounded period to approximate past-N-years history,
    then averages min/max/wind/precip for the +/- DOY window.
    """
    # Period bounds (POWER needs absolute YYYYMMDD)
    end_abs = dt.date.today()
    start_abs = date - relativedelta(years=years_back)
    # Ensure start <= end and both are strings yyyymmdd
    if start_abs > end_abs:
        start_abs = end_abs - relativedelta(years=years_back)
    start_s = start_abs.strftime("%Y%m%d")
    end_s   = end_abs.strftime("%Y%m%d")

    df = _get_power_climatology_point(
        lat=lat, lon=lon, start=start_s, end=end_s,
        parameters=DEFAULT_PARAMS, community=community
    )
    win = _window_on_doy(df, date, window_days)
    if win.empty:
        raise HTTPException(status_code=500, detail="No climatology rows in DOY window")

    # Simple statistics (mean of the window climatology values)
    stats = {
        "T2M_MAX_mean": float(win["T2M_MAX"].mean()) if "T2M_MAX" in win else None,
        "T2M_MIN_mean": float(win["T2M_MIN"].mean()) if "T2M_MIN" in win else None,
        "WS10M_mean":   float(win["WS10M"].mean()) if "WS10M" in win else None,
        "PRECTOT_mean": float(win["PRECTOTCORR"].mean()) if "PRECTOTCORR" in win else None,
        "RH2M_mean":    float(win["RH2M"].mean()) if "RH2M" in win else None,
        "T2M_mean":     float(win["T2M"].mean()) if "T2M" in win else None,
    }

    # Optional dispersion (helps LLM without heavy math)
    disp = {}
    for col in ["T2M_MAX","T2M_MIN","WS10M","PRECTOTCORR"]:
        if col in win:
            s = win[col].dropna()
            if len(s):
                disp[col] = {
                    "p50": float(np.percentile(s, 50)),
                    "p75": float(np.percentile(s, 75)),
                    "p90": float(np.percentile(s, 90))
                }

    weather_json = {
        "source": "NASA POWER climatology",
        "location": {"lat": lat, "lon": lon},
        "target_date": date.isoformat(),
        "period": {"start": start_s, "end": end_s, "years_back": years_back},
        "doy_window_days": window_days,
        "stats": stats,
        "percentiles": disp,
        "note": "Historical averages around the same day-of-year. Not a forecast."
    }
    return weather_json

# --------------- LLM integration (your code) ---------------
# we import your module and make sure it sees the API key as API_KEY env var
import importlib

def init_llm_module(cfg: Settings):
    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing; export it or put in .env")
    os.environ["API_KEY"] = cfg.OPENAI_API_KEY  # llm_side.py expects this name
    mod = importlib.import_module("llm_side")
    if not hasattr(mod, "get_recommendation"):
        raise RuntimeError("llm_side.get_recommendation not found")
    return mod

LLM_MOD = init_llm_module(get_settings())

# --------------- API Schemas ----------------
class Location(BaseModel):
    lat: float
    lon: float

class SimpleQuery(BaseModel):
    location: Location
    date: dt.date
    activity: str

class RecommendResponse(BaseModel):
    rating: str
    why: List[str]
    alternatives: List[str]
    one_liner: str

# --------------- App & CORS -----------------
app = FastAPI(title="Lightweight Weather->LLM Backend", version="1.0.0")

cfg = get_settings()
origins = ["*"] if cfg.CORS_ORIGINS.strip()=="*" else [o.strip() for o in cfg.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Helpers --------------------
def _truncate_words(s: str, max_words: int) -> str:
    words = re.split(r"\s+", (s or "").strip())
    return " ".join(words[:max_words])

def normalize_output(raw: Dict[str, Any], n_activities: int, cfg: Settings) -> Dict[str, Any]:
    out = {
        "rating": raw.get("rating", "CAUTION"),
        "why": raw.get("why") or [],
        "alternatives": raw.get("alternatives") or [],
        "one_liner": raw.get("one_liner") or ""
    }
    if out["rating"] not in {"GO","CAUTION","NO-GO"}:
        out["rating"] = "CAUTION"

    why = out["why"]
    if not isinstance(why, list): why=[str(why)]
    if len(why) < n_activities: why += ["No detail."]*(n_activities-len(why))
    elif len(why) > n_activities: why = why[:n_activities]
    out["why"] = [_truncate_words(x, cfg.WHY_WORDS_MAX) for x in map(str, why)]

    alts = out["alternatives"]
    if not isinstance(alts, list): alts=[str(alts)]
    out["alternatives"] = [_truncate_words(x, 3) for x in alts][:cfg.MAX_ALT]

    out["one_liner"] = _truncate_words(str(out["one_liner"]), cfg.ONE_LINER_WORDS_MAX)
    return out

# --------------- Routes ---------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/summarize-weather")
def summarize_weather(q: SimpleQuery, settings: Settings = Depends(get_settings)):
    """
    Returns the historical-weather JSON that the LLM consumes.
    Frontend can call this if it wants to preview the stats.
    """
    wx = build_weather_summary(
        lat=q.location.lat,
        lon=q.location.lon,
        date=q.date,
        years_back=settings.YEARS_BACK,
        window_days=settings.DOY_WINDOW_DAYS,
        community=settings.POWER_COMMUNITY
    )
    return wx

@app.post("/recommend", response_model=RecommendResponse)
def recommend(q: SimpleQuery, settings: Settings = Depends(get_settings), debug: bool = Query(False)):
    """
    One-stop endpoint: FE sends {lat,lon,date,activity}.
    Backend:
      1) builds POWER-based historical summary for DOY window,
      2) calls your LLM get_recommendation(row),
      3) returns normalized JSON suitable for UI.
    """
    # 1) build weather JSON for the LLM (mean mins/maxes over DOY window)
    weather_json = build_weather_summary(
        lat=q.location.lat,
        lon=q.location.lon,
        date=q.date,
        years_back=settings.YEARS_BACK,
        window_days=settings.DOY_WINDOW_DAYS,
        community=settings.POWER_COMMUNITY
    )

    # 2) call your LLM function verbatim
    row = {"preferred_activities": [q.activity], "weather": weather_json}
    try:
        raw = LLM_MOD.get_recommendation(row)
        if not raw:
            raise RuntimeError("LLM returned empty/None")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    # 3) normalize output to guarantee UI safety
    clean = normalize_output(raw, n_activities=1, cfg=settings)

    # Optional debug passthrough
    if debug:
        clean = {**clean, "debug_weather": weather_json}

    return clean

