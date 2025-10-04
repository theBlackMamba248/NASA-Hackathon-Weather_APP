from openai import OpenAI
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Set it via environment or .env file!")
print(f"API_KEY: ...{API_KEY[-10:]}")



client = OpenAI(api_key=API_KEY)

def get_recommendation(row):
    activity_list = row["preferred_activities"]
    weather_forecast = row["weather"]

    system_prompt = """
You are an outdoor-activity suitability assistant. 
You DO NOT forecast — you interpret the provided weather summary.
Be concise, practical, and safety-aware.
Always return a compact JSON object as described below.
Never add text outside the JSON.

Default heuristics (used only if no custom thresholds provided):
- Parade/Picnic: avoid precipitation ≥2 mm/day; wind <9 m/s; comfort ~0–28 °C.
- Hiking/Walking: avoid storms or heavy rain ≥5 mm/day; ok temp −5…30 °C; wind <12 m/s.
- Camping: avoid heavy rain ≥6 mm/day; wind <10 m/s; temp 0…28 °C.
- Canoe/Kayak/SUP: prefer wind ≤8 m/s; avoid thunderstorms; temp 10…28 °C; rain <5 mm/day.
- Sailing/Windsurf/Kitesurf: good wind 5–14 m/s; caution with gusts >14 m/s or storms.
- Cycling: avoid heavy rain ≥5 mm/day and gusts >14 m/s; temp −2…32 °C.
- Running: avoid thunderstorms; temp −5…30 °C; humidex/heat index >32 °C ⇒ caution.
- Alpine/Cross-country Skiing/Snowshoeing: prefer snow depth ≥10 cm or snow cover ≥50%; ok temp ≤2 °C; wind chill ≤ −20 °C ⇒ caution/no-go.
- Photography/Stargazing: prefer cloud cover <40%; wind <10 m/s; no precipitation.
- Fishing: light wind <8 m/s; avoid thunderstorms; light rain is tolerable.

RATING rules:
- GO: all conditions within comfort thresholds
- CAUTION: minor or borderline issues
- NO-GO: significant safety/weather limitations
"""

    user_prompt = f'''
Evaluate the suitability of each preferred activity given the weather data.

Input JSON:
{{
  "preferred_activities": {json.dumps(activity_list, ensure_ascii=False)},
  "weather": {json.dumps(weather_forecast, ensure_ascii=False)}
}}

Output STRICTLY as JSON with these keys:
{{
  "rating": "GO|CAUTION|NO-GO", // only one rating for all activities
  "why": ["<=12 words", "<=12 words"],  // one for each activity
  "alternatives": ["<=3 words", "<=3 words"],
  "one_liner": "<=22 words summary in English"
}}
'''

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            answer = response.choices[0].message.content.strip()

            # attempt to extract JSON
            json_start = answer.find("{")
            json_end = answer.rfind("}") + 1
            answer_json = answer[json_start:json_end]
            result = json.loads(answer_json)
            return result

        except Exception as e:
            print(f"Error (attempt {attempt+1}): {e}")
            time.sleep(4 * (attempt + 1))
    return None


# Dummy weather function, replace with real API as needed
# get_weather_data.py
from __future__ import annotations
import datetime as dt
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

POWER_BASE = "https://power.larc.nasa.gov/api"

# --- Frontend -> POWER mapping (and helpers) ------------------------------


FRONTEND_TO_POWER: Dict[str, List[str]] = {
    "temperature": ["T2M", "T2M_MAX", "T2M_MIN"],        # °C
    "precipitation": ["PRECTOTCORR"],                    # mm/day
    "wind": ["WS10M"],                                   # m/s
    "humidity": ["RH2M"],                                # %
    "clouds": ["CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"],  # % or proxy
    "pressure": ["PS"],                                  # kPa
    "uvindex": ["ALLSKY_SFC_UV_INDEX"],                  # index
    "visibility": ["ALLSKY_KT"],
}

FRONTEND_LABELS = {
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "wind": "Wind Speed",
    "humidity": "Humidity",
    "clouds": "Cloud Cover",
    "visibility": "Visibility",
    "pressure": "Pressure",
    "uvindex": "UV Index",
}

UNITS = {
    "temperature": "°C (mean), °C (Tmax/Tmin)",
    "precipitation": "mm/day",
    "wind": "m/s",
    "humidity": "%",
    "clouds": "%",
    "visibility": "km (not available from POWER)",
    "pressure": "kPa",
    "uvindex": "index",
}

def _datestr(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def _fetch_power_daily_point(lat: float, lon: float, start: str, end: str, params: List[str], community="RE", timeout=45) -> pd.DataFrame:
    url = f"{POWER_BASE}/temporal/daily/point"
    r = requests.get(
        url,
        params={
            "parameters": ",".join(params),
            "community": community,
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    block = r.json().get("properties", {}).get("parameter", {})
    if not block:
        raise RuntimeError("POWER returned no data for the requested point/range.")
    frames = []
    for p in params:
        if p in block:
            ser = pd.Series(block[p], dtype="float64")
            ser.index = pd.to_datetime(ser.index, format="%Y%m%d", utc=True)
            frames.append(ser.rename(p))
    if not frames:
        raise RuntimeError("Requested parameters missing in POWER response.")
    return pd.concat(frames, axis=1).sort_index()

def _window_by_doy(df: pd.DataFrame, target_date: dt.date, window_days: int) -> pd.DataFrame:
    daily = df.resample("1D").mean()
    doy = daily.index.dayofyear
    center = target_date.timetuple().tm_yday
    mask = (
        (np.abs(doy - center) <= window_days)
        | (np.abs(doy + 366 - center) <= window_days)
        | (np.abs(doy - (center + 366)) <= window_days)
    )
    return daily.loc[mask]

def _mode1(s: pd.Series) -> Optional[float]:
    m = s.dropna().mode()
    return None if m.empty else float(m.iloc[0])

def _stats(s: pd.Series) -> Dict[str, Optional[float]]:
    s = s.dropna()
    if s.empty:
        return {"mean": None, "mode": None, "min": None, "max": None}
    return {"mean": float(s.mean()), "mode": _mode1(s), "min": float(s.min()), "max": float(s.max())}

def _derive_cloud_pct(win: pd.DataFrame) -> Optional[pd.Series]:
    """
    If CLOUD_AMT not present, approximate cloud fraction from
    (1 - ALLSKY/CLRSKY) * 100, clipped to [0,100].
    """
    if "CLOUD_AMT" in win:
        return win["CLOUD_AMT"]
    need = {"ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"}
    if not need.issubset(win.columns):
        return None
    num = win["ALLSKY_SFC_SW_DWN"].astype(float)
    den = win["CLRSKY_SFC_SW_DWN"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 1.0 - (num / den)
    frac = np.clip(frac, 0.0, 1.0)
    return pd.Series(frac * 100.0, index=win.index, name="CLOUD_PROXY")

def get_weather_data(
    *,
    lat: float,
    lon: float,
    date_iso: str,
    frontend_params: List[str],           # e.g., ["temperature","wind","humidity","uvindex"]
    years_back: int = 20,
    window_days: int = 7,
    community: str = "RE",
    include_percentiles: bool = False,
) -> Dict:
    """
    Pulls ~years_back of POWER daily data, filters to ±window_days around the same DOY,
    and returns mean/mode/min/max (plus optional percentiles) for the requested frontend params.

    Returns JSON ready for the LLM: put under row["weather"].
    """
    # 1) Parse date and build range
    try:
        target_date = dt.date.fromisoformat(date_iso)
    except Exception as e:
        raise ValueError("date_iso must be 'YYYY-MM-DD'") from e

    today = dt.date.today()
    start_date = max(dt.date(1981, 1, 1), today - relativedelta(years=years_back))
    if start_date > today:
        start_date = today

    # 2) Build POWER parameter list (include support vars when needed)
    power_params: List[str] = []
    for key in frontend_params:
        power_params += FRONTEND_TO_POWER.get(key, [])
    # Remove duplicates while preserving order
    seen = set()
    power_params = [p for p in power_params if not (p in seen or seen.add(p))]

    # If frontend asked only for visibility, POWER will have nothing; guard by fetching at least T2M to keep call valid
    if not power_params:
        power_params = ["T2M"]

    # 3) Fetch POWER daily data
    df = _fetch_power_daily_point(
        lat=lat,
        lon=lon,
        start=_datestr(start_date),
        end=_datestr(today),
        params=power_params,
        community=community,
    )

    # 4) Filter to ± DOY window
    win = _window_by_doy(df, target_date, window_days)
    if win.empty:
        raise RuntimeError("No historical rows in the ±DOY window for this point.")

    # 5) Compute per-frontend-param stats
    results: Dict[str, Dict] = {}
    percentiles: Dict[str, Dict[str, float]] = {}

    for key in frontend_params:
        label = FRONTEND_LABELS.get(key, key)
        unit = UNITS.get(key, "")

        if key == "temperature":
            # Provide stats for T2M (mean) + nested Tmax/Tmin
            base = _stats(win["T2M"]) if "T2M" in win else {"mean": None, "mode": None, "min": None, "max": None}
            sub = {
                "tmax": _stats(win["T2M_MAX"]) if "T2M_MAX" in win else None,
                "tmin": _stats(win["T2M_MIN"]) if "T2M_MIN" in win else None,
            }
            results[key] = {"label": label, "unit": unit, "stats": base, "components": sub}

            if include_percentiles:
                if "T2M" in win:
                    s = win["T2M"].dropna()
                    if len(s):
                        percentiles[key] = {"p25": float(np.percentile(s, 25)),
                                            "p50": float(np.percentile(s, 50)),
                                            "p75": float(np.percentile(s, 75)),
                                            "p90": float(np.percentile(s, 90))}

        elif key == "clouds":
            cloud_series = _derive_cloud_pct(win)
            if cloud_series is None:
                results[key] = {"label": label, "unit": unit, "stats": None, "available": False,
                                "note": "Cloud cover not directly available; proxy needs ALLSKY & CLRSKY."}
            else:
                results[key] = {"label": label, "unit": unit, "stats": _stats(cloud_series), "available": True}

        elif key == "visibility":
            # Not available from POWER
            results[key] = {"label": label, "unit": unit, "stats": None, "available": False,
                            "note": "Visibility is not provided by NASA POWER."}

        else:
            # Simple one-variable params
            # Map front-end key to the primary POWER variable to summarize
            main_var = {
                "precipitation": "PRECTOTCORR",
                "wind": "WS10M",
                "humidity": "RH2M",
                "pressure": "PS",
                "uvindex": "ALLSKY_SFC_UV_INDEX",
            }.get(key, None)

            if main_var and main_var in win:
                results[key] = {"label": label, "unit": unit, "stats": _stats(win[main_var])}
                if include_percentiles:
                    s = win[main_var].dropna()
                    if len(s):
                        percentiles[key] = {"p25": float(np.percentile(s, 25)),
                                            "p50": float(np.percentile(s, 50)),
                                            "p75": float(np.percentile(s, 75)),
                                            "p90": float(np.percentile(s, 90))}
            else:
                results[key] = {"label": label, "unit": unit, "stats": None, "available": False}

    # 6) Compose the JSON payload for your LLM
    weather_json = {
        "source": "NASA POWER (daily)",
        "location": {"lat": lat, "lon": lon},
        "target_date": target_date.isoformat(),
        "history_window": {
            "years_back": years_back,
            "start": _datestr(start_date),
            "end": _datestr(today),
            "doy_window_days": window_days
        },
        "parameters_requested": frontend_params,
        "results": results,                # <-- mean/mode/min/max per requested param (temperature includes tmax/tmin components)
        "percentiles": percentiles if include_percentiles else None,
        "note": "Stats computed over ± day-of-year window across years. Historical info, not a forecast."
    }
    return weather_json

    
    
