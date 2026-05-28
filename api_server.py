"""
Scout API — FastAPI endpoint for fvno.se hero widget.

POST /analyze  {"url": "https://..."}
→ {score, load_time, mobile, insights, company_name}

Run:
    uvicorn api_server:app --port 8503 --host 0.0.0.0
"""
from __future__ import annotations

import os
import re
import sys
import time
import traceback
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Scout Analyze API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fvno.se",
        "https://www.fvno.se",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Scout/2.0; +https://fvno.se)"}
_TIMEOUT = 15


# ── Scrape ────────────────────────────────────────────────────────────────────
async def _scrape(url: str) -> Optional[dict]:
    """Fetch the target URL once and extract all metrics needed for analysis."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, headers=_HEADERS) as client:
            t0 = time.monotonic()
            r = await client.get(url, timeout=_TIMEOUT)
            load_time = round(time.monotonic() - t0, 2)

        size_bytes = len(r.content)
        soup = BeautifulSoup(r.content, "html.parser")

        scripts = len(soup.find_all("script"))
        css_links = len(soup.find_all(
            "link", rel=lambda v: isinstance(v, list) and "stylesheet" in v
        ))
        has_viewport = bool(soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)}))
        has_h1 = bool(soup.find("h1"))
        imgs = soup.find_all("img")
        imgs_without_alt = sum(1 for img in imgs if not img.get("alt", "").strip())

        return {
            "load_time": load_time,
            "size_bytes": size_bytes,
            "scripts": scripts,
            "css_links": css_links,
            "has_viewport": has_viewport,
            "has_h1": has_h1,
            "total_imgs": len(imgs),
            "imgs_without_alt": imgs_without_alt,
            "company_name": _extract_company_name(soup, url),
            "status_code": r.status_code,
        }
    except Exception as exc:
        print(
            f"[API] scrape failed {repr(exc)}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None


def _extract_company_name(soup: BeautifulSoup, url: str) -> str:
    og_site = soup.find("meta", property="og:site_name")
    if og_site and og_site.get("content"):
        return str(og_site["content"]).strip()[:60]

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        name = str(og_title["content"]).strip()
        parts = re.split(r"\s*[\|–—-]\s*", name)
        return (parts[-1] if len(parts) > 1 else parts[0])[:60]

    if soup.title and soup.title.string:
        t = soup.title.string.strip()
        parts = re.split(r"\s*[\|–—-]\s*", t)
        return (parts[-1] if len(parts) > 1 else parts[0])[:60]

    host = urlparse(url).netloc.removeprefix("www.")
    return host.split(".")[0].title() or "din sajt"


# ── Analysis ──────────────────────────────────────────────────────────────────
def _analyze(m: dict) -> tuple[int, float, bool, List[str]]:
    """
    Derive score (0–100), load_time, mobile_ok, and up to 3 insights
    from scraped metrics.
    """
    load_time   = m["load_time"]
    size_mb     = m["size_bytes"] / 1_000_000
    scripts     = m["scripts"]
    css_links   = m["css_links"]
    has_viewport = m["has_viewport"]
    has_h1      = m["has_h1"]
    total_imgs  = m["total_imgs"]
    imgs_no_alt = m["imgs_without_alt"]

    # ── Score ─────────────────────────────────────────────────────────────────
    score = 100

    if load_time > 5:     score -= 25
    elif load_time > 3:   score -= 15
    elif load_time > 2:   score -= 8

    if size_mb > 3:       score -= 20
    elif size_mb > 1.5:   score -= 10
    elif size_mb > 0.8:   score -= 5

    if not has_viewport:  score -= 20
    if not has_h1:        score -= 10

    if scripts > 15:      score -= 15
    elif scripts > 8:     score -= 7

    if css_links > 6:     score -= 5

    score = max(10, score)

    # ── Insights pool ─────────────────────────────────────────────────────────
    pool: List[tuple[int, str]] = []

    if not has_viewport:
        pool.append((10,
            "Sidan saknar mobilanpassning — vi hittade ingen viewport-inställning. "
            "Mobilbesökare ser en zoomad ut datorvy och Google straffar det hårt "
            "i mobilsökresultaten."
        ))

    lt = round(load_time, 1)
    if load_time > 5:
        pool.append((9,
            f"Sidan tog {lt} sekunder att ladda — mer än dubbelt Googles "
            f"rekommendation på 2,5s. Med den laddningstiden försvinner "
            f"uppskattningsvis varannan besökare innan sidan visas."
        ))
    elif load_time > 3:
        pool.append((7,
            f"Sidan laddade på {lt}s — märkbart över Googles rekommenderade 2,5s. "
            f"Det räcker för att tappa tveksamma besökare till nästa sökresultat."
        ))
    elif load_time > 2:
        pool.append((4,
            f"Laddningstiden är {lt}s — lite över optimalt. Bildoptimering och "
            f"färre externa skript kan ofta ta ner det under 2s."
        ))

    mb = round(size_mb, 1)
    if size_mb > 3:
        pool.append((8,
            f"Sidan laddar {mb} MB data — ungefär tre gånger mer än rekommenderat "
            f"för mobil. Det är oftast okomprimerade bilder eller inaktiva tillägg "
            f"som kan tas bort utan att sidan förändras."
        ))
    elif size_mb > 1.5:
        pool.append((5,
            f"Sidan väger {mb} MB — lite tungt för mobil. "
            f"Bildoptimering ensam kan ofta halvera sidvikten."
        ))

    if scripts > 15:
        pool.append((7,
            f"Vi hittade {scripts} skript-taggar på sidan — varje skript fördröjer "
            f"laddningen och interaktiviteten. Plugins och tredjepartsverktyg som "
            f"inte används aktivt är ofta boven."
        ))
    elif scripts > 8:
        pool.append((4,
            f"Sidan laddar {scripts} skript. Det är hanterbart men värt att granska — "
            f"oanvända plugins syns inte för besökaren men saktar ner sidan."
        ))

    if not has_h1:
        pool.append((6,
            "Sidan saknar en H1-rubrik — det är den tydligaste signalen till Google "
            "om vad sidan handlar om. Utan den tappar sidan sökrelevans för era "
            "viktigaste sökord."
        ))

    if css_links > 6:
        pool.append((3,
            f"Vi räknade {css_links} separata CSS-filer. Varje fil är en extra "
            f"nätverksförfrågan som fördröjer att sidan visas — de kan ofta "
            f"slås ihop till en."
        ))

    if imgs_no_alt > 0 and total_imgs > 0:
        pool.append((3,
            f"{imgs_no_alt} av {total_imgs} bilder saknar alt-text. Google kan inte "
            f"tolka bilderna, och tillgängligheten för synskadade besökare försämras."
        ))

    pool.sort(key=lambda x: x[0], reverse=True)
    insights = [text for _, text in pool[:3]]

    return score, load_time, has_viewport, insights


# ── Request / Response schemas ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    score: Optional[int] = None
    load_time: Optional[float] = None
    mobile: bool = True
    insights: List[str] = []
    company_name: str = "din sajt"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "scout-api"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    metrics = await _scrape(url)

    if not metrics:
        return AnalyzeResponse()

    score, load_time, mobile_ok, insights = _analyze(metrics)

    return AnalyzeResponse(
        score=score,
        load_time=load_time,
        mobile=mobile_ok,
        insights=insights,
        company_name=metrics["company_name"],
    )
