"""
Scout API — FastAPI endpoint for fvno.se hero widget.

POST /analyze  {"url": "https://..."}
→ {score, load_time, mobile, insights, company_name}

Run:
    uvicorn api_server:app --port 8503 --host 0.0.0.0
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Scout Analyze API", version="1.0")

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

# ── PageSpeed ─────────────────────────────────────────────────────────────────
_PS_BASE = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
_PS_KEY  = os.getenv("PAGESPEED_API_KEY", "") or os.getenv("GOOGLE_PAGESPEED_API_KEY", "")
_PS_TIMEOUT = 28  # seconds — PageSpeed can be slow


def _ps_params(url: str, strategy: str) -> dict:
    p = {"url": url, "strategy": strategy, "category": "performance"}
    if _PS_KEY:
        p["key"] = _PS_KEY
    return p


async def _fetch_pagespeed(
    client: httpx.AsyncClient, url: str, strategy: str
) -> Optional[dict]:
    try:
        r = await client.get(
            _PS_BASE,
            params=_ps_params(url, strategy),
            timeout=_PS_TIMEOUT,
        )
        if r.status_code != 200:
            print(
                f"[API] PageSpeed {strategy} HTTP {r.status_code}\n{r.text[:2000]}",
                file=sys.stderr,
            )
            return None
        return r.json()
    except httpx.HTTPStatusError as exc:
        print(
            f"[API] PageSpeed {strategy} HTTP {exc.response.status_code}\n"
            f"{exc.response.text[:2000]}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(f"[API] PageSpeed {strategy} failed: {exc}", file=sys.stderr)
        return None


# ── Page title scrape ─────────────────────────────────────────────────────────
async def _fetch_company_name(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(
            url,
            timeout=8,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 Scout/1.0"},
        )
        soup = BeautifulSoup(r.text, "html.parser")

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
    except Exception:
        pass

    from urllib.parse import urlparse
    host = urlparse(url).netloc.removeprefix("www.")
    return host.split(".")[0].title() or "din sajt"


# ── Insight generation ────────────────────────────────────────────────────────
def _insights(
    mobile_data: Optional[dict],
    desktop_score: Optional[int],
) -> tuple[List[str], Optional[int], Optional[float], bool]:
    """
    Derive up to 3 specific insights from actual PageSpeed audit values.

    Rules:
    - Every insight embeds the real measured number.
    - No generic truisms — if we can't measure it, we don't say it.
    - Fallback message ("kunde inte hämta") goes LAST, only if we have
      exactly 2 real insights. If we have 0–1 or 3+, it's hidden entirely.

    Returns (insights, score, load_time_seconds, mobile_ok).
    """
    _FALLBACK = (
        "Vi kunde inte hämta all prestanda-data — analysen ovan baseras på "
        "delresultat. Prova igen om en stund för ett komplett underlag."
    )

    if not mobile_data:
        return [], None, None, True

    cats   = mobile_data.get("lighthouseResult", {}).get("categories", {})
    audits = mobile_data.get("lighthouseResult", {}).get("audits", {})

    def _nv(key: str) -> Optional[float]:
        return (audits.get(key) or {}).get("numericValue")

    def _score(key: str) -> Optional[float]:
        return (audits.get(key) or {}).get("score")

    def _items(key: str) -> int:
        return len((audits.get(key) or {}).get("details", {}).get("items") or [])

    def _savings(key: str) -> Optional[float]:
        return (audits.get(key) or {}).get("details", {}).get("overallSavingsMs")

    perf      = (_nv("metrics") and (cats.get("performance") or {}).get("score"))
    score     = round(perf * 100) if perf is not None else None
    # Re-derive score from categories directly (metrics nv is TTI, not score)
    perf_cat  = (cats.get("performance") or {}).get("score")
    score     = round(perf_cat * 100) if perf_cat is not None else None

    lcp_ms    = _nv("largest-contentful-paint")
    cls_val   = _nv("cumulative-layout-shift")
    tbt_ms    = _nv("total-blocking-time")
    fcp_ms    = _nv("first-contentful-paint")
    tti_ms    = _nv("interactive")
    ttfb_ms   = _nv("server-response-time")
    boot_ms   = _nv("bootup-time")
    thread_ms = _nv("mainthread-work-breakdown")
    page_bytes= _nv("total-byte-weight")

    viewport_ok = _score("viewport-insight") != 0

    load_time = round(lcp_ms / 1000, 1) if lcp_ms else None

    # ── Pool: (priority, text) — higher priority = picked first ──────────────
    pool: List[tuple[int, str]] = []

    # LCP — most impactful for conversion
    if lcp_ms:
        lcp_s = round(lcp_ms / 1000, 1)
        if lcp_s > 4.0:
            pool.append((10,
                f"Din sida tar {lcp_s} sekunder att ladda klart på mobil — "
                f"Google sätter ribban vid 2,5s. Med den laddningstiden försvinner "
                f"uppskattningsvis varannan mobilbesökare innan de läst din rubrik."
            ))
        elif lcp_s > 2.5:
            pool.append((8,
                f"Huvudinnehållet visas efter {lcp_s}s på mobil — strax över "
                f"Googles rekommenderade 2,5s. Det räcker för att tappa sökrankning "
                f"och skicka tveksamma besökare vidare till nästa träff."
            ))
        elif lcp_s <= 1.5 and score and score >= 80:
            pool.append((2,
                f"Sidan laddar på {lcp_s}s — snabbt för mobil. Den tekniska "
                f"grunden är stark; nästa steg är att säkerställa att budskapet "
                f"och kontaktknapparna konverterar lika bra."
            ))

    # Mobile vs desktop gap
    if score is not None and desktop_score is not None and desktop_score > score + 20:
        gap = desktop_score - score
        pool.append((9,
            f"Mobilversionen får {score}/100 i prestanda medan datorversionen "
            f"får {desktop_score}/100 — ett gap på {gap} poäng. Mobilbesökare, "
            f"som ofta är de närmast ett köpbeslut, upplever alltså en märkbart sämre sajt."
        ))

    # TBT — interactivity
    if tbt_ms:
        tbt_r = round(tbt_ms)
        if tbt_ms > 600:
            pool.append((9,
                f"Sidan blockerar interaktion i {tbt_r} millisekunder — knappar "
                f"och formulär svarar trögt när besökaren klickar. Det upplevs "
                f"direkt som att sidan hänger sig och driver bort folk som annars "
                f"hade fyllt i kontaktformuläret."
            ))
        elif tbt_ms > 200:
            pool.append((5,
                f"Det finns {tbt_r}ms fördröjning på interaktioner (klick, formulär). "
                f"Knappt märkbart på en ny telefon — men på en äldre mobil upplevs "
                f"det som tröghet och kan kosta dig kontakter."
            ))

    # CLS — layout shift
    if cls_val is not None:
        cls_r = round(cls_val, 2)
        if cls_val > 0.25:
            pool.append((9,
                f"Layouten hoppar märkbart (CLS {cls_r}) när sidan laddar — "
                f"element rör sig och knappar hamnar tillfälligt på fel ställe. "
                f"Det skapar frustration och upplevs direkt som oprofessionellt."
            ))
        elif cls_val > 0.1:
            pool.append((6,
                f"Det finns rörelse i layouten (CLS {cls_r}) medan sidan laddar — "
                f"text och knappar hoppar lite. Google noterar det och det stör "
                f"besökare tillräckligt för att minska konverteringen."
            ))

    # TTFB — server speed
    if ttfb_ms and ttfb_ms > 800:
        ttfb_s = round(ttfb_ms / 1000, 1)
        pool.append((8,
            f"Servern svarar på {ttfb_s}s innan sidan ens börjar ladda — "
            f"Google rekommenderar under 0,6s. Byte av hosting-paket är ofta "
            f"den snabbaste och billigaste prestandaförbättringen."
        ))
    elif ttfb_ms and ttfb_ms > 400:
        ttfb_ms_r = round(ttfb_ms)
        pool.append((4,
            f"Servern tar {ttfb_ms_r}ms att svara — lite trögare än optimalt. "
            f"En CDN eller snabbare hosting kan halvera detta utan att ändra ett "
            f"enda ord på sidan."
        ))

    # JS bootup
    if boot_ms and boot_ms > 2000:
        boot_s = round(boot_ms / 1000, 1)
        pool.append((7,
            f"JavaScript-koden tar {boot_s}s att tolka och köra — det är en vanlig "
            f"orsak till trög interaktivitet på mobil. Ofta beror det på tunga "
            f"plugins eller tredjepartsskript som kan trimmas."
        ))

    # Page weight
    if page_bytes and page_bytes > 3_000_000:
        mb = round(page_bytes / 1_000_000, 1)
        pool.append((6,
            f"Sidan laddar {mb} MB data totalt — ungefär tre gånger mer än "
            f"rekommenderat för mobil. Det är oftast okomprimerade bilder eller "
            f"inaktiva tillägg som kan tas bort utan att sidan förändras."
        ))
    elif page_bytes and page_bytes > 1_500_000:
        mb = round(page_bytes / 1_000_000, 1)
        pool.append((3,
            f"Sidan väger {mb} MB — lite tungt för mobil. "
            f"Bildoptimering ensam kan ofta halvera sidvikten."
        ))

    # Render-blocking resources
    rb_items = _items("render-blocking-insight")
    if _score("render-blocking-insight") == 0 and rb_items > 0:
        pool.append((6,
            f"Vi hittade {rb_items} skript eller stilmallar som blockerar sidan "
            f"innan besökaren ser något — de laddas i onödan tidigt och fördröjer "
            f"hela sidvisningen."
        ))

    # Image delivery
    img_items = _items("image-delivery-insight")
    if _score("image-delivery-insight") is not None and _score("image-delivery-insight") < 0.9 and img_items > 0:
        pool.append((5,
            f"Vi hittade {img_items} bilder som inte är optimerade för mobil — "
            f"de laddas i originalstorlek även på en liten skärm och saktar "
            f"ner sidan i onödan."
        ))

    # Viewport (mobile unfriendly)
    if not viewport_ok:
        pool.append((10,
            "Sidan saknar mobilanpassning — vi hittade ingen viewport-inställning, "
            "vilket innebär att mobilbesökare ser en zoomad ut datorvy. "
            "Google straffar det hårt i mobilsökresultaten."
        ))

    # ── Pick top 3 by priority ────────────────────────────────────────────────
    pool.sort(key=lambda x: x[0], reverse=True)
    insights = [text for _, text in pool[:3]]

    # Fallback: only append as #3 when we have exactly 2 real insights.
    # With 0–1 or 3+ real insights it stays hidden.
    if len(insights) == 2:
        insights.append(_FALLBACK)

    return insights, score, load_time, viewport_ok


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

    async with httpx.AsyncClient() as client:
        mobile_task, desktop_task, name_task = await asyncio.gather(
            asyncio.create_task(_fetch_pagespeed(client, url, "mobile")),
            asyncio.create_task(_fetch_pagespeed(client, url, "desktop")),
            asyncio.create_task(_fetch_company_name(client, url)),
            return_exceptions=True,
        )

    mobile_data = mobile_task if isinstance(mobile_task, dict) else None
    desktop_data = desktop_task if isinstance(desktop_task, dict) else None
    company_name = name_task if isinstance(name_task, str) else "din sajt"

    d_cats = (desktop_data or {}).get("lighthouseResult", {}).get("categories", {})
    d_perf = (d_cats.get("performance") or {}).get("score")
    desktop_score = round(d_perf * 100) if d_perf is not None else None

    insights, score, load_time, mobile_ok = _insights(mobile_data, desktop_score)

    return AnalyzeResponse(
        score=score,
        load_time=load_time,
        mobile=mobile_ok,
        insights=insights,
        company_name=company_name,
    )
