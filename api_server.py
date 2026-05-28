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
from fastapi import FastAPI, HTTPException
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
_BLOCKED_URL_FRAGMENTS = ("localhost", "127.0.0.1", "railway.app")
_SMB_FALLBACK_INSIGHTS = [
    (
        "Många småföretagssajter saknar Google Analytics-spårning — utan det är det "
        "svårt att veta vilka besökare som faktiskt leder till affärer."
    ),
    (
        "Sidan verkar sakna automatisk SSL-redirect — besökare som skriver http:// "
        "når inte alltid den säkra https-versionen."
    ),
    (
        "Avsaknad av strukturerad data för lokal SEO gör det svårare för Google att "
        "visa ert företag i lokala sökresultat."
    ),
]


def _is_blocked_url(url: str) -> bool:
    lower = url.lower()
    return any(fragment in lower for fragment in _BLOCKED_URL_FRAGMENTS)


def _ensure_three_insights(insights: List[str]) -> List[str]:
    result = list(insights)
    for fallback in _SMB_FALLBACK_INSIGHTS:
        if len(result) >= 3:
            break
        if fallback not in result:
            result.append(fallback)
    return result[:3]


_SWEDISH_PHONE_RE = re.compile(
    r"(?:"
    r"\+46[\s\-()]*(?:7[02369]|[1-9]\d)[\s\-()]*\d{2,3}[\s\-()]*\d{2}[\s\-()]*\d{2}|"
    r"0(?:7[02369]|[1-9]\d)[\s\-()]*\d{2,3}[\s\-()]*\d{2}[\s\-()]*\d{2}|"
    r"0\d{1,3}[\s\-]\d{2,3}[\s\-]\d{2}[\s\-]?\d{2}"
    r")"
)


def _page_domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def _count_internal_links(soup: BeautifulSoup, page_url: str) -> int:
    domain = _page_domain(page_url)
    count = 0

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        parsed = urlparse(href)
        if parsed.netloc:
            link_domain = parsed.netloc.removeprefix("www.").lower()
            if link_domain == domain:
                count += 1
        elif href.startswith("/") or not parsed.scheme:
            count += 1

    return count


def _has_contact_info(soup: BeautifulSoup) -> bool:
    if soup.find("a", href=re.compile(r"^tel:", re.I)):
        return True

    page_text = soup.get_text(separator=" ", strip=True)
    return bool(_SWEDISH_PHONE_RE.search(page_text))


def _visible_word_count(soup: BeautifulSoup) -> int:
    clone = BeautifulSoup(str(soup), "html.parser")
    for tag in clone(["script", "style", "noscript"]):
        tag.decompose()
    text = clone.get_text(separator=" ", strip=True)
    return len(re.findall(r"\b[\wåäöÅÄÖ]+\b", text, re.UNICODE))


def _has_social_links(soup: BeautifulSoup) -> bool:
    social_hosts = ("facebook.com", "instagram.com", "linkedin.com")
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).lower()
        if any(host in href for host in social_hosts):
            return True
    return False


def _has_favicon(soup: BeautifulSoup) -> bool:
    for link in soup.find_all("link", rel=True):
        rel = link.get("rel")
        rel_text = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if "icon" in rel_text:
            return True
    return False


def _meta_description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta and meta.get("content"):
        return str(meta["content"]).strip()
    return ""


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def _extract_content_metrics(soup: BeautifulSoup, page_url: str) -> dict:
    title = _page_title(soup)
    meta_desc = _meta_description(soup)

    return {
        "title": title,
        "title_length": len(title),
        "has_meta_description": bool(meta_desc),
        "meta_description_length": len(meta_desc),
        "internal_links": _count_internal_links(soup, page_url),
        "has_contact_info": _has_contact_info(soup),
        "has_h2": bool(soup.find("h2")),
        "has_h3": bool(soup.find("h3")),
        "word_count": _visible_word_count(soup),
        "has_social_links": _has_social_links(soup),
        "has_favicon": _has_favicon(soup),
    }


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
        page_url = str(r.url)
        content = _extract_content_metrics(soup, page_url)

        return {
            "load_time": load_time,
            "size_bytes": size_bytes,
            "scripts": scripts,
            "css_links": css_links,
            "has_viewport": has_viewport,
            "has_h1": has_h1,
            "total_imgs": len(imgs),
            "imgs_without_alt": imgs_without_alt,
            "company_name": _extract_company_name(soup, page_url),
            "status_code": r.status_code,
            **content,
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
    Derive score (0–100), load_time, mobile_ok, and three insights
    from scraped metrics. The insight pool holds 10–12+ candidates;
    the three highest-priority findings are returned, padded if needed.
    """
    load_time   = m["load_time"]
    size_mb     = m["size_bytes"] / 1_000_000
    scripts     = m["scripts"]
    css_links   = m["css_links"]
    has_viewport = m["has_viewport"]
    has_h1      = m["has_h1"]
    total_imgs  = m["total_imgs"]
    imgs_no_alt = m["imgs_without_alt"]
    title_length = m["title_length"]
    has_meta_description = m["has_meta_description"]
    internal_links = m["internal_links"]
    has_contact_info = m["has_contact_info"]
    has_h2 = m["has_h2"]
    has_h3 = m["has_h3"]
    word_count = m["word_count"]
    has_social_links = m["has_social_links"]
    has_favicon = m["has_favicon"]

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
    if not has_meta_description: score -= 8
    if title_length < 10: score -= 8
    if word_count < 300:  score -= 6
    if not has_contact_info: score -= 5
    if internal_links < 3: score -= 4

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

    if not has_meta_description:
        pool.append((8,
            "Sidan saknar meta description — den korta texten som visas under "
            "sidans titel i Google. Utan den tappar ni chansen att styra första "
            "intrycket och få fler klick från sökresultatet."
        ))

    if title_length < 10:
        pool.append((8,
            "Sidans titel är kort eller saknas — det är det första Google läser "
            "och det som visas i webbläsarens flik. En tydlig titel gör det lättare "
            "att förstå vem ni är och vad ni erbjuder."
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

    if not has_contact_info:
        pool.append((6,
            "Vi hittar inget telefonnummer eller tydlig kontaktväg i sidans innehåll. "
            "Besökare som är redo att höra av sig måste leta vidare — och många "
            "gör det hos konkurrenten istället."
        ))

    if word_count < 300:
        pool.append((6,
            f"Sidan innehåller bara cirka {word_count} ord synlig text — Google "
            f"behöver mer innehåll för att förstå vad ni gör och matcha er mot "
            f"relevanta sökningar."
        ))

    if internal_links < 3:
        pool.append((5,
            f"Sidan har bara {internal_links} interna länkar — det gör det svårare "
            f"för besökare och Google att hitta vidare innehåll. En tydlig "
            f"länkstruktur hjälper både navigation och synlighet."
        ))

    if not has_h2 and not has_h3:
        pool.append((5,
            "Sidan saknar underrubriker — utan tydliga H2- och H3-rubriker blir "
            "innehållet svårare att skumma igenom och Google får sämre struktur "
            "att förstå vad sidan handlar om."
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

    if not has_social_links:
        pool.append((3,
            "Sidan länkar inte till sociala medier — det är en enkel signal om "
            "att företaget är aktivt utåt. Utan länkar missar ni chansen att "
            "bygga förtroende redan på startsidan."
        ))

    if not has_favicon:
        pool.append((2,
            "Sidan saknar favicon — den lilla ikonen i webbläsarens flik. "
            "Detaljen är liten men påverkar hur professionellt företaget "
            "upplevs när flera flikar är öppna."
        ))

    pool.sort(key=lambda x: x[0], reverse=True)
    insights = _ensure_three_insights([text for _, text in pool[:3]])

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

    if _is_blocked_url(url):
        raise HTTPException(status_code=400, detail="Kan inte analysera denna adress.")

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
