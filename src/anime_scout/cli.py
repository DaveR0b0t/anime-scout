#!/user/bin/env python3

"""

anime_scout.py

A small CLI that queries the Anilist GraphQL API to:
- search anime by title and/or genre
- show details + description
- show official external links (where to watch)
- open/play trailers locally (YouTube)

Note: "Dubbed" is not reliably available in general metadata APIs.
We show "Dubbed: Unknown" for now (can be extended later).
"""

import argparse
import json
import re
import subprocess
import sys
import hashlib
import time
import textwrap
import webbrowser
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import date
from typing import Any, Dict, List, Optional

import requests

console = Console()

ANILIST_API = "https://graphql.anilist.co"

# ------------------------
# Cache + Rate limiting
# ------------------------

# Default cache location: `/.cache/anime-scout (Linux)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "anime-scout"

# Simple global rate limit: at most 1 request per MIN_INTERVAL seconds
MIN_INTERVAL_SECONDS = 1.0
_last_request_time = 0.0 # monotonic seconds

def rate_limit(min_interval: float = MIN_INTERVAL_SECONDS) -> None:
    """
    Ensure we don't send request too quickly.
    Uses time.monotonic() so it isn't affected by clock changes
    :param min_interval:
    :return:
    """
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    _last_request_time = time.monotonic()

def cache_key(query: str, variables: dict) -> str:
    """
    Creat a stable hash key for (query + variables).
    We sort keys in JSON so the same variables always hash the same.

    :param query:
    :param variables:
    :return:
    """
    payload = {"query": query, "variables": variables}
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()

def cache_path_for(key: str, cache_dir: Path) -> Path:
    """
    Store each cached response as one file:
         <cache_dir>/<first2>/<hash>.json
    This avoids too many files in one directory.
    :param key:
    :param cache_dir:
    :return:
    """
    sub = cache_dir / key[:2]
    return sub / f"{key}.json"

def cache_get(key: str, cache_dir: Path, ttl_seconds: int) -> dict | None:
    """
    Return cached JSON (dict) if present and not expired, else None.
    :param key:
    :param cache_dir:
    :param ttl_seconds:
    :return:
    """
    path = cache_path_for(key, cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Expect {"saved_at": <unix>, "data": {...}}
    saved_at = data.get("saved_at")
    payload = data.get("data")

    if not isinstance(saved_at, (int, float)) or payload is None:
        return None

    age = time.time() - float(saved_at)
    if age > ttl_seconds:
        return None

    return payload

def cache_set(key: str, cache_dir: Path, payload: dict) -> None:
    """
    Save payload to cache as JSON with timestamp
    :param key:
    :param cache_dr:
    :param payload:
    :return:
    """
    path = cache_path_for(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = {
        "saved_at": time.time(),
        "data": payload
    }

    path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------
# Helpers: HTTP + GraphQL
# -----------------------


def anilist_query(
    query: str,
    variables: Dict[str, Any],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl: int = 60 * 60 * 24,   # 24 hours
    use_cache: bool = True,
    min_interval: float = MIN_INTERVAL_SECONDS
) -> Dict[str, Any]:
    """
    Send a GraphQL query to AniList and return response["data"].

    Added:
    - rate limiting (min_interval seconds between requests)
    - file cache (TTL-based)
    """
    key = cache_key(query, variables)

    # 1) Try cache first
    if use_cache:
        cached = cache_get(key, cache_dir, cache_ttl)
        if cached is not None:
            return cached

    # 2) Rate limit before making the request
    rate_limit(min_interval=min_interval)

    # 3) Make the request
    resp = requests.post(
        ANILIST_API,
        json={"query": query, "variables": variables},
        timeout=20,
        headers={"User-Agent": "anime-scout/0.1 (CLI; learning project)"}
    )
    resp.raise_for_status()
    payload = resp.json()

    if "errors" in payload:
        raise RuntimeError(json.dumps(payload["errors"], indent=2))

    data = payload["data"]

    # 4) Save to cache
    if use_cache:
        cache_set(key, cache_dir, data)

    return data


def strip_html(text: str) -> str:
    """
    AniList descriptions often contain HTML tags. This removes tags
    and collapses whitespace so it reads nicely in a terminal.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text or "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# AniList Queries
# -----------------------


SEARCH_QUERY = """
query ($search: String, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { currentPage hasNextPage }
    media(type: ANIME, search: $search, sort: POPULARITY_DESC) {
      id
      title { romaji english native }
      format
      status
      episodes
      season
      seasonYear
      genres
    }
  }
}
"""


INFO_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title { romaji english native }
    format
    status
    episodes
    duration
    season
    seasonYear
    genres
    averageScore
    popularity
    description(asHtml: true)

    # Trailer metadata (commonly YouTube)
    trailer { id site }

    # Official / external links (often includes streaming services)
    externalLinks {
      site
      url
      language
    }
  }
}
"""

TRENDING_QUERY = """
query ($page: Int, $perPage: Int, $season: MediaSeason, $seasonYear: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, season: $season, seasonYear: $seasonYear, sort: TRENDING_DESC) {
      id
      title { romaji english native }
      format
      status
      episodes
      season
      seasonYear
      genres
    }
  }
}
"""

POPULAR_QUERY = """
query ($page: Int, $perPage: Int, $season: MediaSeason, $seasonYear: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, season: $season, seasonYear: $seasonYear, sort: POPULARITY_DESC) {
      id
      title { romaji english native }
      format
      status
      episodes
      season
      seasonYear
      genres
    }
  }
}
"""


def normalize_genre(g: str) -> str:
    """Normalize genre strings for case-insensitive matching."""
    return (g or "").strip().lower()


def parse_genre_inputs(genres: list[str] | None, genres_csv: str | None) -> list[str]:
    """
    Combine repeatable -- genre values and comma-separated -- genres into one list.
    Example:
        --genre Action --genre "Slice of Life" --genres "Combedy, Drama"
    :param genres:
    :param genres_csv:
    :return:
    """

    out: list[str] = []

    if genres:
        out.extend(genres)

    if genres_csv:
        out.extend([part.strip() for part in genres_csv.split(",")])

    # normalize, remove empties, de-dup (preserve order)
    seen = set()
    cleaned = []
    for g in out:
        ng = normalize_genre(g)
        if not ng or ng in seen:
            continue
        seen.add(ng)
        cleaned.append(ng)

    return cleaned

def matches_genres(anime_genres: list[str] | None, wanted: list[str], match_mode: str) -> bool:
    """
    match_mode:
        - 'any': at least one wanted genre is present
        - 'all': all wanted genres must be present
    :param anime_genres:
    :param wanted:
    :param match_mode:
    :return:
    """
    if not wanted:
        return True

    have = {normalize_genre(g) for g in (anime_genres or [])}

    if match_mode == "all":
        return all(g in have for g in wanted)
    return any(g in have for g in wanted) # default 'any'

def current_season_and_year(today: date | None = None) -> tuple[str, int]:
    """
    Approximate AniList season from month:
      WINTER: Jan–Mar
      SPRING: Apr–Jun
      SUMMER: Jul–Sep
      FALL:   Oct–Dec
    """
    today = today or date.today()
    m = today.month
    y = today.year

    if 1 <= m <= 3:
        return "WINTER", y
    if 4 <= m <= 6:
        return "SPRING", y
    if 7 <= m <= 9:
        return "SUMMER", y
    return "FALL", y


def next_season_and_year(today: date | None = None) -> tuple[str, int]:
    season, year = current_season_and_year(today)
    order = ["WINTER", "SPRING", "SUMMER", "FALL"]
    i = order.index(season)
    nxt = order[(i + 1) % 4]
    # If we wrap from FALL -> WINTER, year increments
    if season == "FALL":
        year += 1
    return nxt, year


def normalize_season_input(season: str) -> str:
    """
    Accepts: current, next, winter, spring, summer, fall (any case)
    Returns: CURRENT/NEXT or WINTER/SPRING/SUMMER/FALL
    """
    s = (season or "").strip().upper()
    if s in {"CURRENT", "NOW"}:
        return "CURRENT"
    if s in {"NEXT", "UPCOMING"}:
        return "NEXT"
    if s in {"WINTER", "SPRING", "SUMMER", "FALL"}:
        return s
    raise ValueError("Season must be one of: current, next, WINTER, SPRING, SUMMER, FALL")

def compute_season_vars(season_arg: str | None, year_arg: int | None) -> tuple[Optional[str], Optional[int]]:
    """
    Returns (season, seasonYear) or (None, None) if no filter is requested.
    """
    if not season_arg:
        return None, None

    s = normalize_season_input(season_arg)

    if s == "CURRENT":
        season, year = current_season_and_year()
        return season, (year_arg if year_arg is not None else year)

    if s == "NEXT":
        season, year = next_season_and_year()
        return season, (year_arg if year_arg is not None else year)

    # Explicit season given (WINTER/SPRING/SUMMER/FALL)
    if year_arg is None:
        # If user specified a season without year, default to current year (reasonable default)
        _, current_year = current_season_and_year()
        year_arg = current_year
    return s, year_arg


# --------------------------
# Printing / formatting
# --------------------------

def pick_title(title_obj: Dict[str, Any]) -> str:
    """Choose the best display title: English -> Romaji -> Native."""
    return(
        title_obj.get("english")
        or title_obj.get("romaji")
        or title_obj.get("native")
        or "Unknown Title"
    )



def cmd_trending(args: argparse.Namespace) -> None:
    season, seasonYear = compute_season_vars(args.season, args.year)

    data = anilist_query(
        TRENDING_QUERY,
        {"page": 1, "perPage": args.limit, "season": season, "seasonYear": seasonYear},
        cache_dir=Path(args.cache_dir).expanduser(),
        cache_ttl=args.cache_ttl,
        use_cache=not args.no_cache,
        min_interval=args.rate
    )
    items = data["Page"]["media"] or []
    if not items:
        console.print("[dim]No results.[/dim]")
        return

    label = "Trending anime"
    if season and seasonYear:
        label += f" — {season} {seasonYear}"

    console.print(f"[bold]{label}[/bold]\n")
    print_search_results(items)


def cmd_popular(args: argparse.Namespace) -> None:
    season, seasonYear = compute_season_vars(args.season, args.year)

    data = anilist_query(
        TRENDING_QUERY,
        {"page": 1, "perPage": args.limit, "season": season, "seasonYear": seasonYear},
        cache_dir=Path(args.cache_dir).expanduser(),
        cache_ttl=args.cache_ttl,
        use_cache=not args.no_cache,
        min_interval=args.rate
    )
    items = data["Page"]["media"] or []
    if not items:
        console.print("[dim]No results.[/dim]")
        return

    label = "Trending anime"
    if season and seasonYear:
        label += f" — {season} {seasonYear}"

    console.print(f"[bold]{label}[/bold]\n")
    print_search_results(items)


def print_search_results(items: List[Dict[str, Any]]) -> None:
    """
    Print search results as a nice table:
    - ID
    - Title
    - Meta (format/season/episodes/status)
    - Genres (first few)
    """
    table = Table(title="Search Results", box=box.SIMPLE_HEAVY)
    table.add_column("ID", justify="right", style="bold")
    table.add_column("Title", style="bold")
    table.add_column("Meta")
    table.add_column("Genres")

    for m in items:
        title = pick_title(m["title"])

        meta_bits = []
        if m.get("format"):
            meta_bits.append(m["format"])
        if m.get("season") and m.get("seasonYear"):
            meta_bits.append(f"{m['season']} {m['seasonYear']}")
        if m.get("episodes"):
            meta_bits.append(f"{m['episodes']} eps")
        if m.get("status"):
            meta_bits.append(m["status"])

        meta = " • ".join(meta_bits)

        genres = m.get("genres") or []
        # Keep it readable (avoid overly-wide tables)
        genres_txt = ", ".join(genres[:4]) + ("…" if len(genres) > 4 else "")

        table.add_row(str(m["id"]), title, meta, genres_txt)

    console.print(table)


def print_info(media: Dict[str, Any]) -> None:
    """
    Display a single anime in a readable, structured way.
    """
    title = pick_title(media["title"])
    romaji = media["title"].get("romaji") or ""
    native = media["title"].get("native") or ""

    # Header line
    header = Text()
    header.append(title, style="bold")
    header.append(f"  (AniList ID: {media['id']})", style="dim")

    console.print("\n" + str(header))

    # Alternate titles
    if romaji and romaji != title:
        console.print(f"[dim]Romaji:[/dim] {romaji}")
    if native and native not in (title, romaji):
        console.print(f"[dim]Native:[/dim] {native}")

    # Facts row
    facts = []
    if media.get("format"):
        facts.append(media["format"])
    if media.get("episodes"):
        facts.append(f"{media['episodes']} eps")
    if media.get("duration"):
        facts.append(f"{media['duration']} min/ep")
    if media.get("status"):
        facts.append(media["status"])
    if media.get("season") and media.get("seasonYear"):
        facts.append(f"{media['season']} {media['seasonYear']}")
    if media.get("averageScore"):
        facts.append(f"Score: {media['averageScore']}/100")

    if facts:
        console.print(" • ".join(facts))

    # Genres + Dubbed
    if media.get("genres"):
        console.print(f"[dim]Genres:[/dim] {', '.join(media['genres'])}")

    console.print("[dim]Dubbed:[/dim] Unknown (depends on streaming service/region)")

    # Description (strip HTML, then show as a panel)
    desc = strip_html(media.get("description") or "")
    if desc:
        console.print(Panel(desc, title="Description", box=box.ROUNDED))

    # Where to watch / official links
    links = media.get("externalLinks") or []
    if links:
        watch_table = Table(title="Where to watch / official links", box=box.SIMPLE)
        watch_table.add_column("Site", style="bold")
        watch_table.add_column("Language", style="dim", width=10)
        watch_table.add_column("URL")

        for l in links:
            site = l.get("site") or "Link"
            url = l.get("url") or ""
            lang = l.get("language") or ""
            if url:
                watch_table.add_row(site, lang, url)

        console.print(watch_table)

    # Trailer
    tr = media.get("trailer")
    if tr and tr.get("site") and tr.get("id"):
        trailer_url = trailer_to_url(tr["site"], tr["id"])
        console.print(f"[bold]Trailer:[/bold] {trailer_url}")
    else:
        console.print("[bold]Trailer:[/bold] (none listed)")


def trailer_to_url(site: str, vid: str) -> str:
    """
    Convert AniList trailer metadata into a playable URL.
    """
    site_l = site.lower()
    if site_l == "youtube":
        return f"https://www.youtube.com/watch?v={vid}"
    return vid


# ------------------------------
# CLI actions
# ------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    wanted = parse_genre_inputs(args.genre, args.genres)

    # How many items to fetch per page from AniList
    per_page = max(25, min(args.fetch, 50))  # AniList perPage max is typically 50

    matches: list[dict] = []
    page = 1

    # Keep paging until we have enough matches or hit max-pages
    while len(matches) < args.limit and page <= args.max_pages:
        data = anilist_query(
            SEARCH_QUERY,
            {"search": args.query, "page": page, "perPage": per_page},
            cache_dir=Path(args.cache_dir).expanduser(),
            cache_ttl=args.cache_ttl,
            use_cache=not args.no_cache,
            min_interval=args.rate
        )

        items = data["Page"]["media"] or []
        if not items:
            break  # no more results

        if wanted:
            items = [m for m in items if matches_genres(m.get("genres"), wanted, args.match)]

        matches.extend(items)
        page += 1

    # Trim to limit
    matches = matches[: args.limit]

    if not matches:
        if wanted:
            console.print(f"[dim]No results.[/dim] (Filters: {args.match} of {', '.join(wanted)})")
        else:
            console.print("[dim]No results.[/dim]")
        return

    if wanted:
        console.print(f"[dim]Filters:[/dim] {args.match} of {', '.join(wanted)}")
        console.print(f"[dim]Scanned up to {page-1} page(s). Showing {len(matches)} result(s).[/dim]\n")

    print_search_results(matches)



def cmd_info(args: argparse.Namespace) -> None:
    data = anilist_query(INFO_QUERY, {"id": args.id})
    media = data["Media"]
    print_info(media)


def cmd_trailer(arg: argparse.Namespace) -> None:
    # Fetch media just to get trailer info
    data = anilist_query(INFO_QUERY, {"id": arg.id})
    media = data["Media"]

    tr = media.get("trailer")
    if not tr or not tr.get("site") or not tr.get("id"):
        print("No trailer listed for this anime.")
        return

    url = trailer_to_url(tr["site"], tr["id"])

    if arg.open:
        # Open in default browser
        webbrowser.open(url)
        print(f"Open Trailer: {url}")
        return

    if arg.mpv:
        # Play locally via mpv (must be installed)
        # mpv can play Youtube URLs if it's built with ytdl support or configured
        try:
            subprocess.run(["mpv", url], check=True)
        except FileNotFoundError:
            print("mpv not found. Install it or use --open instead.")
            sys.exit(1)
        except subprocess.CalledProcessError:
            print("mpv failed to play the URL. Try ---open instead.")
            sys.exit(1)
        return

    # Defualt behavior: just print the URL
    print(url)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anime", description="Anime search/infor CLI (AniList)")

    p.add_argument("--no-cache", action="store_true", help="Disable cache")
    p.add_argument("--cache-ttl", type=int, default=60 * 60 * 24,
                   help="Cache TTL in seconds (default: 86400 / 24h)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Cache directory (default: ~/.cache/anime-scout)")
    p.add_argument("--rate", type=float, default=MIN_INTERVAL_SECONDS,
                   help="Minimum seconds between API requests (default: 1.0)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # search
    ps = sub.add_parser("search", help="Search anime by title and/or genre(s)")
    ps.add_argument("--max-pages", type=int, default=10,
                    help="Max pages to scan for genre matches (default: 10)")
    ps.add_argument("query", nargs="?", default=None, help="Anime title (optional if genre filters used)")

    # trending
    ptr = sub.add_parser("trending", help="Show currently trending anime")
    ptr.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    ptr.add_argument("--season", help="Filter by season: current, next, WINTER, SPRING, SUMMER, FALL")
    ptr.add_argument("--year", type=int,
                     help="Season year (used with --season). If omitted with current/next, auto-calculated.")
    ptr.set_defaults(func=cmd_trending)

    # popular
    pp = sub.add_parser("popular", help="Show currently popular anime")
    pp.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    pp.set_defaults(func=cmd_popular)

    # repeatable: --genre Action --genre Comedy
    ps.add_argument("--genre", action="append",
                    help="Genre filter (repeatable). Example: --genre Action --genre Comedy")

    # comma-separated convenience: --genres "Action, Comedy"
    ps.add_argument("--genres", help='Comma-separated genres. Example: --genres "Action, Comedy"')

    # OR vs AND matching
    ps.add_argument("--match", choices=["any", "all"], default="any",
                    help="Genre match mode: any (OR) or all (AND). Default: any")

    # When filtering client-side, fetch a bit more than limit so filtering doesn't empty results
    ps.add_argument("--limit", type=int, default=10, help="Max results to show (default: 10)")
    ps.add_argument("--fetch", type=int, default=50,
                    help="How many results to fetch before filtering (default: 50)")

    ps.set_defaults(func=cmd_search)

    # info
    pi = sub.add_parser("info", help="Show detailed info for an AniList anime ID")
    pi.add_argument("id", type=int, help="AniList anime ID (from search)")
    pi.set_defaults(func=cmd_info)

    # trailer
    pt = sub.add_parser("trailer", help="Show/open/play trailer for an AniList anime ID")
    pt.add_argument("id", type=int, help="AniList anime ID")
    pt.add_argument("--open", action="store_true", help="Open trailer in web browser")
    pt.add_argument("--mpv", action="store_true", help="Play trailer locally with mpv (if installed)")
    pt.set_defaults(func=cmd_trailer)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)



if __name__ == "__main__":
    main()


