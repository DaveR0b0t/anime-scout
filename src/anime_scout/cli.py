#!/usr/bin/env python3

"""
anime_scout.py

A small CLI that queries the AniList GraphQL API to:
- search anime by title and/or genre
- show details + description
- show official external links (where to watch)
- open/play trailers locally (YouTube)

Note: "Dubbed" is not reliably available in general metadata APIs.
We show "Dubbed: Unknown" for now.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from urllib.parse import urlparse
import webbrowser
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

ANILIST_API = "https://graphql.anilist.co"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "anime-scout"
MIN_INTERVAL_SECONDS = 1.0
_last_request_time = 0.0


# ------------------------
# Cache + Rate limiting
# ------------------------

def rate_limit(min_interval: float = MIN_INTERVAL_SECONDS) -> None:
    """Ensure we do not send requests too quickly."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    _last_request_time = time.monotonic()



def cache_key(query: str, variables: dict) -> str:
    """Create a stable hash key for query and variables."""
    payload = {"query": query, "variables": variables}
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()



def cache_path_for(key: str, cache_dir: Path) -> Path:
    """Store each cached response as one file under a subdirectory."""
    sub = cache_dir / key[:2]
    return sub / f"{key}.json"



def cache_get(key: str, cache_dir: Path, ttl_seconds: int) -> dict | None:
    """Return cached JSON if present and not expired."""
    path = cache_path_for(key, cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    saved_at = data.get("saved_at")
    payload = data.get("data")

    if not isinstance(saved_at, (int, float)) or payload is None:
        return None

    age = time.time() - float(saved_at)
    if age > ttl_seconds:
        return None

    return payload



def cache_set(key: str, cache_dir: Path, payload: dict) -> None:
    """Save payload to cache as JSON with a timestamp."""
    path = cache_path_for(key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = {
        "saved_at": time.time(),
        "data": payload,
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
    cache_ttl: int = 60 * 60 * 24,
    use_cache: bool = True,
    min_interval: float = MIN_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    """Send a GraphQL query to AniList and return response data."""
    key = cache_key(query, variables)

    if use_cache:
        cached = cache_get(key, cache_dir, cache_ttl)
        if cached is not None:
            return cached

    rate_limit(min_interval=min_interval)

    try:
        resp = requests.post(
            ANILIST_API,
            json={"query": query, "variables": variables},
            timeout=20,
            headers={"User-Agent": "anime-scout/0.1 (CLI; learning project)"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to AniList failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("AniList returned an invalid JSON response.") from exc

    if "errors" in payload:
        messages = "; ".join(
            error.get("message", str(error)) for error in payload["errors"]
        )
        raise RuntimeError(f"AniList API error: {messages}")

    data = payload.get("data")
    if data is None:
        raise RuntimeError("AniList returned no data.")

    if use_cache:
        cache_set(key, cache_dir, data)

    return data



def query_with_args(query: str, variables: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run an AniList query using the CLI cache and rate-limit settings."""
    return anilist_query(
        query,
        variables,
        cache_dir=Path(args.cache_dir).expanduser(),
        cache_ttl=args.cache_ttl,
        use_cache=not args.no_cache,
        min_interval=args.rate,
    )



def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace for terminal output."""
    text = re.sub(r"<[^>]+>", "", text or "")
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
    trailer { id site }
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
    return (g or "").strip().lower()



def parse_genre_inputs(genres: list[str] | None, genres_csv: str | None) -> list[str]:
    """Combine repeatable and comma-separated genre inputs into one list."""
    out: list[str] = []

    if genres:
        out.extend(genres)

    if genres_csv:
        out.extend([part.strip() for part in genres_csv.split(",")])

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
    if not wanted:
        return True

    have = {normalize_genre(g) for g in (anime_genres or [])}

    if match_mode == "all":
        return all(g in have for g in wanted)
    return any(g in have for g in wanted)



def validate_search_args(query: str | None, wanted_genres: list[str]) -> None:
    """Require a search term or at least one genre filter."""
    if query:
        return
    if wanted_genres:
        return
    raise ValueError("Provide a search query or at least one genre filter.")



def current_season_and_year(today: date | None = None) -> tuple[str, int]:
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
    if season == "FALL":
        year += 1
    return nxt, year



def normalize_season_input(season: str) -> str:
    s = (season or "").strip().upper()
    if s in {"CURRENT", "NOW"}:
        return "CURRENT"
    if s in {"NEXT", "UPCOMING"}:
        return "NEXT"
    if s in {"WINTER", "SPRING", "SUMMER", "FALL"}:
        return s
    raise ValueError("Season must be one of: current, next, WINTER, SPRING, SUMMER, FALL")



def positive_int(value: str) -> int:
    """argparse type: require an integer > 0."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return n



def non_negative_int(value: str) -> int:
    """argparse type: require an integer >= 0."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater.")
    return n



def positive_float(value: str) -> float:
    """argparse type: require a float > 0."""
    n = float(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return n



def compute_season_vars(season_arg: str | None, year_arg: int | None) -> tuple[Optional[str], Optional[int]]:
    if not season_arg:
        return None, None

    s = normalize_season_input(season_arg)

    if s == "CURRENT":
        season, year = current_season_and_year()
        return season, (year_arg if year_arg is not None else year)

    if s == "NEXT":
        season, year = next_season_and_year()
        return season, (year_arg if year_arg is not None else year)

    if year_arg is None:
        _, current_year = current_season_and_year()
        year_arg = current_year
    return s, year_arg


# --------------------------
# Printing / formatting
# --------------------------

def pick_title(title_obj: Dict[str, Any]) -> str:
    return (
        title_obj.get("english")
        or title_obj.get("romaji")
        or title_obj.get("native")
        or "Unknown Title"
    )



def cmd_trending(args: argparse.Namespace) -> None:
    season, season_year = compute_season_vars(args.season, args.year)

    data = query_with_args(
        TRENDING_QUERY,
        {"page": 1, "perPage": args.limit, "season": season, "seasonYear": season_year},
        args,
    )
    items = data["Page"]["media"] or []
    if not items:
        console.print("[dim]No results.[/dim]")
        return

    label = "Trending anime"
    if season and season_year:
        label += f" - {season} {season_year}"

    console.print(f"[bold]{label}[/bold]\n")
    print_search_results(items)



def cmd_popular(args: argparse.Namespace) -> None:
    season, season_year = compute_season_vars(args.season, args.year)

    data = query_with_args(
        POPULAR_QUERY,
        {"page": 1, "perPage": args.limit, "season": season, "seasonYear": season_year},
        args,
    )
    items = data["Page"]["media"] or []
    if not items:
        console.print("[dim]No results.[/dim]")
        return

    label = "Popular anime"
    if season and season_year:
        label += f" - {season} {season_year}"

    console.print(f"[bold]{label}[/bold]\n")
    print_search_results(items)



def print_search_results(items: List[Dict[str, Any]]) -> None:
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
        genres_txt = ", ".join(genres[:4]) + ("…" if len(genres) > 4 else "")

        table.add_row(str(m["id"]), title, meta, genres_txt)

    console.print(table)



def print_info(media: Dict[str, Any]) -> None:
    title = pick_title(media["title"])
    romaji = media["title"].get("romaji") or ""
    native = media["title"].get("native") or ""

    header = Text()
    header.append(title, style="bold")
    header.append(f"  (AniList ID: {media['id']})", style="dim")

    console.print()
    console.print(header)

    if romaji and romaji != title:
        console.print(f"[dim]Romaji:[/dim] {romaji}")
    if native and native not in (title, romaji):
        console.print(f"[dim]Native:[/dim] {native}")

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

    if media.get("genres"):
        console.print(f"[dim]Genres:[/dim] {', '.join(media['genres'])}")

    console.print("[dim]Dubbed:[/dim] Unknown (depends on streaming service/region)")

    desc = strip_html(media.get("description") or "")
    if desc:
        console.print(Panel(desc, title="Description", box=box.ROUNDED))

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

    tr = media.get("trailer")
    if tr and tr.get("site") and tr.get("id"):
        trailer_url = trailer_to_url(tr["site"], tr["id"])
        if trailer_url:
            console.print(f"[bold]Trailer:[/bold] {trailer_url}")
        else:
            console.print("[bold]Trailer:[/bold] (unsupported trailer provider)")
    else:
        console.print("[bold]Trailer:[/bold] (none listed)")



def trailer_to_url(site: str, vid: str) -> str:
    site_l = (site or "").strip().lower()
    trailer_id = (vid or "").strip()

    if not trailer_id:
        return ""
    if site_l == "youtube":
        return f"https://www.youtube.com/watch?v={trailer_id}"
    if site_l == "dailymotion":
        return f"https://www.dailymotion.com/video/{trailer_id}"
    if site_l == "vimeo":
        return f"https://vimeo.com/{trailer_id}"
    if is_safe_external_url(trailer_id):
        return trailer_id
    return ""



def is_safe_external_url(url: str) -> bool:
    """Allow only plain HTTP(S) URLs before opening external programs."""
    parsed = urlparse((url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ------------------------------
# CLI actions
# ------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    wanted = parse_genre_inputs(args.genre, args.genres)
    validate_search_args(args.query, wanted)
    per_page = max(25, min(args.fetch, 50))

    matches: list[dict] = []
    page = 1

    while len(matches) < args.limit and page <= args.max_pages:
        data = query_with_args(
            SEARCH_QUERY,
            {"search": args.query, "page": page, "perPage": per_page},
            args,
        )

        items = data["Page"]["media"] or []
        if not items:
            break

        if wanted:
            items = [m for m in items if matches_genres(m.get("genres"), wanted, args.match)]

        matches.extend(items)
        page += 1

    matches = matches[: args.limit]

    if not matches:
        if wanted:
            console.print(f"[dim]No results.[/dim] (Filters: {args.match} of {', '.join(wanted)})")
        else:
            console.print("[dim]No results.[/dim]")
        return

    if wanted:
        console.print(f"[dim]Filters:[/dim] {args.match} of {', '.join(wanted)}")
        console.print(f"[dim]Scanned up to {page - 1} page(s). Showing {len(matches)} result(s).[/dim]\n")

    print_search_results(matches)



def cmd_info(args: argparse.Namespace) -> None:
    data = query_with_args(INFO_QUERY, {"id": args.id}, args)
    media = data["Media"]
    if media is None:
        raise RuntimeError(f"No anime found for AniList ID {args.id}.")
    print_info(media)



def cmd_trailer(args: argparse.Namespace) -> None:
    data = query_with_args(INFO_QUERY, {"id": args.id}, args)
    media = data["Media"]
    if media is None:
        raise RuntimeError(f"No anime found for AniList ID {args.id}.")

    tr = media.get("trailer")
    if not tr or not tr.get("site") or not tr.get("id"):
        console.print("[dim]No trailer listed for this anime.[/dim]")
        return

    url = trailer_to_url(tr["site"], tr["id"])
    if not url:
        raise RuntimeError("Trailer provider is unsupported or missing a usable URL.")

    if args.open:
        if not is_safe_external_url(url):
            raise RuntimeError("Trailer URL is not a safe HTTP(S) URL.")
        webbrowser.open(url)
        console.print(f"Opened trailer: {url}")
        return

    if args.mpv:
        if not is_safe_external_url(url):
            raise RuntimeError("Trailer URL is not a safe HTTP(S) URL.")
        try:
            subprocess.run(["mpv", url], check=True)
        except FileNotFoundError:
            raise RuntimeError("mpv not found. Install it or use --open instead.")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("mpv failed to play the trailer URL. Try --open instead.") from exc
        return

    console.print(url)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anime", description="Anime search and info CLI using AniList")

    p.add_argument("--no-cache", action="store_true", help="Disable cache")
    p.add_argument(
        "--cache-ttl",
        type=non_negative_int,
        default=60 * 60 * 24,
        help="Cache TTL in seconds (default: 86400)",
    )
    p.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory (default: ~/.cache/anime-scout)",
    )
    p.add_argument(
        "--rate",
        type=positive_float,
        default=MIN_INTERVAL_SECONDS,
        help="Minimum seconds between API requests (default: 1.0)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="Search anime by title and or genre")
    ps.add_argument("--max-pages", type=positive_int, default=10, help="Max pages to scan for genre matches")
    ps.add_argument("query", nargs="?", default=None, help="Anime title")
    ps.add_argument("--genre", action="append", help="Genre filter, repeatable")
    ps.add_argument("--genres", help="Comma-separated genres")
    ps.add_argument("--match", choices=["any", "all"], default="any", help="Genre match mode")
    ps.add_argument("--limit", type=positive_int, default=10, help="Max results to show")
    ps.add_argument("--fetch", type=positive_int, default=50, help="How many results to fetch before filtering")
    ps.set_defaults(func=cmd_search)

    ptr = sub.add_parser("trending", help="Show trending anime")
    ptr.add_argument("--limit", type=positive_int, default=10, help="Max results")
    ptr.add_argument("--season", help="Filter by season: current, next, WINTER, SPRING, SUMMER, FALL")
    ptr.add_argument("--year", type=int, help="Season year")
    ptr.set_defaults(func=cmd_trending)

    pp = sub.add_parser("popular", help="Show popular anime")
    pp.add_argument("--limit", type=positive_int, default=10, help="Max results")
    pp.add_argument("--season", help="Filter by season: current, next, WINTER, SPRING, SUMMER, FALL")
    pp.add_argument("--year", type=int, help="Season year")
    pp.set_defaults(func=cmd_popular)

    pi = sub.add_parser("info", help="Show detailed info for an AniList anime ID")
    pi.add_argument("id", type=int, help="AniList anime ID")
    pi.set_defaults(func=cmd_info)

    pt = sub.add_parser("trailer", help="Show, open, or play trailer for an AniList anime ID")
    pt.add_argument("id", type=int, help="AniList anime ID")
    pt.add_argument("--open", action="store_true", help="Open trailer in web browser")
    pt.add_argument("--mpv", action="store_true", help="Play trailer locally with mpv")
    pt.set_defaults(func=cmd_trailer)

    return p



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("[red]Interrupted.[/red]")
        raise SystemExit(130)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
