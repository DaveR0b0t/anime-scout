# anime-scout

[![Tests](https://github.com/DaveR0b0t/anime-scout/actions/workflows/tests.yml/badge.svg)](https://github.com/DaveR0b0t/anime-scout/actions/workflows/tests.yml)

anime-scout is a terminal-based anime discovery CLI powered by the AniList GraphQL API.

You can search anime by title or genre, check trending and popular shows, view detailed information, and open or play trailers from your terminal.

## Features

- Search anime by title
- Filter results by one or more genres
- View trending anime
- View popular anime
- Filter trending and popular results by season and year
- Show detailed anime information with cleaned descriptions
- Display official external links
- Print, open, or play trailers
- Use built-in caching and rate limiting
- View results in a Rich-powered terminal interface

## Installation

### Option 1: Install with pipx

```bash
pipx install git+https://github.com/DaveR0b0t/anime-scout.git
```

### Option 2: Development install

```bash
git clone https://github.com/DaveR0b0t/anime-scout.git
cd anime-scout

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Search

```bash
anime search "naruto" --limit 10
anime search --genre Action --genre Adventure --match all
anime search --genres "Slice of Life, Comedy" --match all --limit 20
```

### Anime info

```bash
anime info 20
```

### Trailers

```bash
anime trailer 20
anime trailer 20 --open
anime trailer 20 --mpv
```

### Trending and popular

```bash
anime trending --limit 15
anime popular --limit 15
```

### Season filters

```bash
anime trending --season current
anime trending --season next
anime popular --season FALL --year 2024
```

### Cache and rate limiting

Responses are cached locally by default, and API requests are rate-limited.

```bash
anime search "bleach" --no-cache
anime search "bleach" --cache-ttl 3600
anime search "bleach" --rate 2.0
anime search "bleach" --cache-dir ~/.cache/anime-scout
```

## Notes

Dubbed availability is not reliably exposed by general anime metadata APIs, so it is currently shown as Unknown.

Streaming availability is based on official external links from AniList and may vary by region.

## Requirements

- Python 3.10+
- Optional: mpv for local trailer playback

## Running Tests

```bash
python -m unittest discover -s tests
```

## License

MIT License

## Acknowledgements

- Data provided by the AniList GraphQL API
- Terminal UI powered by Rich

Last updated to retrigger GitHub Actions.
