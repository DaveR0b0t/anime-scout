📺 anime-scout

A fast, terminal-friendly anime discovery CLI powered by the AniList GraphQL API.

Search anime by title or genre, explore trending and popular shows, view detailed information, and watch trailers — all from your terminal.

✨ Features

🔍 Search anime by title and/or multiple genres (AND / OR matching)

📈 Trending & popular anime feeds

🗓️ Season filters (current, next, or specific seasons)

🧾 Detailed info view with cleaned descriptions

🌐 Official “where to watch” links

🎬 Trailer support

Print trailer URL

Open in browser (--open)

Play locally with mpv (--mpv)

⚡ Built-in caching & rate limiting

🎨 Rich terminal UI (tables, panels, clean formatting)

🚀 Installation
Recommended (via pipx)

pipx installs the CLI in an isolated environment and exposes it globally.
```
pipx install git+https://github.com/DaveR0b0t/anime-scout.git
```
Development install (editable)
```
git clone https://github.com/DaveR0b0t/anime-scout.git
cd anime-scout

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

🧠 Usage
🔎 Search
```
anime search "naruto" --limit 10
anime search --genre Action --genre Adventure --match all
anime search --genres "Slice of Life, Comedy" --match all --limit 20
```

ℹ️ Anime info
```
anime info 20
```

🎬 Trailers
```
anime trailer 20
anime trailer 20 --open
anime trailer 20 --mpv
```

📈 Trending & Popular
```
anime trending --limit 15
anime popular --limit 15
```

🗓️ Season filters
```
anime trending --season current
anime trending --season next
anime popular --season FALL --year 2024
```

⚙️ Cache & Rate Limiting

By default, responses are cached locally and API requests are rate-limited.

```
anime search "bleach" --no-cache
anime search "bleach" --cache-ttl 3600
anime search "bleach" --rate 2.0
anime search "bleach" --cache-dir ~/.cache/anime-scout
```

ℹ️ Notes

Dubbed availability is not reliably exposed by general anime metadata APIs and is currently shown as Unknown.

Streaming availability is provided via official external links from AniList and may vary by region.

🐍 Requirements

Python 3.10+

Optional: mpv (for local trailer playback)

📄 License

MIT License

🙌 Acknowledgements

Data provided by the AniList GraphQL API

Terminal UI powered by Rich
