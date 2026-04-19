import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_scout.cli import (  # noqa: E402
    compute_season_vars,
    current_season_and_year,
    next_season_and_year,
    parse_genre_inputs,
    trailer_to_url,
)


class AnimeScoutHelperTests(unittest.TestCase):
    def test_parse_genre_inputs_merges_and_deduplicates(self):
        result = parse_genre_inputs(["Action", "Comedy"], "Comedy, Drama")
        self.assertEqual(result, ["action", "comedy", "drama"])

    def test_current_season_and_year(self):
        self.assertEqual(current_season_and_year(date(2026, 4, 19)), ("SPRING", 2026))

    def test_next_season_and_year_wraps_year(self):
        self.assertEqual(next_season_and_year(date(2026, 11, 1)), ("WINTER", 2027))

    def test_compute_season_vars_current(self):
        season, year = compute_season_vars("current", None)
        self.assertIn(season, {"WINTER", "SPRING", "SUMMER", "FALL"})
        self.assertIsInstance(year, int)

    def test_trailer_to_url_for_youtube(self):
        self.assertEqual(
            trailer_to_url("youtube", "abc123"),
            "https://www.youtube.com/watch?v=abc123",
        )


if __name__ == "__main__":
    unittest.main()
