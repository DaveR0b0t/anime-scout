import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_scout.cli import (  # noqa: E402
    compute_season_vars,
    current_season_and_year,
    matches_genres,
    next_season_and_year,
    normalize_season_input,
    parse_genre_inputs,
    strip_html,
    trailer_to_url,
    validate_search_args,
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

    def test_strip_html_removes_tags_and_extra_space(self):
        self.assertEqual(strip_html("<b>Hello</b>   world<br>"), "Hello world")

    def test_matches_genres_any_mode(self):
        self.assertTrue(matches_genres(["Action", "Drama"], ["drama", "comedy"], "any"))

    def test_matches_genres_all_mode(self):
        self.assertTrue(matches_genres(["Action", "Drama"], ["action", "drama"], "all"))
        self.assertFalse(matches_genres(["Action", "Drama"], ["action", "comedy"], "all"))

    def test_validate_search_args_accepts_query(self):
        validate_search_args("naruto", [])

    def test_validate_search_args_accepts_genres(self):
        validate_search_args(None, ["action"])

    def test_validate_search_args_requires_query_or_genre(self):
        with self.assertRaises(ValueError):
            validate_search_args(None, [])

    def test_normalize_season_input_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            normalize_season_input("holiday")


if __name__ == "__main__":
    unittest.main()
