import argparse
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_scout.cli import (  # noqa: E402
    anilist_query,
    compute_season_vars,
    current_season_and_year,
    matches_genres,
    next_season_and_year,
    normalize_season_input,
    parse_genre_inputs,
    positive_float,
    positive_int,
    is_safe_external_url,
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


    def test_positive_int_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_positive_float_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_float("-0.1")

    def test_is_safe_external_url_allows_https(self):
        self.assertTrue(is_safe_external_url("https://www.youtube.com/watch?v=abc123"))

    def test_is_safe_external_url_blocks_non_http_schemes(self):
        self.assertFalse(is_safe_external_url("javascript:alert(1)"))


class AniListRequestTests(unittest.TestCase):
    @patch("anime_scout.cli.requests.post")
    def test_anilist_query_returns_data_on_success(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": {"Media": {"id": 20}}}
        mock_post.return_value = response

        result = anilist_query("query Test { Media(id: 20) { id } }", {"id": 20}, use_cache=False)

        self.assertEqual(result, {"Media": {"id": 20}})
        mock_post.assert_called_once()

    @patch("anime_scout.cli.requests.post")
    def test_anilist_query_raises_clean_error_for_api_errors(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"errors": [{"message": "Not found"}]}
        mock_post.return_value = response

        with self.assertRaises(RuntimeError) as ctx:
            anilist_query("query Test { Media(id: 999999) { id } }", {"id": 999999}, use_cache=False)

        self.assertIn("AniList API error: Not found", str(ctx.exception))

    @patch("anime_scout.cli.requests.post")
    def test_anilist_query_raises_clean_error_for_request_failures(self, mock_post):
        mock_post.side_effect = requests.RequestException("network down")

        with self.assertRaises(RuntimeError) as ctx:
            anilist_query("query Test { Media(id: 20) { id } }", {"id": 20}, use_cache=False)

        self.assertIn("Request to AniList failed", str(ctx.exception))

    @patch("anime_scout.cli.requests.post")
    def test_anilist_query_raises_clean_error_for_invalid_json(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")
        mock_post.return_value = response

        with self.assertRaises(RuntimeError) as ctx:
            anilist_query("query Test { Media(id: 20) { id } }", {"id": 20}, use_cache=False)

        self.assertIn("invalid JSON", str(ctx.exception))

    @patch("anime_scout.cli.requests.post")
    def test_anilist_query_raises_when_data_is_missing(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        mock_post.return_value = response

        with self.assertRaises(RuntimeError) as ctx:
            anilist_query("query Test { Media(id: 20) { id } }", {"id": 20}, use_cache=False)

        self.assertIn("returned no data", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
