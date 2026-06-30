"""Tests for llm/newsapi.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from unmute.cache import CacheError


class TestNewsResponse:
    def test_create(self):
        from unmute.llm.newsapi import Article, NewsResponse, Source

        response = NewsResponse(
            status="ok",
            totalResults=1,
            articles=[
                Article(
                    source=Source(id="the-verge", name="The Verge"),
                    author="John Doe",
                    title="Test Article",
                    description="A test article",
                    publishedAt="2024-01-01T00:00:00Z",
                    content="Content here",
                )
            ],
        )
        assert response.status == "ok"
        assert len(response.articles) == 1
        assert response.articles[0].title == "Test Article"

    def test_minimal_article(self):
        from unmute.llm.newsapi import Article, NewsResponse, Source

        response = NewsResponse(
            status="ok",
            totalResults=1,
            articles=[
                Article(
                    source=Source(id=None, name="Unknown"),
                    author=None,
                    title="Test",
                    description=None,
                    publishedAt="2024-01-01T00:00:00Z",
                    content=None,
                )
            ],
        )
        assert response.articles[0].author is None


class TestGetNewsWithoutCaching:
    def test_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Need to reimport to pick up the empty env var
            import importlib

            import unmute.llm.newsapi as newsapi

            with patch.object(newsapi, "newsapi_api_key", None):
                result = newsapi.get_news_without_caching()
                assert result is None

    def test_success(self):
        with patch("unmute.llm.newsapi.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "ok",
                "totalResults": 1,
                "articles": [
                    {
                        "source": {"id": "the-verge", "name": "The Verge"},
                        "author": "John",
                        "title": "Test",
                        "description": "Desc",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "content": "Content",
                    }
                ],
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            import unmute.llm.newsapi as newsapi

            with patch.object(newsapi, "newsapi_api_key", "test-key"):
                result = newsapi.get_news_without_caching()
                assert result is not None
                assert result.status == "ok"
                assert len(result.articles) == 1

    def test_http_error(self):
        with patch("unmute.llm.newsapi.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("403 Forbidden")
            mock_get.return_value = mock_response

            import unmute.llm.newsapi as newsapi

            with patch.object(newsapi, "newsapi_api_key", "test-key"):
                with pytest.raises(Exception, match="403 Forbidden"):
                    newsapi.get_news_without_caching()


class TestGetNews:
    def test_cached_news(self):
        from unmute.llm.newsapi import Article, NewsResponse, Source

        cached_response = NewsResponse(
            status="ok",
            totalResults=1,
            articles=[
                Article(
                    source=Source(id="the-verge", name="The Verge"),
                    author="John",
                    title="Cached Article",
                    description="Desc",
                    publishedAt="2024-01-01T00:00:00Z",
                    content="Content",
                )
            ],
        )

        fake_cache = MagicMock()
        fake_cache.get.return_value = cached_response.model_dump_json()

        with patch("unmute.llm.newsapi.cache", fake_cache):
            import unmute.llm.newsapi as newsapi

            result = newsapi.get_news()
            assert result is not None
            assert result.articles[0].title == "Cached Article"

    def test_cache_miss_fetches_new(self):
        fake_cache = MagicMock()
        fake_cache.get.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "source": {"id": "the-verge", "name": "The Verge"},
                    "author": "Jane",
                    "title": "Fresh News",
                    "description": "Desc",
                    "publishedAt": "2024-06-01T00:00:00Z",
                    "content": "Content",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.llm.newsapi.cache", fake_cache),
            patch("unmute.llm.newsapi.requests.get", return_value=mock_response),
            patch.object(__import__("unmute.llm.newsapi", fromlist=["newsapi_api_key"]), "newsapi_api_key", "test-key"),
        ):
            import unmute.llm.newsapi as newsapi

            with patch.object(newsapi, "newsapi_api_key", "test-key"):
                result = newsapi.get_news()
                assert result is not None
                assert result.articles[0].title == "Fresh News"

    def test_cache_error(self):
        fake_cache = MagicMock()
        fake_cache.get.side_effect = CacheError("Cache unavailable")

        with patch("unmute.llm.newsapi.cache", fake_cache):
            import unmute.llm.newsapi as newsapi

            result = newsapi.get_news()
            assert result is None

    def test_fetch_error_returns_none(self):
        fake_cache = MagicMock()
        fake_cache.get.return_value = None

        with (
            patch("unmute.llm.newsapi.cache", fake_cache),
            patch("unmute.llm.newsapi.requests.get", side_effect=Exception("Connection error")),
            patch.object(__import__("unmute.llm.newsapi", fromlist=["newsapi_api_key"]), "newsapi_api_key", "test-key"),
        ):
            import unmute.llm.newsapi as newsapi

            with patch.object(newsapi, "newsapi_api_key", "test-key"):
                result = newsapi.get_news()
                assert result is None
