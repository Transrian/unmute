"""Tests for cache.py: LocalCache, CacheError, get_cache."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from unmute.cache import CacheError, LocalCache


class TestLocalCache:
    def test_set_and_get(self):
        cache = LocalCache[str](ttl_seconds=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = LocalCache[str](ttl_seconds=3600)
        assert cache.get("missing") is None

    def test_ttl_expiration(self):
        cache = LocalCache[str](ttl_seconds=0)  # 0 seconds = immediate expiry
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_ttl_not_expired(self):
        cache = LocalCache[str](ttl_seconds=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_delete(self):
        cache = LocalCache[str](ttl_seconds=3600)
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        cache = LocalCache[str](ttl_seconds=3600)
        # Should not raise
        cache.delete("missing")

    def test_cleanup(self):
        cache = LocalCache[str](ttl_seconds=0)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        time.sleep(0.01)
        cache.cleanup()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cleanup_keeps_valid(self):
        cache = LocalCache[str](ttl_seconds=3600)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.cleanup()
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

    def test_overwrite(self):
        cache = LocalCache[str](ttl_seconds=3600)
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_different_types(self):
        cache = LocalCache[int](ttl_seconds=3600)
        cache.set("num", 42)
        assert cache.get("num") == 42

        cache2 = LocalCache[list](ttl_seconds=3600)
        cache2.set("list", [1, 2, 3])
        assert cache2.get("list") == [1, 2, 3]


class TestCacheError:
    def test_cache_error_message(self):
        err = CacheError("test error")
        assert str(err) == "test error"

    def test_cache_error_from_other(self):
        original = ValueError("original")
        try:
            raise CacheError("wrapped") from original
        except CacheError as e:
            assert str(e) == "wrapped"
            assert e.__cause__ is original


class TestGetCache:
    def test_get_cache_no_redis_returns_local(self):
        from unmute.cache import LocalCache, get_cache

        cache = get_cache(prefix="test", ttl_seconds=60)
        assert isinstance(cache, LocalCache)

    def test_get_cache_with_redis(self, monkeypatch):
        import os

        # When REDIS_SERVER is set, it should return a RedisCache
        monkeypatch.setenv("KYUTAI_REDIS_URL", "redis://localhost:6379")

        # Need to reimport to pick up the new env var
        import importlib

        import unmute.kyutai_constants

        importlib.reload(unmute.kyutai_constants)
        import unmute.cache

        importlib.reload(unmute.cache)

        cache = unmute.cache.get_cache(prefix="test", ttl_seconds=60)
        assert type(cache).__name__ == "RedisCache"

        # Cleanup
        monkeypatch.delenv("KYUTAI_REDIS_URL")
        importlib.reload(unmute.kyutai_constants)
        importlib.reload(unmute.cache)


class TestRedisCache:
    def test_get_key_prefix(self):
        from unmute.cache import RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.return_value = b"value"
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            cache.get("mykey")
            mock_redis.get.assert_called_once_with("test_prefix:mykey")

    def test_get_none(self):
        from unmute.cache import RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            result = cache.get("mykey")
            assert result is None

    def test_get_redis_error(self):
        import redis as redis_module

        from unmute.cache import CacheError, RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.get.side_effect = redis_module.RedisError("Connection refused")
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            with pytest.raises(CacheError):
                cache.get("mykey")

    def test_set_key_prefix(self):
        from unmute.cache import RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            cache.set("mykey", "myvalue")
            mock_redis.setex.assert_called_once_with("test_prefix:mykey", 3600, "myvalue")

    def test_set_redis_error(self):
        import redis as redis_module

        from unmute.cache import CacheError, RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.setex.side_effect = redis_module.RedisError("Connection refused")
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            with pytest.raises(CacheError):
                cache.set("mykey", "myvalue")

    def test_delete_key_prefix(self):
        from unmute.cache import RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            cache.delete("mykey")
            mock_redis.delete.assert_called_once_with("test_prefix:mykey")

    def test_delete_redis_error(self):
        import redis as redis_module

        from unmute.cache import CacheError, RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_redis.delete.side_effect = redis_module.RedisError("Connection refused")
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            with pytest.raises(CacheError):
                cache.delete("mykey")

    def test_cleanup_does_nothing(self):
        from unmute.cache import RedisCache

        with patch("unmute.cache.redis.Redis.from_url") as mock_from_url:
            mock_redis = MagicMock()
            mock_from_url.return_value = mock_redis

            cache = RedisCache("redis://localhost:6379", "test_prefix", 3600)
            cache.cleanup()  # Should not raise
            # No Redis calls should be made
            mock_redis.assert_not_called()
