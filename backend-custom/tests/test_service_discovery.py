"""Tests for service_discovery.py."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unmute.exceptions import MissingServiceAtCapacity, MissingServiceTimeout
from unmute.service_discovery import async_ttl_cached, get_instances, find_instance


class TestAsyncTtlCached:
    @pytest.mark.asyncio
    async def test_cache_hits(self):
        from functools import partial

        call_count = 0

        @partial(async_ttl_cached, ttl_sec=1.0)
        async def my_func(key):
            nonlocal call_count
            call_count += 1
            return f"result_{key}"

        result1 = await my_func("a")
        await asyncio.sleep(0)
        result2 = await my_func("a")
        assert result1 == "result_a"
        assert result2 == "result_a"

    @pytest.mark.asyncio
    async def test_cache_miss_different_keys(self):
        from functools import partial

        call_count = 0

        @partial(async_ttl_cached, ttl_sec=1.0)
        async def my_func(key):
            nonlocal call_count
            call_count += 1
            return f"result_{key}"

        result_a = await my_func("a")
        result_b = await my_func("b")
        assert result_a == "result_a"
        assert result_b == "result_b"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        from functools import partial

        call_count = 0

        @partial(async_ttl_cached, ttl_sec=0.01)
        async def my_func(key):
            nonlocal call_count
            call_count += 1
            return f"result_{key}_{call_count}"

        result1 = await my_func("a")
        await asyncio.sleep(0.05)
        result2 = await my_func("a")
        assert call_count == 2
        assert result1 != result2


class TestGetInstances:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        with patch("unmute.service_discovery.SERVICES") as mock_services:
            mock_services.__getitem__ = lambda self, key: "ws://tts:8080"

            async def fake_resolve(hostname):
                return ["10.0.0.1", "10.0.0.2"]

            with patch("unmute.service_discovery._resolve", new=fake_resolve):
                instances = await get_instances("tts")
                assert len(instances) == 2
                for inst in instances:
                    assert inst.startswith("ws://")
                    assert ":8080" in inst


class TestFindInstance:
    @pytest.mark.asyncio
    async def test_successful_connection(self):
        fake_client = MagicMock()
        fake_client.start_up = AsyncMock()

        def factory(instance):
            return fake_client

        async def fake_get_instances(name):
            return ["ws://instance1:8080"]

        with (
            patch("unmute.service_discovery.get_instances", new=fake_get_instances),
        ):
            result = await find_instance("tts", factory)
            assert result is fake_client
            fake_client.start_up.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        def factory(instance):
            nonlocal call_count
            client = MagicMock()

            async def start_up():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("failed")

            client.start_up = start_up
            return client

        async def fake_get_instances(name):
            return ["ws://i1:8080", "ws://i2:8080", "ws://i3:8080"]

        with patch("unmute.service_discovery.get_instances", new=fake_get_instances):
            result = await find_instance("tts", factory, max_trials=3)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_at_capacity(self):
        def factory(instance):
            client = MagicMock()

            async def start_up():
                raise MissingServiceAtCapacity("tts")

            client.start_up = start_up
            return client

        async def fake_get_instances(name):
            return ["ws://i1:8080"]

        with patch("unmute.service_discovery.get_instances", new=fake_get_instances):
            with pytest.raises(MissingServiceAtCapacity):
                await find_instance("tts", factory, max_trials=1)

    @pytest.mark.asyncio
    async def test_raises_timeout(self):
        def factory(instance):
            client = MagicMock()

            async def start_up():
                raise TimeoutError("timed out")

            client.start_up = start_up
            return client

        async def fake_get_instances(name):
            return ["ws://i1:8080"]

        with patch("unmute.service_discovery.get_instances", new=fake_get_instances):
            with pytest.raises(MissingServiceTimeout):
                await find_instance("tts", factory, max_trials=1)
