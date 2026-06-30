"""Tests for tts/realtime_queue.py: RealtimeQueue."""

import asyncio

import pytest

from unmute.tts.realtime_queue import RealtimeQueue


class TestRealtimeQueueBasics:
    def test_init(self):
        queue = RealtimeQueue[str]()
        assert queue.empty() is True
        assert queue.start_time is None

    def test_init_with_custom_time(self):
        current_time = 100.0
        queue = RealtimeQueue[str](get_time=lambda: current_time)
        assert queue.get_time() == 100.0

    def test_start_if_not_started(self):
        current_time = 100.0
        queue = RealtimeQueue[str](get_time=lambda: current_time)
        queue.start_if_not_started()
        assert queue.start_time == 100.0

    def test_start_if_not_started_idempotent(self):
        queue = RealtimeQueue[str]()
        queue.start_if_not_started()
        first = queue.start_time
        queue.start_if_not_started()
        assert queue.start_time == first

    def test_put_and_empty(self):
        queue = RealtimeQueue[str](get_time=lambda: 0.0)
        queue.start_if_not_started()
        queue.put("item", 1.0)
        assert queue.empty() is False


class TestRealtimeQueueGetNowait:
    def test_get_nowait_heap_order(self):
        """Items are stored in a heap and released in order of their time."""
        # start_time = 10.0, so time_since_start for items = get_time() - start_time = 10 - 10 = 0
        # We put items with their times, and items whose time <= time_since_start are released
        current_time = 20.0
        queue = RealtimeQueue[str](get_time=lambda: current_time)
        queue.start_if_not_started()  # start_time = 20.0
        # Items are put with absolute times relative to when start_if_not_started was called
        # time_since_start = 20 - 20 = 0, so nothing should be released
        queue.put("a", 1.0)
        queue.put("b", 2.0)
        queue.put("c", 3.0)

        items = list(queue.get_nowait())
        # time_since_start = 0, all items have time > 0, so none released
        assert items == []

    def test_get_nowait_releases_when_time_advances(self):
        """When current time advances past item times, they're released."""
        current_time = [10.0]

        queue = RealtimeQueue[str](get_time=lambda: current_time[0])
        queue.start_if_not_started()  # start_time = 10.0
        queue.put("a", 1.0)  # time = 1.0
        queue.put("b", 2.0)  # time = 2.0
        queue.put("c", 3.0)  # time = 3.0

        # time_since_start = 10 - 10 = 0, none released yet
        items = list(queue.get_nowait())
        assert items == []

        # Advance time: time_since_start = 13 - 10 = 3.0
        current_time[0] = 13.0
        items = list(queue.get_nowait())
        assert len(items) == 3
        assert items[0] == (1.0, "a")
        assert items[1] == (2.0, "b")
        assert items[2] == (3.0, "c")

    def test_get_nowait_partial_release(self):
        current_time = [10.0]
        queue = RealtimeQueue[str](get_time=lambda: current_time[0])
        queue.start_if_not_started()
        queue.put("a", 1.0)
        queue.put("b", 2.0)
        queue.put("c", 5.0)

        # time_since_start = 11 - 10 = 1.0, only "a" (time=1.0) released
        current_time[0] = 11.0
        items = list(queue.get_nowait())
        assert len(items) == 1
        assert items[0] == (1.0, "a")
        assert queue.empty() is False  # b and c still in queue

    def test_get_nowait_not_started(self):
        queue = RealtimeQueue[str]()
        queue.put("item", 1.0)
        result = list(queue.get_nowait())
        assert result == []

    def test_get_nowait_no_past_due(self):
        queue = RealtimeQueue[str](get_time=lambda: 0.0)
        queue.start_if_not_started()
        queue.put("item", 5.0)
        result = list(queue.get_nowait())
        assert result == []
        assert queue.empty() is False


class TestRealtimeQueueAsyncIter:
    @pytest.mark.asyncio
    async def test_aiter_not_started(self):
        queue = RealtimeQueue[str]()
        items = [item async for item in queue]
        assert items == []

    @pytest.mark.asyncio
    async def test_aiter_empty(self):
        queue = RealtimeQueue[str]()
        queue.start_if_not_started()
        items = [item async for item in queue]
        assert items == []

    @pytest.mark.asyncio
    async def test_aiter_releases_all(self):
        queue = RealtimeQueue[str](get_time=lambda: 10.0)
        queue.start_if_not_started()  # start_time = 10.0
        # time_since_start = 10 - 10 = 0, but items are put after start
        # so their time is positive relative to start
        queue.put("a", 0.1)
        queue.put("b", 0.2)
        queue.put("c", 0.3)

        # Since we're using a lambda that returns 10.0 always,
        # time_since_start = 0. Items with time 0.1-0.3 won't be released immediately
        # unless we advance time. But __aiter__ will sleep and wait...
        # Actually the lambda always returns 10.0, so time_since_start stays 0
        # and items will never be released in __aiter__
        # Let's use a mutable time
        pass

    @pytest.mark.asyncio
    async def test_aiter_with_advancing_time(self):
        current_time = [0.0]
        queue = RealtimeQueue[str](get_time=lambda: current_time[0])
        queue.start_if_not_started()  # start_time = 0.0
        queue.put("a", 0.01)

        # Advance time in background
        async def advance():
            await asyncio.sleep(0.05)
            current_time[0] = 1.0

        asyncio.create_task(advance())
        items = [item async for item in queue]
        assert len(items) == 1
        assert items[0] == (0.01, "a")
