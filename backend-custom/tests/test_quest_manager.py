"""Tests for quest_manager.py: Quest, QuestManager."""

import asyncio

import pytest

from unmute.quest_manager import Quest, QuestManager


class TestQuest:
    @pytest.mark.asyncio
    async def test_quest_from_run_step(self):
        run_called = asyncio.Event()

        async def run_task():
            run_called.set()

        quest = Quest.from_run_step("test", run_task)
        assert quest.name == "test"
        assert quest.close is None

        # Run the quest
        async with quest:
            await asyncio.sleep(0.01)
            await run_called.wait()

    @pytest.mark.asyncio
    async def test_quest_with_init(self):
        async def _init():
            return 42

        init_result = None

        async def _run(data):
            nonlocal init_result
            init_result = data

        quest = Quest("test", _init, _run)
        async with quest:
            await asyncio.sleep(0.01)

        assert init_result == 42

    @pytest.mark.asyncio
    async def test_quest_get(self):
        async def _init():
            return "init_data"

        async def _run(data):
            await asyncio.sleep(0.1)

        quest = Quest("test", _init, _run)
        async with quest:
            data = await quest.get()
            assert data == "init_data"

    @pytest.mark.asyncio
    async def test_quest_get_nowait(self):
        async def _init():
            return "data"

        async def _run(data):
            await asyncio.sleep(0.1)

        quest = Quest("test", _init, _run)
        # Before entering context, _data is not done
        assert quest.get_nowait() is None

        async with quest:
            await asyncio.sleep(0.01)
            assert quest.get_nowait() == "data"

    @pytest.mark.asyncio
    async def test_quest_init_exception(self):
        async def _init():
            raise ValueError("init failed")

        async def _run(data):
            pass

        quest = Quest("test", _init, _run)
        with pytest.raises(ValueError, match="init failed"):
            async with quest:
                await quest.get()

    @pytest.mark.asyncio
    async def test_quest_with_close(self):
        close_called = False

        async def _init():
            return "data"

        async def _run(data):
            await asyncio.sleep(0.01)

        async def _close(data):
            nonlocal close_called
            close_called = True

        quest = Quest("test", _init, _run, _close)
        async with quest:
            await asyncio.sleep(0.05)

        assert close_called is True


class TestQuestManager:
    @pytest.mark.asyncio
    async def test_quest_manager_context(self):
        async with QuestManager() as qm:
            assert qm.quests == {}

    @pytest.mark.asyncio
    async def test_add_quest(self):
        run_completed = asyncio.Event()

        async def _init():
            return None

        async def _run(data):
            run_completed.set()
            await asyncio.sleep(0.05)

        async with QuestManager() as qm:
            quest = Quest("test", _init, _run)
            await qm.add(quest)
            assert "test" in qm.quests
            await run_completed.wait()

    @pytest.mark.asyncio
    async def test_remove_quest(self):
        async def _init():
            return None

        async def _run(data):
            await asyncio.sleep(1.0)

        async with QuestManager() as qm:
            quest = Quest("test", _init, _run)
            await qm.add(quest)
            await qm.remove("test")
            assert "test" not in qm.quests

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        async with QuestManager() as qm:
            # Should not raise
            await qm.remove("nonexistent")

    @pytest.mark.asyncio
    async def test_add_replaces_existing(self):
        run_count = 0

        async def _init():
            return None

        async def _run(data):
            nonlocal run_count
            run_count += 1
            await asyncio.sleep(0.05)

        async with QuestManager() as qm:
            await qm.add(Quest("test", _init, _run))
            await asyncio.sleep(0.01)
            # Add a new quest with the same name
            await qm.add(Quest("test", _init, _run))
            await asyncio.sleep(0.1)

        assert run_count >= 1

    @pytest.mark.asyncio
    async def test_wait_propagates_exception(self):
        async def _init():
            return None

        async def _run(data):
            raise RuntimeError("quest error")

        with pytest.raises(RuntimeError, match="quest error"):
            async with QuestManager() as qm:
                await qm.add(Quest("test", _init, _run))
                await qm.wait()

    @pytest.mark.asyncio
    async def test_cleanup_on_exit(self):
        close_called = False
        keep_running = asyncio.Event()

        async def _init():
            return "data"

        async def _run(data):
            await keep_running.wait()

        async def _close(data):
            nonlocal close_called
            close_called = True

        async with QuestManager() as qm:
            await qm.add(Quest("test", _init, _run, _close))
            await asyncio.sleep(0.01)

        # After exiting context, close should have been called
        # Set the event so the run task can finish
        keep_running.set()
        await asyncio.sleep(0.05)
        assert close_called is True

    @pytest.mark.asyncio
    async def test_multiple_quests(self):
        results = []
        keep_running = asyncio.Event()

        async def init_a():
            return "a"

        async def init_b():
            return "b"

        async def run_a(data):
            results.append(data)
            await keep_running.wait()

        async def run_b(data):
            results.append(data)
            await keep_running.wait()

        async with QuestManager() as qm:
            await qm.add(Quest("a", init_a, run_a))
            await qm.add(Quest("b", init_b, run_b))
            await asyncio.sleep(0.05)

        keep_running.set()
        await asyncio.sleep(0.01)
        assert "a" in results
        assert "b" in results
