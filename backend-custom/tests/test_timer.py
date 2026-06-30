"""Tests for timer.py: Stopwatch, PhasesStopwatch."""

import asyncio

import pytest

from unmute.timer import PhasesStopwatch, Stopwatch


class TestStopwatch:
    def test_stopwatch_autostart(self):
        sw = Stopwatch(autostart=True)
        assert sw.started is True
        assert sw.time() >= 0.0

    def test_stopwatch_no_autostart(self):
        sw = Stopwatch(autostart=False)
        assert sw.started is False
        with pytest.raises(RuntimeError, match="not started"):
            sw.time()

    def test_stopwatch_start_if_not_started(self):
        sw = Stopwatch(autostart=False)
        sw.start_if_not_started()
        assert sw.started is True
        assert sw.time() >= 0.0

    def test_stopwatch_start_if_not_started_idempotent(self):
        sw = Stopwatch(autostart=True)
        original_time = sw.time()
        sw.start_if_not_started()
        # Should not reset the start time
        assert sw.time() >= original_time

    def test_stopwatch_stop(self):
        sw = Stopwatch(autostart=True)
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.01))
        elapsed = sw.stop()
        assert elapsed is not None
        assert elapsed > 0

    def test_stopwatch_stop_twice(self):
        sw = Stopwatch(autostart=True)
        sw.stop()
        assert sw.stop() is None  # Already stopped

    def test_stopwatch_not_started_stop(self):
        sw = Stopwatch(autostart=False)
        assert sw.stop() is None


class TestPhasesStopwatch:
    def test_init(self):
        phases = PhasesStopwatch(["start", "middle", "end"])
        assert phases.phases == ["start", "middle", "end"]
        assert phases.times == [None, None, None]

    def test_time_phase(self):
        phases = PhasesStopwatch(["a", "b", "c"])
        phases.time_phase_if_not_started("a")
        assert phases.times[0] is not None

    def test_time_phase_twice_keeps_first(self):
        phases = PhasesStopwatch(["a", "b"])
        phases.time_phase_if_not_started("a", t=10.0)
        phases.time_phase_if_not_started("a", t=20.0)
        assert phases.times[0] == 10.0

    def test_time_phase_check_previous(self):
        phases = PhasesStopwatch(["a", "b", "c"])
        # Should raise if "a" is not done
        with pytest.raises(RuntimeError, match="hasn't started"):
            phases.time_phase_if_not_started("b", check_previous=True)

    def test_time_phase_no_check_previous(self):
        phases = PhasesStopwatch(["a", "b", "c"])
        # Should not raise
        phases.time_phase_if_not_started("b", check_previous=False)
        assert phases.times[1] is not None

    def test_get_phase_index(self):
        phases = PhasesStopwatch(["a", "b", "c"])
        assert phases.get_phase_index("a") == 0
        assert phases.get_phase_index("b") == 1
        assert phases.get_phase_index("c") == 2

    def test_get_phase_index_invalid(self):
        phases = PhasesStopwatch(["a", "b"])
        with pytest.raises(ValueError, match="not in phases"):
            phases.get_phase_index("z")

    def test_get_time_for_phase(self):
        phases = PhasesStopwatch(["a", "b"])
        phases.time_phase_if_not_started("a", t=5.0)
        assert phases.get_time_for_phase("a") == 5.0

    def test_get_time_for_phase_not_started(self):
        phases = PhasesStopwatch(["a", "b"])
        with pytest.raises(RuntimeError, match="not started"):
            phases.get_time_for_phase("a")

    def test_get_time_for_phase_invalid(self):
        phases = PhasesStopwatch(["a", "b"])
        with pytest.raises(ValueError, match="not in phases"):
            phases.get_time_for_phase("z")

    def test_phase_dict(self):
        phases = PhasesStopwatch(["a", "b"])
        phases.time_phase_if_not_started("a", t=1.0)
        phases.time_phase_if_not_started("b", t=2.0, check_previous=False)
        assert phases.phase_dict() == {"a": 1.0, "b": 2.0}

    def test_phase_dict_partial(self):
        phases = PhasesStopwatch(["a", "b", "c"])
        phases.time_phase_if_not_started("a", t=1.0)
        result = phases.phase_dict_partial()
        assert result == {"a": 1.0, "b": None, "c": None}

    def test_reset(self):
        phases = PhasesStopwatch(["a", "b"])
        phases.time_phase_if_not_started("a", t=1.0)
        phases.reset()
        assert phases.times == [None, None]
