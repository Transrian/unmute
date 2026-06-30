"""Extended tests for stt/exponential_moving_average.py."""

import pytest


class TestExponentialMovingAverageUpdate:
    def test_update_attack(self):
        """When new_value > current value, use attack_time."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=0.0)
        result = ema.update(dt=0.1, new_value=1.0)
        # With attack_time=0.1, after 0.1s we should reach ~50% of the way to 1.0
        assert 0.4 < result < 0.6

    def test_update_release(self):
        """When new_value < current value, use release_time."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=1.0)
        result = ema.update(dt=0.1, new_value=0.0)
        # With release_time=1.0, after 0.1s we should be close to 1.0 (slow decay)
        assert 0.9 < result < 1.0

    def test_update_same_value(self):
        """When new_value == current value, no change."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=0.5)
        result = ema.update(dt=0.1, new_value=0.5)
        assert result == pytest.approx(0.5)

    def test_update_small_dt(self):
        """Small dt should produce minimal change."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=0.0)
        result = ema.update(dt=0.001, new_value=1.0)
        # Very small dt means almost no change
        assert 0.0 < result < 0.1

    def test_update_large_dt_attack(self):
        """Large dt with attack should approach target quickly."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=0.0)
        result = ema.update(dt=1.0, new_value=1.0)
        # Large dt should approach target
        assert result > 0.9

    def test_update_zero_dt_raises(self):
        """dt=0 should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError, match="dt must be positive"):
            ema.update(dt=0.0, new_value=1.0)

    def test_update_negative_dt_raises(self):
        """Negative dt should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError, match="dt must be positive"):
            ema.update(dt=-0.1, new_value=1.0)

    def test_update_negative_new_value_raises(self):
        """Negative new_value should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError, match="new_value must be non-negative"):
            ema.update(dt=0.1, new_value=-0.1)

    def test_multiple_updates_approach_target(self):
        """Repeated updates should gradually approach the target value."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=0.1, initial_value=0.0)
        values = []
        for _ in range(10):
            val = ema.update(dt=0.1, new_value=1.0)
            values.append(val)
        # Should be monotonically increasing
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]
        # Last value should be close to 1.0
        assert values[-1] > 0.95


class TestExponentialMovingAverageTimeToDecay:
    def test_time_to_decay_to_0_5(self):
        """Time to decay to 0.5 should equal release_time."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        time = ema.time_to_decay_to(0.5)
        assert time == pytest.approx(1.0)

    def test_time_to_decay_to_small_value(self):
        """Time to decay to a small value should be longer."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        time = ema.time_to_decay_to(0.01)
        assert time > 1.0

    def test_time_to_decay_to_close_to_1(self):
        """Time to decay to value close to 1 should be shorter."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        time = ema.time_to_decay_to(0.9)
        assert time < 1.0

    def test_time_to_decay_value_out_of_range_zero(self):
        """Value of 0 should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError):
            ema.time_to_decay_to(0.0)

    def test_time_to_decay_value_out_of_range_one(self):
        """Value of 1 should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError):
            ema.time_to_decay_to(1.0)

    def test_time_to_decay_value_negative(self):
        """Negative value should raise assertion error."""
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        with pytest.raises(AssertionError):
            ema.time_to_decay_to(-0.5)


class TestExponentialMovingAverageInit:
    def test_default_initial_value(self):
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0)
        assert ema.value == 0.0

    def test_custom_initial_value(self):
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.1, release_time=1.0, initial_value=0.75)
        assert ema.value == 0.75

    def test_attributes_stored(self):
        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        ema = ExponentialMovingAverage(attack_time=0.5, release_time=2.0)
        assert ema.attack_time == 0.5
        assert ema.release_time == 2.0
