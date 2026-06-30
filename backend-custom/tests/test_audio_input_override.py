"""Tests for audio_input_override.py."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from unmute.audio_input_override import AudioInputOverride


class TestAudioInputOverride:
    def test_override(self, tmp_path: Path):
        # Create a fake audio file
        fake_data = np.arange(0, 9600, dtype=np.int16).reshape(2, 4800)

        with patch("unmute.audio_input_override.sphn") as mock_sphn:
            mock_sphn.read.return_value = (fake_data, 24000)

            override = AudioInputOverride(tmp_path / "fake.wav")

            # Create original data
            original = np.zeros((2, 480), dtype=np.int16)
            result = override.override(original)

            # Result should come from the override file
            assert result.shape == original.shape
            assert not np.array_equal(result, original)

    def test_override_exhausted_returns_original(self, tmp_path: Path):
        # Create a small fake audio file (only 480 samples)
        fake_data = np.arange(0, 480, dtype=np.int16).reshape(2, 240)

        with patch("unmute.audio_input_override.sphn") as mock_sphn:
            mock_sphn.read.return_value = (fake_data, 24000)

            override = AudioInputOverride(tmp_path / "fake.wav")

            # First call with 480 samples (file has 240 per channel, so it's exhausted)
            original = np.zeros((2, 480), dtype=np.int16)
            result = override.override(original)

            # When the file is shorter than the request, return original
            assert np.array_equal(result, original)

    def test_float32_conversion(self, tmp_path: Path):
        # Test that float32 data is converted to int16
        fake_data = np.random.randn(2, 4800).astype(np.float32)

        with patch("unmute.audio_input_override.sphn") as mock_sphn:
            mock_sphn.read.return_value = (fake_data, 24000)

            override = AudioInputOverride(tmp_path / "fake.wav")
            # Should have been converted to int16
            assert override.data.dtype == np.int16
