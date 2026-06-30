"""Tests for audio_stream_saver.py."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from unmute.audio_stream_saver import AudioStreamSaver


class TestAudioStreamSaver:
    def test_add_chunks(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=1.0,
            output_path=output,
            max_saves=1,
        )

        # Add chunks that total less than 1 second
        chunk = np.random.randn(1000).astype(np.float32)  # ~0.04s at 24kHz
        saver.add(chunk)
        # Should not have saved yet

    def test_saves_when_interval_reached(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=0.01,  # Very short interval
            output_path=output,
            max_saves=1,
        )

        # Add enough samples to trigger a save (10ms = 240 samples at 24kHz)
        chunk = np.random.randn(300).astype(np.float32)
        saver.add(chunk)

        assert output.exists()

    def test_max_saves(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=0.001,
            output_path=output,
            max_saves=1,
        )

        # Add multiple chunks
        for _ in range(5):
            chunk = np.random.randn(300).astype(np.float32)
            saver.add(chunk)

        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 1

    def test_max_saves_multiple(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=0.001,
            output_path=output,
            max_saves=3,
        )

        for _ in range(10):
            chunk = np.random.randn(300).astype(np.float32)
            saver.add(chunk)

        wav_files = list(tmp_path.glob("*.wav"))
        assert len(wav_files) == 3

    def test_float32_assertion(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=1.0,
            output_path=output,
        )

        # Should raise for non-float32
        with pytest.raises(AssertionError):
            chunk = np.random.randn(100).astype(np.int16)
            saver.add(chunk)

    def test_1d_assertion(self, tmp_path: Path):
        output = tmp_path / "output.wav"
        saver = AudioStreamSaver(
            interval_sec=1.0,
            output_path=output,
        )

        # Should raise for 2D arrays
        with pytest.raises(AssertionError):
            chunk = np.random.randn(2, 100).astype(np.float32)
            saver.add(chunk)

    def test_default_output_path(self, tmp_path: Path):
        saver = AudioStreamSaver(
            interval_sec=1.0,
            max_saves=1,
        )
        # Default path should be under debug/
        assert "debug" in str(saver.output_path) or "out.wav" in str(saver.output_path)
