"""Tests for audio_codec.py: decode/encode helpers for /v1/audio endpoints."""

import io
import wave

import numpy as np
import pytest

from unmute.audio_codec import (
    decode_audio_to_pcm,
    encode_pcm_to_audio,
    guess_audio_format,
)


# ─── Helpers ────────────────────────────────────────────────────────


def make_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a WAV file with a single sine wave."""
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate
    samples = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
    int16 = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


# ─── guess_audio_format ─────────────────────────────────────────────


class TestGuessAudioFormat:
    def test_known_extensions(self):
        assert guess_audio_format("audio.wav") == "wav"
        assert guess_audio_format("audio.mp3") == "mp3"
        assert guess_audio_format("audio.flac") == "flac"
        assert guess_audio_format("audio.m4a") == "ipod"
        assert guess_audio_format("audio.ogg") == "ogg"
        assert guess_audio_format("audio.webm") == "webm"
        assert guess_audio_format("audio.mp4") == "mp4"

    def test_case_insensitive(self):
        assert guess_audio_format("Audio.WAV") == "wav"
        assert guess_audio_format("audio.Mp3") == "mp3"

    def test_unknown_extension(self):
        assert guess_audio_format("audio.xyz") is None
        assert guess_audio_format("noextension") is None


# ─── decode_audio_to_pcm ────────────────────────────────────────────


class TestDecodeAudioToPcm:
    def test_decode_wav_441khz_to_24khz(self):
        """Decode a 44.1kHz WAV and resample to 24kHz."""
        data = make_wav_bytes(duration_sec=1.0, sample_rate=44100)
        pcm = decode_audio_to_pcm(data, "audio.wav", target_sample_rate=24000)

        assert pcm.dtype == np.float32
        assert len(pcm) == pytest.approx(24000, abs=200)  # ~1 second at 24kHz
        assert pcm.min() >= -1.0
        assert pcm.max() <= 1.0

    def test_decode_wav_already_24khz(self):
        """Decode a 24kHz WAV (no resampling needed)."""
        data = make_wav_bytes(duration_sec=0.5, sample_rate=24000)
        pcm = decode_audio_to_pcm(data, "audio.wav", target_sample_rate=24000)

        assert pcm.dtype == np.float32
        assert len(pcm) == pytest.approx(12000, abs=100)  # ~0.5s at 24kHz

    def test_amplitude_preserved(self):
        """Check that amplitude is roughly preserved after decode."""
        data = make_wav_bytes(amplitude=0.5)
        pcm = decode_audio_to_pcm(data, "audio.wav")

        # Peak should be close to 0.5 (with small tolerance for resampling)
        assert abs(pcm.max() - 0.5) < 0.05

    def test_decode_mp3(self):
        """Encode to mp3 then decode back."""
        pcm_original = np.sin(2 * np.pi * 440 * np.arange(48000) / 24000).astype(
            np.float32
        ) * 0.3
        mp3_bytes, _ = encode_pcm_to_audio(pcm_original, "mp3", 24000)
        pcm_decoded = decode_audio_to_pcm(mp3_bytes, "audio.mp3")

        assert pcm_decoded.dtype == np.float32
        # MP3 encoding uses frames; duration may be slightly shorter
        assert len(pcm_decoded) == pytest.approx(48000, abs=5000)

    def test_unknown_format_raises(self):
        """Unknown format should raise an error."""
        with pytest.raises(Exception):
            decode_audio_to_pcm(b"not audio data", "audio.xyz")


# ─── encode_pcm_to_audio ───────────────────────────────────────────


class TestEncodePcmToAudio:
    def test_encode_to_mp3(self):
        pcm = np.random.randn(24000).astype(np.float32) * 0.1
        data, mime = encode_pcm_to_audio(pcm, "mp3", 24000)

        assert len(data) > 0
        assert mime == "audio/mpeg"
        # MP3 should be much smaller than raw PCM
        assert len(data) < 24000 * 4  # less than raw float32

    def test_encode_to_wav(self):
        pcm = np.random.randn(24000).astype(np.float32) * 0.1
        data, mime = encode_pcm_to_audio(pcm, "wav", 24000)

        assert len(data) > 0
        assert mime == "audio/wav"
        # WAV is PCM: 2 bytes per sample (s16le) + 78-byte header
        assert len(data) == 24000 * 2 + 78

    def test_encode_to_flac(self):
        pcm = np.random.randn(24000).astype(np.float32) * 0.1
        data, mime = encode_pcm_to_audio(pcm, "flac", 24000)

        assert len(data) > 0
        assert mime == "audio/flac"

    def test_encode_to_ogg(self):
        pcm = np.random.randn(24000).astype(np.float32) * 0.1
        data, mime = encode_pcm_to_audio(pcm, "ogg", 24000)

        assert len(data) > 0
        assert mime == "audio/ogg"

    def test_unsupported_format_raises(self):
        pcm = np.array([0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported output format"):
            encode_pcm_to_audio(pcm, "aac", 24000)

    def test_clips_out_of_range_values(self):
        """Values outside [-1, 1] should be clipped, not crash."""
        # Use a larger array; PyAV has issues with tiny buffers
        pcm = np.tile(np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32), 1000)
        data, _ = encode_pcm_to_audio(pcm, "wav", 24000)
        assert len(data) > 0


# ─── Round-trip ─────────────────────────────────────────────────────


class TestRoundTrip:
    def test_wav_roundtrip(self):
        """Decode WAV → encode WAV → decode again."""
        original = make_wav_bytes(duration_sec=0.5, sample_rate=24000, amplitude=0.5)
        pcm1 = decode_audio_to_pcm(original, "audio.wav")

        wav_bytes, _ = encode_pcm_to_audio(pcm1, "wav", 24000)
        pcm2 = decode_audio_to_pcm(wav_bytes, "audio.wav")

        assert len(pcm1) == len(pcm2)
        # Should be nearly identical (lossless for WAV)
        np.testing.assert_allclose(pcm1, pcm2, atol=0.001)

    def test_mp3_roundtrip(self):
        """Decode WAV → encode MP3 → decode (lossy but should be close)."""
        original = make_wav_bytes(duration_sec=1.0, sample_rate=24000, amplitude=0.3)
        pcm1 = decode_audio_to_pcm(original, "audio.wav")

        mp3_bytes, _ = encode_pcm_to_audio(pcm1, "mp3", 24000)
        pcm2 = decode_audio_to_pcm(mp3_bytes, "audio.mp3")

        # MP3 is lossy and frame-based; lengths may differ by a few percent
        assert len(pcm2) == pytest.approx(len(pcm1), abs=5000)
        # Mean absolute error should be small
        min_len = min(len(pcm1), len(pcm2))
        mae = np.mean(np.abs(pcm1[:min_len] - pcm2[:min_len]))
        assert mae < 0.05  # reasonable for MP3
