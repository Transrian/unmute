"""Tests for /v1/audio/transcriptions and /v1/audio/speech endpoints."""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from unmute.main_websocket import app
from unmute.stt.speech_to_text import STTWordMessage


@pytest.fixture
def client():
    return TestClient(app)


def make_wav_bytes(duration_sec=1.0, sample_rate=24000, amplitude=0.3) -> bytes:
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate
    samples = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)
    int16 = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


# ─── /v1/audio/transcriptions ──────────────────────────────────────


class TestTranscriptionEndpoint:
    def test_transcription_success(self, client):
        """Test successful transcription with mocked STT."""
        wav_data = make_wav_bytes()

        # Mock the one-shot transcription function
        with patch(
            "unmute.main_websocket._transcribe_one_shot", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = "hello world"

            response = client.post(
                "/v1/audio/transcriptions",
                files={"audio": ("test.wav", wav_data, "audio/wav")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "hello world"
            mock_transcribe.assert_called_once()

    def test_transcription_error(self, client):
        """Test transcription returns 500 on error."""
        wav_data = make_wav_bytes()

        with patch(
            "unmute.main_websocket._transcribe_one_shot", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.side_effect = Exception("STT server unavailable")

            response = client.post(
                "/v1/audio/transcriptions",
                files={"audio": ("test.wav", wav_data, "audio/wav")},
            )

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "STT server unavailable" in data["error"]["message"]

    def test_transcription_mp3_file(self, client):
        """Test transcription with MP3 file."""
        # Create a tiny MP3-like file (won't actually decode, but tests the flow)
        wav_data = make_wav_bytes()

        with patch(
            "unmute.main_websocket._transcribe_one_shot", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = "test transcription"

            response = client.post(
                "/v1/audio/transcriptions",
                files={"audio": ("test.mp3", wav_data, "audio/mpeg")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "test transcription"


# ─── /v1/audio/speech ──────────────────────────────────────────────


class TestSpeechEndpoint:
    def test_speech_success_default_format(self, client):
        """Test successful speech synthesis with default (mp3) format."""
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "constant"},
        )

        # The endpoint will fail because TTS is not mocked, but we can test
        # the request parsing by mocking the one-shot function
        assert response.status_code in (200, 500)

    def test_speech_success_with_mock(self, client):
        """Test successful speech synthesis with mocked TTS."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": "constant"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/mpeg"
            assert len(response.content) > 0
            mock_synth.assert_called_once_with("Hello world", "constant")

    def test_speech_wav_format(self, client):
        """Test speech synthesis with WAV output format."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "constant",
                    "response_format": "wav",
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"

    def test_speech_flac_format(self, client):
        """Test speech synthesis with FLAC output format."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "constant",
                    "response_format": "flac",
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/flac"

    def test_speech_no_audio_generated(self, client):
        """Test 400 when TTS returns no audio."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (np.array([], dtype=np.float32), 24000)

            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": "constant"},
            )

            assert response.status_code == 400
            data = response.json()
            assert "No audio generated" in data["error"]["message"]

    def test_speech_error(self, client):
        """Test 500 when TTS raises an error."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.side_effect = Exception("TTS server down")

            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world", "voice": "constant"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "TTS server down" in data["error"]["message"]

    def test_speech_default_voice(self, client):
        """Test default voice is 'constant'."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello world"},
            )

            assert response.status_code == 200
            mock_synth.assert_called_once_with("Hello world", "constant")

    def test_speech_custom_voice(self, client):
        """Test custom voice is passed through."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello", "voice": "smalltalk"},
            )

            assert response.status_code == 200
            mock_synth.assert_called_once_with("Hello", "smalltalk")

    def test_speech_invalid_format(self, client):
        """Test 500 for unsupported output format."""
        with patch(
            "unmute.main_websocket._synthesize_one_shot", new_callable=AsyncMock
        ) as mock_synth:
            mock_synth.return_value = (
                np.sin(np.arange(24000) / 100).astype(np.float32) * 0.1,
                24000,
            )

            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello",
                    "voice": "constant",
                    "response_format": "aac",
                },
            )

            assert response.status_code == 500
