"""Tests for stt/speech_to_text.py: STT message types and SpeechToText class."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from unmute.stt.speech_to_text import (
    STTErrorMessage,
    STTMarkerMessage,
    STTReadyMessage,
    STTStepMessage,
    STTEndWordMessage,
    STTWordMessage,
    SpeechToText,
)


class TestSTTWordMessage:
    def test_create(self):
        msg = STTWordMessage(
            type="Word",
            text="hello",
            start_time=0.5,
        )
        assert msg.type == "Word"
        assert msg.text == "hello"
        assert msg.start_time == 0.5


class TestSTTEndWordMessage:
    def test_create(self):
        msg = STTEndWordMessage(
            type="EndWord",
            stop_time=1.0,
        )
        assert msg.type == "EndWord"
        assert msg.stop_time == 1.0


class TestSTTMarkerMessage:
    def test_create(self):
        msg = STTMarkerMessage(
            type="Marker",
            id=42,
        )
        assert msg.type == "Marker"
        assert msg.id == 42


class TestSTTStepMessage:
    def test_create(self):
        msg = STTStepMessage(
            type="Step",
            step_idx=10,
            prs=[0.1, 0.2, 0.3],
        )
        assert msg.type == "Step"
        assert msg.step_idx == 10
        assert msg.prs == [0.1, 0.2, 0.3]


class TestSTTErrorMessage:
    def test_create(self):
        msg = STTErrorMessage(
            type="Error",
            message="Service unavailable",
        )
        assert msg.type == "Error"
        assert msg.message == "Service unavailable"


class TestSTTReadyMessage:
    def test_create(self):
        msg = STTReadyMessage(type="Ready")
        assert msg.type == "Ready"


class TestSpeechToText:
    def test_init(self):
        stt = SpeechToText("ws://custom-stt:8080", delay_sec=0.2)
        assert stt.stt_instance == "ws://custom-stt:8080"
        assert stt.delay_sec == 0.2
        assert stt.websocket is None
        assert stt.sent_samples == 0
        assert stt.received_words == 0
        assert stt.waiting_first_step is True
        # pause_prediction should be initialized
        assert stt.pause_prediction.value == 1.0

    def test_init_defaults(self):
        from unmute.stt.speech_to_text import STT_DELAY_SEC

        # Check the constants are used as defaults
        assert STT_DELAY_SEC > 0

    def test_state_not_created(self):
        stt = SpeechToText()
        assert stt.state() == "not_created"

    def test_state_connected(self):
        stt = SpeechToText()
        import websockets.protocol

        mock_ws = MagicMock()
        mock_ws.state = websockets.protocol.State.OPEN
        stt.websocket = mock_ws
        assert stt.state() == "connected"

    def test_state_closed(self):
        stt = SpeechToText()
        import websockets.protocol

        mock_ws = MagicMock()
        mock_ws.state = websockets.protocol.State.CLOSED
        stt.websocket = mock_ws
        assert stt.state() == "closed"

    @pytest.mark.asyncio
    async def test_send_audio_wrong_dims(self):
        stt = SpeechToText()
        audio = np.zeros((2, 4800), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 1D array"):
            await stt.send_audio(audio)

    @pytest.mark.asyncio
    async def test_send_audio_converts_dtype(self):
        stt = SpeechToText()
        stt._send = AsyncMock()
        audio = np.zeros(4800, dtype=np.int16)  # Not float32
        await stt.send_audio(audio)
        stt._send.assert_called_once()
        assert stt.sent_samples == 4800

    @pytest.mark.asyncio
    async def test_send_audio_float32(self):
        stt = SpeechToText()
        stt._send = AsyncMock()
        audio = np.zeros(4800, dtype=np.float32)
        await stt.send_audio(audio)
        stt._send.assert_called_once()
        assert stt.sent_samples == 4800

    @pytest.mark.asyncio
    async def test_send_marker(self):
        stt = SpeechToText()
        stt._send = AsyncMock()
        await stt.send_marker(42)
        stt._send.assert_called_once_with({"type": "Marker", "id": 42})

    @pytest.mark.asyncio
    async def test_send_no_websocket(self):
        stt = SpeechToText()
        stt.websocket = None
        # _send should not raise, just log a warning
        await stt._send({"type": "Audio", "pcm": []})
