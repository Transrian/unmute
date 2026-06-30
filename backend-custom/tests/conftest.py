"""Shared test fixtures and fake services for the Unmute backend test suite."""

import asyncio
import os

import numpy as np
import pytest

# Ensure the unmute package is importable
os.environ.setdefault("KYUTAI_LLM_URL", "http://localhost:8091")
os.environ.setdefault("KYUTAI_LLM_MODEL", "test-model")
os.environ.pop("KYUTAI_LLM_API_KEY", None)
os.environ.pop("KYUTAI_STT_URL", None)
os.environ.pop("KYUTAI_TTS_URL", None)
os.environ.pop("NEWSAPI_API_KEY", None)
os.environ.pop("FREESOUND_API_KEY", None)


# ──────────────────────────────────────────────────────────────
# Fake STT Service
# ──────────────────────────────────────────────────────────────

class FakeSpeechToText:
    """Fake STT service that yields predefined messages instead of connecting to a real server."""

    def __init__(
        self,
        messages: list[dict] | None = None,
        delay_sec: float = 0.5,
        stt_instance: str = "ws://fake-stt:8080",
    ):
        self.stt_instance = stt_instance
        self.delay_sec = delay_sec
        self.messages = messages or []
        self.message_index = 0
        self.sent_samples = 0
        self.received_words = 0
        self.current_time = -delay_sec
        self.time_since_first_audio_sent = None
        self.waiting_first_step = True
        self.shutdown_complete = asyncio.Event()
        self.audio_chunks: list[np.ndarray] = []

        from unmute.stt.exponential_moving_average import ExponentialMovingAverage

        self.pause_prediction = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )

    def state(self) -> str:
        return "connected"

    async def send_audio(self, audio: np.ndarray) -> None:
        self.audio_chunks.append(audio.copy())
        self.sent_samples += len(audio)

    async def send_marker(self, id: int) -> None:
        pass

    async def _send(self, data: dict) -> None:
        pass

    async def start_up(self) -> None:
        pass

    async def shutdown(self) -> None:
        if not self.shutdown_complete.is_set():
            self.shutdown_complete.set()

    async def __aiter__(self):
        from unmute.stt.speech_to_text import STTMarkerMessage, STTWordMessage

        FRAME_TIME_SEC = 0.08
        for msg_dict in self.messages:
            msg_type = msg_dict.get("type")
            if msg_type == "Word":
                msg = STTWordMessage(
                    type="Word",
                    text=msg_dict["text"],
                    start_time=msg_dict.get("start_time", 0.0),
                )
                self.received_words += 1
                yield msg
            elif msg_type == "Marker":
                yield STTMarkerMessage(type="Marker", id=msg_dict.get("id", 0))
            elif msg_type == "Step":
                self.current_time += FRAME_TIME_SEC
                if self.waiting_first_step:
                    self.waiting_first_step = False
                prs = msg_dict.get("prs", [0.0, 0.0, 0.5])
                self.pause_prediction.update(
                    dt=FRAME_TIME_SEC, new_value=prs[2]
                )
            elif msg_type == "EndWord":
                pass  # Skip
        # Signal end
        self.shutdown_complete.set()


# ──────────────────────────────────────────────────────────────
# Fake TTS Service
# ──────────────────────────────────────────────────────────────

class FakeTextToSpeech:
    """Fake TTS service that yields predefined messages instead of connecting to a real server."""

    def __init__(
        self,
        messages: list[dict] | None = None,
        voice: str | None = None,
        get_time=None,
        tts_instance: str = "ws://fake-tts:8080",
    ):
        self.tts_instance = tts_instance
        self.messages = messages or []
        self.message_index = 0
        self.voice = voice
        self.received_samples = 0
        self.received_samples_yielded = 0
        self.sent_texts: list[str] = []
        self.shutdown_complete = asyncio.Event()
        self.shutdown_lock = asyncio.Lock()
        self.websocket = object()  # Fake websocket object
        self.text_output_queue = None
        self.time_since_first_text_sent = None
        self.waiting_first_audio = True

        from unmute.tts.realtime_queue import RealtimeQueue
        self.text_output_queue = RealtimeQueue(get_time=get_time)

    def state(self) -> str:
        return "connected"

    async def send(self, message) -> None:
        if isinstance(message, str):
            self.sent_texts.append(message)

    async def start_up(self) -> None:
        pass

    async def shutdown(self) -> None:
        if not self.shutdown_complete.is_set():
            self.shutdown_complete.set()

    async def __aiter__(self):
        from unmute.tts.text_to_speech import TTSAudioMessage, TTSTextMessage

        for msg_dict in self.messages:
            msg_type = msg_dict.get("type")
            if msg_type == "Audio":
                msg = TTSAudioMessage(
                    type="Audio",
                    pcm=msg_dict.get("pcm", [0.0] * 100),
                )
                self.received_samples += len(msg.pcm)
                yield msg
            elif msg_type == "Text":
                msg = TTSTextMessage(
                    type="Text",
                    text=msg_dict.get("text", ""),
                    start_s=msg_dict.get("start_s", 0.0),
                    stop_s=msg_dict.get("stop_s", 0.0),
                )
                yield msg
            elif msg_type == "Ready":
                pass
        self.shutdown_complete.set()


# ──────────────────────────────────────────────────────────────
# Fake LLM Stream
# ──────────────────────────────────────────────────────────────

class FakeVLLMStream:
    """Fake LLM stream that yields predefined text chunks."""

    def __init__(self, responses: list[str], temperature: float = 1.0, client=None):
        self.responses = responses
        self.temperature = temperature
        self.client = client

    async def chat_completion(self, messages):
        for response in self.responses:
            # Yield in chunks to simulate streaming
            for i in range(0, len(response), 3):
                yield response[i : i + 3]
                await asyncio.sleep(0)  # Yield to event loop


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def fake_stt_messages() -> list[dict]:
    """Default fake STT messages simulating a user saying 'hello world'."""
    return [
        {"type": "Step", "prs": [0.1, 0.1, 0.3]},
        {"type": "Word", "text": "hello", "start_time": 0.1},
        {"type": "Step", "prs": [0.1, 0.1, 0.2]},
        {"type": "Word", "text": "world", "start_time": 0.2},
        {"type": "EndWord", "stop_time": 0.3},
    ]


@pytest.fixture
def fake_tts_messages() -> list[dict]:
    """Default fake TTS messages simulating a short response."""
    return [
        {"type": "Ready"},
        {"type": "Text", "text": "Hello", "start_s": 0.0, "stop_s": 0.5},
        {"type": "Audio", "pcm": [0.0] * 100},
        {"type": "Text", "text": " there", "start_s": 0.5, "stop_s": 1.0},
        {"type": "Audio", "pcm": [0.0] * 100},
    ]


@pytest.fixture
def fake_llm_response() -> list[str]:
    """Default fake LLM response."""
    return ["Hello there!"]



