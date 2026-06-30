import os

from unmute.websocket_utils import http_to_ws

HEADERS = {"kyutai-api-key": "public_token"}

# The defaults are already ws://, but make the env vars support http:// and https://
STT_SERVER = http_to_ws(os.environ.get("KYUTAI_STT_URL", "ws://localhost:8090"))
TTS_SERVER = http_to_ws(os.environ.get("KYUTAI_TTS_URL", "ws://localhost:8089"))
LLM_SERVER = os.environ.get("KYUTAI_LLM_URL", "http://localhost:8091")
KYUTAI_LLM_MODEL = os.environ.get("KYUTAI_LLM_MODEL")
KYUTAI_LLM_API_KEY = os.environ.get("KYUTAI_LLM_API_KEY")
# If None, a dict-based cache will be used instead of Redis
REDIS_SERVER = os.environ.get("KYUTAI_REDIS_URL")

SPEECH_TO_TEXT_PATH = "/api/asr-streaming"
TEXT_TO_SPEECH_PATH = "/api/tts_streaming"

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
FRAME_TIME_SEC = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08
# TODO: make it so that we can read this from the ASR server?
STT_DELAY_SEC = 0.5
