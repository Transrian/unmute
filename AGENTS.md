# Unmute — Repository Map for LLM Agents

This file is a structural guide to help LLM agents navigate this repository efficiently. Read it before browsing the codebase.

## What is Unmute?

Unmute wraps a text LLM with Kyutai's streaming Speech-to-Text (STT) and Text-to-Speech (TTS) models to create real-time voice conversations. The pipeline is:

```
User microphone → STT (transcription) → LLM (text response) → TTS (speech synthesis) → User speakers
```

**Protocol**: WebSocket-based, loosely compatible with the [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime).

---

## Repository Origin

- **Original repo**: [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute)
- **This is a fork** — all development happens in `frontend-custom/` and `backend-custom/`

---

## Directory Structure

```
unmute/                          ← Root
├── AGENTS.md                    ← You are here
├── README_CUSTOM.md             ← Custom setup notes
├── .env                         ← Environment variables (LLM URL, model, API key)
│
├── docker-compose-custom.yml    ← Custom compose (no Traefik, no vLLM, external LLM, GPU 1 pinned)
├── .dockerignore                ← Patterns excluded from Docker builds
│
├── frontend-custom/             ← Custom frontend (Next.js 15, React 19, TypeScript)
│   ├── package.json             ← Node dependencies
│   ├── Dockerfile               ← Production frontend build
│   ├── hot-reloading.Dockerfile ← Dev frontend build (volume-mounted src/)
│   ├── next.config.ts           ← Next.js config
│   ├── public/                  ← Static assets (Opus encoder/decoder WebAssembly workers)
│   └── src/app/                 ← All frontend source
│
├── backend-custom/              ← Custom backend (FastAPI) — self-contained build context
│   ├── Dockerfile               ← Backend Docker image (uv-based, hot-reload + prod targets)
│   ├── pyproject.toml           ← Python project config (FastAPI, deps, tooling)
│   ├── uv.lock                  ← Locked Python dependencies
│   ├── voices.yaml              ← Voice character definitions (name, instructions, voice path)
│   │
│   ├── unmute/                  ← Python package
│   │   ├── main_websocket.py    ← ⭐ MAIN ENTRY POINT — FastAPI app, WebSocket routes, HTTP endpoints
│   │   ├── unmute_handler.py    ← ⭐ CORE LOGIC — UnmuteHandler class (conversation state machine)
│   │   ├── kyutai_constants.py  ← Environment variable constants (server URLs, sample rate, etc.)
│   │   ├── openai_realtime_api_events.py ← WebSocket message type definitions (Pydantic models)
│   │   ├── quest_manager.py     ← Async task lifecycle manager (init → run → close pattern)
│   │   ├── service_discovery.py ← Service instance discovery (DNS-based, Redis optional)
│   │   ├── cache.py             ← Cache abstraction (Local dict or Redis)
│   │   ├── recorder.py          ← Session event recording (JSONL files)
│   │   ├── exceptions.py        ← Custom exception types
│   │   ├── metrics.py           ← Prometheus metrics definitions
│   │   ├── timer.py             ← Stopwatch utilities
│   │   ├── websocket_utils.py   ← HTTP↔WS URL conversion helpers
│   │   ├── webrtc_utils.py      ← WebRTC utilities
│   │   ├── audio_input_override.py  ← Debug: inject audio file instead of mic
│   │   ├── audio_stream_saver.py    ← Audio stream persistence
│   │   ├── process_recording.py     ← Recording post-processing
│   │   │
│   │   ├── stt/                 ← Speech-to-Text client
│   │   │   ├── speech_to_text.py          ← STT WebSocket client (msgpack protocol, VAD pause prediction)
│   │   │   ├── exponential_moving_average.py ← EMA smoothing for pause detection
│   │   │   └── dummy_speech_to_text.py    ← Mock STT for testing
│   │   │
│   │   ├── tts/                 ← Text-to-Speech client + voice management
│   │   │   ├── text_to_speech.py    ← TTS WebSocket client (msgpack protocol, realtime queue)
│   │   │   ├── realtime_queue.py    ← Time-aware queue for audio/text synchronization
│   │   │   ├── voice_cloning.py     ← Voice cloning via external server + cache
│   │   │   ├── voices.py            ← VoiceList loader (voices.yaml parser, upload utilities)
│   │   │   └── ...                  ← Voice donation pipeline scripts
│   │   │
│   │   ├── llm/                 ← LLM integration
│   │   │   ├── chatbot.py             ← Chat history management, conversation state machine
│   │   │   ├── llm_utils.py           ← OpenAI client wrapper, VLLMStream, message preprocessing
│   │   │   ├── system_prompt.py       ← System prompt templates (smalltalk, quiz, news, etc.)
│   │   │   ├── newsapi.py             ← News fetching from The Verge via NewsAPI
│   │   │   └── quiz_show_questions.py ← Quiz show question bank
│   │   │
│   │   ├── loadtest/            ← Load testing tools
│   │   └── scripts/             ← Utility scripts (examples, voice management)
│
├── services/                    ← External service configs and Dockerfiles
│   ├── moshi-server/            ← Kyutai's moshi-server (Rust binary, STT+TTS)
│   │   ├── public_custom.Dockerfile    ← Custom Dockerfile (CUDA_COMPUTE_CAP=80, pre-compiled)
│   │   ├── start_moshi_server_public_custom.sh  ← Custom startup (pre-compiled binary)
│   │   └── configs/
│   │       ├── stt.toml               ← STT worker config (dev)
│   │       └── tts-custom.toml        ← Custom TTS config (batch_size=2, en_fr tokenizer)
│   └── ...
│
├── tests/                       ← Python tests
├── notebooks/                   ← Jupyter notebooks
├── docs/                        ← Documentation
├── .github/                     ← GitHub configs
└── volumes/                     ← Docker volume mounts (caches, models, logs)
```

---

## Architecture Overview

### Services and their roles

| Service | Technology | Port (custom) | Description |
|---------|-----------|---------------|-------------|
| **Frontend** | Next.js 15 / React 19 | 3000 | Web UI, microphone capture, Opus encoding, audio playback |
| **Backend** | FastAPI (Python) | 8000 (80 in Docker) | WebSocket orchestrator, connects STT→LLM→TTS pipeline |
| **STT** | moshi-server (Rust) | 8080 (Docker internal) | Streaming speech-to-text with VAD (Voice Activity Detection) |
| **TTS** | moshi-server (Rust) | 8080 (Docker internal) | Streaming text-to-speech (text-in → audio-out) |
| **LLM** | external (Ollama) | external | Text generation (OpenAI-compatible API) |

### Data Flow

```
Browser microphone
  → Opus encoding (opus-recorder + WASM encoder)
  → WebSocket → Backend (main_websocket.py)
    → STT server (speech_to_text.py) — msgpack over WebSocket
    → Transcription text → Chatbot (chatbot.py)
    → LLM (llm_utils.py / VLLMStream) — OpenAI-compatible streaming
    → TTS server (text_to_speech.py) — msgpack over WebSocket
    → PCM audio → Opus encoding → WebSocket → Browser
  → Web Audio API decoding (WASM decoder)
  → Speakers
```

### Key Communication Protocols

- **Frontend ↔ Backend**: JSON over WebSocket, OpenAI Realtime API-compatible messages
- **Backend ↔ STT/TTS**: msgpack over WebSocket (binary protocol)
- **Backend ↔ LLM**: OpenAI-compatible REST API streaming (uses `openai` Python SDK)

---

## Key Entry Points

- **Start the system**: `docker compose -f docker-compose-custom.yml up --build`
- **Backend entry**: `backend-custom/unmute/main_websocket.py` — FastAPI app with `uvicorn`
- **Frontend entry**: `frontend-custom/src/app/page.tsx` → `Unmute.tsx` (main React component)
- **Conversation orchestration**: `backend-custom/unmute/unmute_handler.py` — `UnmuteHandler` class (state machine)
- **WebSocket protocol types**: `backend-custom/unmute/openai_realtime_api_events.py` — all message schemas
- **Voice definitions**: `backend-custom/voices.yaml` — character names, instructions, voice file paths

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KYUTAI_STT_URL` | STT server WebSocket URL | `ws://stt:8080` |
| `KYUTAI_TTS_URL` | TTS server WebSocket URL | `ws://tts:8080` |
| `KYUTAI_LLM_URL` | LLM server HTTP URL | from `.env` |
| `KYUTAI_LLM_MODEL` | LLM model name | from `.env` |
| `KYUTAI_LLM_API_KEY` | LLM API key | from `.env` |
| `HUGGING_FACE_HUB_TOKEN` | HuggingFace token for model downloads | required for STT/TTS |
| `NEXT_PUBLIC_IN_DOCKER` | Frontend: whether running in Docker | `"false"` |

---

## Frontend Structure (`frontend-custom/src/app/`)

| File | Purpose |
|------|---------|
| `Unmute.tsx` | ⭐ Main component — WebSocket connection, audio processing, message handling |
| `UnmuteConfigurator.tsx` | Voice/instruction selector UI |
| `PositionedAudioVisualizer.tsx` | Animated audio visualization circles |
| `Subtitles.tsx` | Real-time subtitle display |
| `useAudioProcessor.ts` | Audio capture, Opus encoding, playback decoding |
| `useMicrophoneAccess.ts` | Microphone permission handling |
| `useBackendServerUrl.ts` | Backend URL resolution (Docker vs local) |
| `useKeyboardShortcuts.ts` | Keyboard shortcuts (S=subtitles, D=dev mode) |
| `audioUtil.ts` | Opus encode/decode helpers |
| `chatHistory.ts` | Chat message compression utilities |
| `VoiceRecorder.tsx` / `VoiceUpload.tsx` | Voice cloning upload UI |
| `CouldNotConnect.tsx` | Error screen when backend is unreachable |
| `ConsentModal.tsx` | Recording consent dialog |
| `useRecordingCanvas.ts` | Canvas-based conversation recording |
| `voice-donation/` | Voice donation sub-pages |
| `ErrorMessages.tsx` | Toast-style error message display (auto-dismiss after 10s) |
| `layout.tsx` | Root layout — sets page metadata, loads Satoshi font, mounts ConsentModal |
| `page.tsx` | Home page — renders the `<Unmute />` component |
| `Modal.tsx` | Hover/click-triggered modal (positioned on desktop, fullscreen on mobile) |
| `SingleRoleSubtitles.tsx` | Subtitles for a single speaker with word-wrapping and max line count |
| `SlantedButton.tsx` | Stylized button with slanted (skewed) border (primary/secondary/disabled) |
| `SquareButton.tsx` | Square-cornered button with dashed border (primary/primaryOff/secondary) |
| `TrimmedAudioPreview.tsx` | Audio preview player for uploaded voice clips (10s max playback) |
| `UnmuteHeader.tsx` | Header bar with title, Kyutai logo, and "More info" modal |
| `VoiceAttribution.tsx` | Voice source attribution display (file description or Freesound link) |
| `cssUtil.ts` | Helper to read CSS custom properties (`getCSSVariable()`) |
| `opus-recorder.d.ts` | TypeScript type declarations for the `opus-recorder` package |
| `useAudioVisualizerCircle.ts` | Canvas-based audio visualizer hook (frequency-reactive circle drawing) |
| `useGoogleAnalytics.ts` | GA event tracking for conversation start/end and recording downloads |
| `useLocalStorage.ts` | `useState` hook synced with `localStorage` persistence |
| `useWakeLock.ts` | Prevents screen sleep during active conversations (Screen Wake Lock API) |

---

## Important Classes and Their Roles

| Class | File | Role |
|-------|------|------|
| `UnmuteHandler` | `backend-custom/unmute/unmute_handler.py` | Core conversation state machine — orchestrates STT→LLM→TTS pipeline, handles interruptions, turn transitions |
| `Chatbot` | `backend-custom/unmute/llm/chatbot.py` | Chat history management, conversation state (`waiting_for_user` / `user_speaking` / `bot_speaking`) |
| `SpeechToText` | `backend-custom/unmute/stt/speech_to_text.py` | STT WebSocket client — sends audio, receives transcriptions + VAD pause predictions |
| `TextToSpeech` | `backend-custom/unmute/tts/text_to_speech.py` | TTS WebSocket client — sends text, receives PCM audio + text timing |
| `VLLMStream` | `backend-custom/unmute/llm/llm_utils.py` | OpenAI-compatible LLM streaming wrapper |
| `Quest` / `QuestManager` | `backend-custom/unmute/quest_manager.py` | Async task lifecycle manager — init→run→close with cancellation support |
| `Quest` names used | `backend-custom/unmute/unmute_handler.py` | `"stt"` (always running), `"tts"` (per-turn), `"llm"` (per-turn) |
| `VoiceList` | `backend-custom/unmute/tts/voices.py` | Loads `voices.yaml`, provides voice metadata to frontend |
| `RealtimeQueue` | `backend-custom/unmute/tts/realtime_queue.py` | Time-aware queue for synchronizing TTS audio/text output |

---

## Conversation State Machine

The `UnmuteHandler` maintains three conversation states (from `chatbot.py`):

1. **`waiting_for_user`** — Bot finished speaking, waiting for user input. After 7s of silence, sends `...` to prompt.
2. **`user_speaking`** — User is talking. STT sends transcriptions. VAD detects pause → triggers LLM response.
3. **`bot_speaking`** — LLM is generating, TTS is speaking. Can be interrupted by STT VAD (after 3s grace period) or by transcribed words.

Interruption flow: `interrupt_bot()` cancels TTS + LLM quests, clears output queue, resets to `waiting_for_user`.

---

## Voices Configuration (`backend-custom/voices.yaml`)

Each voice entry has:
- `name`: Display name
- `good`: Whether to include in the frontend voice list
- `instructions`: LLM system prompt type (`constant`, `smalltalk`, `quiz_show`, `guess_animal`, `news`, `unmute_explanation`)
- `source`: Voice audio file location (`file` or `freesound` type with `path_on_server`)

Instructions are loaded by `VoiceList` and sent to the backend via `session.update` WebSocket message. The backend's `Chatbot` builds the full system prompt from the instruction type.
