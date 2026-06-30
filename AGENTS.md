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
- **This is a fork** with custom modifications
- **Custom files** are marked with `-custom` suffix or have `custom` in the name

---

## Directory Structure

```
unmute/                          ← Root
├── AGENTS.md                    ← You are here
├── README.md                    ← Original project documentation
├── README_CUSTOM.md             ← Your custom setup notes
├── pyproject.toml               ← Python project config (FastAPI, deps, tooling)
├── uv.lock                      ← Locked Python dependencies
├── voices.yaml                  ← Voice character definitions (name, instructions, voice path)
├── .env                         ← Environment variables (LLM URL, model, API key)
│
├── docker-compose.yml           ← Original Docker Compose (with Traefik + vLLM LLM)
├── docker-compose-custom.yml    ← YOUR custom compose (no Traefik, no vLLM, external LLM, GPU 1 pinned)
├── Dockerfile                   ← Backend Docker image (uv-based, hot-reload + prod targets)
├── .dockerignore                ← Patterns excluded from Docker builds (*.pyc, .venv, node_modules, volumes, etc.)
│
├── dockerless/                  ← Non-Docker startup scripts (manual service launching)
│   ├── start_backend.sh         ← FastAPI backend (uvicorn, port 8000)
│   ├── start_frontend.sh        ← Next.js frontend (pnpm dev)
│   ├── start_llm.sh             ← vLLM LLM server (port 8091)
│   ├── start_stt.sh             ← moshi-server STT worker (port 8090)
│   └── start_tts.sh             ← moshi-server TTS worker (port 8089)
│
├── frontend/                    ← Next.js 15 frontend (React 19, TypeScript)
│   ├── package.json             ← Node dependencies
│   ├── Dockerfile               ← Production frontend build
│   ├── hot-reloading.Dockerfile ← Dev frontend build (volume-mounted src/)
│   ├── next.config.ts             ← Next.js config (MDX support, image domains, etc.)
│   ├── public/                  ← Static assets (Opus encoder/decoder WebAssembly workers)
│   └── src/app/                 ← All frontend source
│
├── services/                    ← External service configs and Dockerfiles
│   ├── moshi-server/            ← Kyutai's moshi-server (Rust binary, STT+TTS)
│   │   ├── public.Dockerfile           ← Original TTS/STT Dockerfile
│   │   ├── public_custom.Dockerfile    ← YOUR custom Dockerfile (CUDA_COMPUTE_CAP=80, pre-compiled)
│   │   ├── private.Dockerfile          ← Private/internal Dockerfile
│   │   ├── start_moshi_server_public.sh         ← Public startup: HF auth + cargo install moshi-server from crates.io
│   │   ├── start_moshi_server_public_custom.sh  ← YOUR custom startup: pre-compiled binary, no cargo install
│   │   ├── start_moshi_server_private.sh        ← Kyutai-internal startup: HF auth + cargo run from source
│   │   └── configs/                   ← moshi-server TOML configs
│   │       ├── stt.toml               ← STT worker config (dev)
│   │       ├── stt-prod.toml          ← STT worker config (prod, batch_size=64, en_fr model, full transformer params)
│   │       ├── tts.toml               ← TTS worker config (dev)
│   │       ├── tts-custom.toml        ← YOUR custom TTS config (batch_size=2, en_fr tokenizer)
│   │       ├── tts-prod.toml          ← TTS worker config (prod, batch_size=16, kyutai/tts-voices folder, cfg_coef=2.0)
│   │       └── voice-cloning.toml           ← Voice cloning server config (embedding extraction parameters)
│   ├── grafana/               ← Monitoring dashboards (Grafana.ini, provisioning, dashboard JSON)
│   ├── prometheus/            ← Prometheus metrics config (scrape targets, prometheus.yml)
│   └── debugger/              ← Debug service (Dockerfile for ad-hoc debugging container)
│
├── unmute/                    ← Python backend package (FastAPI)
│   ├── main_websocket.py      ← ⭐ MAIN ENTRY POINT — FastAPI app, WebSocket routes, HTTP endpoints
│   ├── unmute_handler.py      ← ⭐ CORE LOGIC — UnmuteHandler class (conversation state machine)
│   ├── kyutai_constants.py    ← Environment variable constants (server URLs, sample rate, etc.)
│   ├── openai_realtime_api_events.py ← WebSocket message type definitions (Pydantic models)
│   ├── quest_manager.py       ← Async task lifecycle manager (init → run → close pattern)
│   ├── service_discovery.py   ← Service instance discovery (DNS-based, Redis optional)
│   ├── cache.py               ← Cache abstraction (Local dict or Redis)
│   ├── recorder.py            ← Session event recording (JSONL files)
│   ├── exceptions.py          ← Custom exception types
│   ├── metrics.py             ← Prometheus metrics definitions
│   ├── timer.py               ← Stopwatch utilities
│   ├── websocket_utils.py     ← HTTP↔WS URL conversion helpers
│   ├── webrtc_utils.py        ← WebRTC utilities
│   ├── audio_input_override.py ← Debug: inject audio file instead of mic
│   ├── audio_stream_saver.py  ← Audio stream persistence
│   ├── process_recording.py   ← Recording post-processing
│   │
│   ├── stt/                   ← Speech-to-Text client
│   │   ├── speech_to_text.py  ← STT WebSocket client (msgpack protocol, VAD pause prediction)
│   │   ├── exponential_moving_average.py ← EMA smoothing for pause detection
│   │   └── dummy_speech_to_text.py ← Mock STT for testing
│   │
│   ├── tts/                   ← Text-to-Speech client + voice management
│   │   ├── text_to_speech.py  ← TTS WebSocket client (msgpack protocol, realtime queue)
│   │   ├── realtime_queue.py  ← Time-aware queue for audio/text synchronization
│   │   ├── voice_cloning.py   ← Voice cloning via external server + cache
│   │   ├── voices.py          ← VoiceList loader (voices.yaml parser, upload utilities)
│   │   ├── voice_donation.py  ← Voice donation submission pipeline
│   │   ├── create_voice_donation_table.py     ← Flatten voice donation metadata to TSV for manual review/approval
│   │   ├── copy_approved_voice_donations.py   ← Copy approved voice .wav files with proper naming, trim trailing silence
│   │   ├── freesound_download.py ← Freesound.org audio download
│   │   ├── trim_voice_donation_clip.py          ← Trim leading/trailing silence from voice donation WAV clips
│   │   └── voice_donation_sentences.txt   ← Pool of ~10K verification sentences for voice donors (filtered from Mozilla CommonVoice)
│   │
│   ├── llm/                   ← LLM integration
│   │   ├── chatbot.py         ← Chat history management, conversation state machine
│   │   ├── llm_utils.py       ← OpenAI client wrapper, VLLMStream, message preprocessing
│   │   ├── system_prompt.py   ← System prompt templates (smalltalk, quiz, news, etc.)
│   │   ├── newsapi.py         ← News fetching from The Verge via NewsAPI
│   │   └── quiz_show_questions.py ← Quiz show question bank
│   │
│   ├── loadtest/              ← Load testing tools
│   │   ├── loadtest_client.py ← Multi-worker load test client
│   │   ├── loadtest_result.py ← Result analysis
│   │   ├── dummy_tts_server.py ← Mock TTS for testing
│   │   ├── generate_dataset_for_vllm.py   ← Generate 10K random prompts for vLLM benchmark_serving.py load testing
│   │   └── voices/            ← Test voice samples
│   │
│   └── scripts/               ← Utility scripts (examples, voice management)
│       ├── example_websocket_client.py              ← WebSocket client: send audio file → receive TTS response (OpenAI Realtime API)
│       ├── stt_from_file_example.py                 ← STT example: transcribe audio file via moshi-server
│       ├── stt_microphone_example.py                ← STT example: live microphone transcription via moshi-server
│       ├── tts_example.py                           ← TTS example: synthesize text via moshi-server
│       ├── output_tts.py                            ← fastrtc Stream: send "Hello, world!" to TTS, measure audio throughput ratio
│       ├── output_from_file.py                      ← fastrtc Stream: play back audio file through WebRTC
│       ├── output_sine.py                           ← fastrtc Stream: generate 440Hz sine wave (sync StreamHandler)
│       ├── output_sine_async.py                     ← fastrtc Stream: generate 440Hz sine wave (async AsyncStreamHandler)
│       ├── pitch_detection_handler.py               ← fastrtc Stream: detect input pitch via librosa.yin, synthesize matching sine wave
│       ├── update_voice_list.py                     ← Upload local voices to server and save updated voice list
│       ├── vllm_wrapper_example.py                  ← VLLMStream example: stream LLM chat completion with TTFT timing
│       ├── check_hugging_face_token_not_write.py    ← Security check: verify HF token is read-only (enforced in deployments)
│       └── copy_voice_to_prod.py                    ← Copy a voice file to the production server
│
├── tests/                     ← Python tests
│   ├── test_exponential_moving_average.py   ← Unit tests for EMA smoothing (attack/release timing)
│   └── test_llm_utils.py                    ← Unit tests for rechunk_to_words and preprocess_messages_for_llm
│
├── notebooks/                 ← Jupyter notebooks
│   └── create-voice-donation-sentences.ipynb   ← Filter CommonVoice dataset → voice_donation_sentences.txt (ASCII, 30-80 chars, no proper names)
│
├── docs/                      ← Documentation
│   └── browser_backend_communication.md ← WebSocket protocol documentation
│
├── .github/                   ← GitHub configs
│   ├── workflows/ci.yml               ← CI: pre-commit hooks (ruff, pyright, pnpm lint/build) + pytest
│   └── PULL_REQUEST_TEMPLATE.md       ← PR template with CLA checklist
│
├── volumes/                   ← Docker volume mounts (caches, models, logs)
│   ├── hf-cache/              ← HuggingFace model cache
│   ├── uv-cache/              ← uv package cache
│   ├── tts-logs/ / stt-logs/  ← Service logs
│   └── ...                    ← Other caches (cargo, vllm, etc.)
│
├── SWARM.md                   ← Docker Swarm deployment guide (production)
├── swarm-deploy.yml           ← Docker Swarm stack definition
├── setup_gpu_swarm_node.py    ← GPU node setup script
├── bake_deploy_prod.sh        ← Production deployment script
├── CONTRIBUTING.md            ← Contribution guidelines + Contributor License Agreement (CLA) terms
├── LICENSE                    ← MIT License (© 2025 kyutai)
└── .pre-commit-config.yaml    ← Pre-commit hooks: nbstripout, ruff lint/format, pyright, pnpm lint/build, trailing-whitespace
```

---

## Architecture Overview

### Services and their roles

| Service | Technology | Port (dockerless) | Description |
|---------|-----------|-------------------|-------------|
| **Frontend** | Next.js 15 / React 19 | 3000 | Web UI, microphone capture, Opus encoding, audio playback |
| **Backend** | FastAPI (Python) | 8000 (80 in Docker) | WebSocket orchestrator, connects STT→LLM→TTS pipeline |
| **STT** | moshi-server (Rust) | 8090 | Streaming speech-to-text with VAD (Voice Activity Detection) |
| **TTS** | moshi-server (Rust) | 8089 | Streaming text-to-speech (text-in → audio-out) |
| **LLM** | vLLM / external | 8091 (or external) | Text generation (OpenAI-compatible API) |
| **Voice Cloning** | moshi-server (Rust) | 8092 | Voice embedding extraction from audio samples |
| **Traefik** | Reverse proxy | 80 | HTTP routing (Docker Compose only; absent in custom setup) |

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

- **Frontend ↔ Backend**: JSON over WebSocket, OpenAI Realtime API-compatible messages (see `openai_realtime_api_events.py`)
- **Backend ↔ STT/TTS**: msgpack over WebSocket (binary protocol, see `speech_to_text.py` / `text_to_speech.py`)
- **Backend ↔ LLM**: OpenAI-compatible REST API streaming (uses `openai` Python SDK)

---

## Custom Modifications (Yours)

| File | What it does |
|------|--------------|
| `docker-compose-custom.yml` | Custom compose: no Traefik, no vLLM container, external LLM, GPU 1 pinned for STT+TTS, direct port mapping (3000, 8000) |
| `.env` | External LLM config (Ollama at 10.0.0.102:11434, gemma-4-31B-fast model) |
| `services/moshi-server/public_custom.Dockerfile` | Pre-compiled moshi-server binary with `CUDA_COMPUTE_CAP=80` (Ampere architecture) |
| `services/moshi-server/start_moshi_server_public_custom.sh` | Custom startup script (no cargo install at runtime — binary pre-compiled) |
| `services/moshi-server/configs/tts-custom.toml` | Custom TTS config: `batch_size=2`, English-French tokenizer, `cfg_coef=2.0` |
| `README_CUSTOM.md` | Documentation for the custom setup |

---

## Key Entry Points

- **Start the system (custom)**: `docker compose -f docker-compose-custom.yml up --build`
- **Backend entry**: `unmute/main_websocket.py` — FastAPI app with `uvicorn`
- **Frontend entry**: `frontend/src/app/page.tsx` → `Unmute.tsx` (main React component)
- **Conversation orchestration**: `unmute/unmute_handler.py` — `UnmuteHandler` class (state machine)
- **WebSocket protocol types**: `unmute/openai_realtime_api_events.py` — all message schemas
- **Voice definitions**: `voices.yaml` — character names, instructions, voice file paths

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KYUTAI_STT_URL` | STT server WebSocket URL | `ws://localhost:8090` |
| `KYUTAI_TTS_URL` | TTS server WebSocket URL | `ws://localhost:8089` |
| `KYUTAI_LLM_URL` | LLM server HTTP URL | `http://localhost:8091` |
| `KYUTAI_LLM_MODEL` | LLM model name (optional, auto-detected) | — |
| `KYUTAI_LLM_API_KEY` | LLM API key | — |
| `KYUTAI_VOICE_CLONING_URL` | Voice cloning server URL | `http://localhost:8092` |
| `KYUTAI_REDIS_URL` | Redis URL for distributed caching (optional) | local dict cache |
| `KYUTAI_RECORDINGS_DIR` | Directory for session recordings (optional) | no recording |
| `HUGGING_FACE_HUB_TOKEN` | HuggingFace token for model downloads | required for STT/TTS |
| `NEXT_PUBLIC_IN_DOCKER` | Frontend: whether running in Docker | `"true"` |

---

## Frontend Structure (`frontend/src/app/`)

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
| `cssUtil.ts` | Helper to read CSS custom properties (`getCSSVariable()`)
| `opus-recorder.d.ts` | TypeScript type declarations for the `opus-recorder` package |
| `useAudioVisualizerCircle.ts` | Canvas-based audio visualizer hook (frequency-reactive circle drawing) |
| `useGoogleAnalytics.ts` | GA event tracking for conversation start/end and recording downloads |
| `useLocalStorage.ts` | `useState` hook synced with `localStorage` persistence |
| `useWakeLock.ts` | Prevents screen sleep during active conversations (Screen Wake Lock API) |

---

## Testing & Development

- **Run backend (dev)**: `uv run fastapi dev unmute/main_websocket.py`
- **Run backend (prod)**: `uv run fastapi run unmute/main_websocket.py`
- **Run tests**: `uv run pytest`
- **Run loadtest**: `uv run unmute/loadtest/loadtest_client.py --server-url ws://localhost:8000`
- **Pre-commit hooks**: `pre-commit install --hook-type pre-commit`
- **Dev mode in frontend**: Edit `useKeyboardShortcuts.ts`, set `ALLOW_DEV_MODE = true`, then press `D`

---

## Important Classes and Their Roles

| Class | File | Role |
|-------|------|------|
| `UnmuteHandler` | `unmute/unmute_handler.py` | Core conversation state machine — orchestrates STT→LLM→TTS pipeline, handles interruptions, turn transitions |
| `Chatbot` | `unmute/llm/chatbot.py` | Chat history management, conversation state (`waiting_for_user` / `user_speaking` / `bot_speaking`) |
| `SpeechToText` | `unmute/stt/speech_to_text.py` | STT WebSocket client — sends audio, receives transcriptions + VAD pause predictions |
| `TextToSpeech` | `unmute/tts/text_to_speech.py` | TTS WebSocket client — sends text, receives PCM audio + text timing |
| `VLLMStream` | `unmute/llm/llm_utils.py` | OpenAI-compatible LLM streaming wrapper |
| `Quest` / `QuestManager` | `unmute/quest_manager.py` | Async task lifecycle manager — init→run→close with cancellation support |
| `Quest` names used | `unmute/unmute_handler.py` | `"stt"` (always running), `"tts"` (per-turn), `"llm"` (per-turn) |
| `VoiceList` | `unmute/tts/voices.py` | Loads `voices.yaml`, provides voice metadata to frontend |
| `RealtimeQueue` | `unmute/tts/realtime_queue.py` | Time-aware queue for synchronizing TTS audio/text output |
| `QuestManager` | `unmute/quest_manager.py` | Manages named async tasks; `add()` replaces existing quest with same name |

---

## Conversation State Machine

The `UnmuteHandler` maintains three conversation states (from `chatbot.py`):

1. **`waiting_for_user`** — Bot finished speaking, waiting for user input. After 7s of silence, sends `...` to prompt.
2. **`user_speaking`** — User is talking. STT sends transcriptions. VAD detects pause → triggers LLM response.
3. **`bot_speaking`** — LLM is generating, TTS is speaking. Can be interrupted by STT VAD (after 3s grace period) or by transcribed words.

Interruption flow: `interrupt_bot()` cancels TTS + LLM quests, clears output queue, resets to `waiting_for_user`.

---

## Voices Configuration (`voices.yaml`)

Each voice entry has:
- `name`: Display name
- `good`: Whether to include in the frontend voice list
- `instructions`: LLM system prompt type (`constant`, `smalltalk`, `quiz_show`, `guess_animal`, `news`, `unmute_explanation`)
- `source`: Voice audio file location (`file` or `freesound` type with `path_on_server`)

Instructions are loaded by `VoiceList` and sent to the backend via `session.update` WebSocket message. The backend's `Chatbot` builds the full system prompt from the instruction type.
