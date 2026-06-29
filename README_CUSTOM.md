# Custom Docker Compose Setup

This setup (`docker-compose-custom.yml`) is configured for a **multi-GPU environment with an external LLM server**.

## `.env` Configuration

Create a `.env` file and fill in the following variables:

```env
# External LLM provider (e.g. Ollama)
KYUTAI_LLM_URL=http://<your-llm-host>:<port>
KYUTAI_LLM_MODEL=<model-name>
KYUTAI_LLM_API_KEY=<your-api-key>

```

### Variable descriptions

| Variable | Description |
|---|---|
| `KYUTAI_LLM_URL` | URL of your external LLM server (e.g. `http://10.0.0.102:11434` for Ollama) |
| `KYUTAI_LLM_MODEL` | Model name to use (e.g. `gemma-4-31B-fast`) |
| `KYUTAI_LLM_API_KEY` | API key for the LLM provider (if required) |

## Key Differences from Standard Setup

- **No self-hosted LLM** — uses an external LLM endpoint instead of a local vLLM container
- **Pre-built TTS/STT images** — uses `public_custom.Dockerfile` with `CUDA_COMPUTE_CAP=80`
- **GPU pinned** — TTS and STT are assigned to GPU 1 only (`device_ids: ["1"]`)

## Running

```bash
docker compose -f docker-compose-custom.yml up --build
```
