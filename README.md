# Local Multimodal LLM Simulation

A local-first alternative to cloud-based realtime AI APIs (e.g. Gemini Live, OpenAI Realtime). Processes voice and vision input through a pipeline of local, open-weight models:

```
Silero VAD → Whisper STT → Gemma 4 LLM → Kokoro TTS
```

All inference runs locally — no cloud API calls (except optional agent escalation). Designed for edge devices with a GPU (e.g. NVIDIA DGX Spark, Jetson Orin).

## Features

- **Real-time voice conversation** with barge-in interruption (VAD-driven)
- **Camera vision** — feeds frames to the multimodal LLM
- **Function calling** — frontend declares tools via WebSocket `session.update`, LLM calls them
- **Agent escalation** — forwards complex queries to an external OpenAI-compatible API
- **Person identification** — tool-based face recognition via an external API (fully opt-in)
- **Sentence-streaming TTS** — generates audio as the LLM streams text
- **Web test console** — built-in UI for mic, camera, and conversation

## Architecture

```
Client (browser/app)
  │
  ├── Audio (16kHz PCM, binary) ──┐
  ├── Video frames (base64 JPEG) ─┤
  └── Control JSON ───────────────┤
                                  ▼
                        ┌──────────────────┐
                        │   gateway.py     │
                        │  (FastAPI + WS)  │
                        ├──────────────────┤
                        │ Silero VAD       │ ← voice activity detection
                        │ Whisper STT      │ ← speech to text
                        │ Gemma 4 LLM      │ ← reasoning + tool calls
                        │ Kokoro TTS       │ ← text to speech
                        └──────┬───────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              LLM Server   Agent API   Face Rec API
              (vLLM)       (optional)  (optional)
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended) — STT and LLM benefit massively
- An OpenAI-compatible LLM server (e.g. [vLLM](https://github.com/vllm-project/vllm)) serving a Gemma model
- Kokoro TTS model files (`.onnx` + voices)

### Install

```bash
git clone https://github.com/chishiki37/Local-Multimodal-LLM-Simulation.git
cd Local-Multimodal-LLM-Simulation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download Model Files

The gateway uses three local models. Download them after installing dependencies:

#### 1. Kokoro TTS (text-to-speech)

Kokoro needs two files — the ONNX model and the voices pack:

```bash
mkdir -p models/kokoro
cd models/kokoro

# Model (~300 MB)
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# Voices pack (~27 MB)
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

cd ../..
```

Available voices: `af_heart` (default), `af_bella`, `af_nova`, `am_adam`, `bf_emma`, and more. See the [Kokoro project](https://github.com/hexgrad/kokoro) for the full list.

#### 2. Whisper STT (speech-to-text)

Whisper models auto-download from HuggingFace on first run — no manual step needed. The first transcription will be slower as it fetches the model (~1.5 GB for `large-v3-turbo`).

**GPU (transformers backend — default, faster):**
- Model: `large-v3-turbo` (~1.5 GB) — set via `STT_MODEL` in `.env`

**CPU alternative (faster-whisper):**
If you don't have a CUDA GPU, use the faster-whisper backend instead:

```bash
pip install faster-whisper
```

Then in `.env`:
```
STT_BACKEND=faster_whisper
STT_DEVICE=cpu
STT_COMPUTE=int8
```

Models auto-download from HuggingFace on first use. Smaller options: `base`, `small`, `medium`, `large-v3`.

#### 3. Gemma 4 LLM (reasoning)

The gateway connects to an **external** OpenAI-compatible LLM server — it does not load the LLM itself. You need a separate server running:

**Option A — vLLM (recommended for GPU):**
```bash
pip install vllm
vllm serve google/gemma-3-12b-it --port 8000
```

**Option B — llama.cpp (CPU/lightweight GPU):**
```bash
# Download a GGUF quantized model from HuggingFace
# Serve with OpenAI-compatible API:
./llama-server -m gemma-3-12b-it-Q4_K_M.gguf --port 8000
```

Then point the gateway at it via `.env`:
```
VLLM_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=gemma-3-12b-it
```

> **Note:** Any OpenAI-compatible API works here — Ollama, LM Studio, TGI, etc. Just set `VLLM_URL` and `VLLM_MODEL` to match.

#### 4. Silero VAD (voice activity detection)

Auto-installed via `silero-vad` in requirements.txt. Model weights download automatically on first run (~2 MB).

### Configure

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_PORT` | `8001` | WebSocket + HTTP port |
| `VLLM_URL` | `http://localhost:8000/v1/chat/completions` | LLM server endpoint |
| `VLLM_MODEL` | `gemma-4-12b` | Model name on the LLM server |
| `STT_MODEL` | `large-v3-turbo` | Whisper model |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `TTS_VOICE` | `af_heart` | Kokoro voice |
| `ENABLE_VISION` | `1` | Accept camera frames |
| `ENABLE_ESCALATION` | `1` | Enable agent escalation fallback |

### Run

```bash
python gateway.py
```

Open `https://localhost:8001/` in your browser (self-signed cert — accept the warning).

## WebSocket Protocol

Connect to `wss://host:8001/realtime`.

### Client → Server

| Type | Format | Description |
|---|---|---|
| Audio | Binary PCM | 16kHz, 16-bit, mono, little-endian |
| `video_frame` | JSON | `{type: "video_frame", data: "<base64 JPEG>"}` |
| `session.update` | JSON | Set instructions + tools (see below) |
| `conversation.item.create` | JSON | Send text input instead of voice |
| `response.cancel` | JSON | Client-initiated interruption |

### Server → Client

| Type | Description |
|---|---|
| Audio | Binary PCM at 24kHz from TTS |
| `transcript` | STT result |
| `response_start` | LLM response beginning |
| `response_text` | Accumulated response text (streaming) |
| `response_end` | Response complete |
| `interrupted` | Barge-in detected |
| `toolCall` | Tool execution notification |
| `toolResponse` | Tool result |
| `escalating` | Forwarding to agent backend |
| `error` | Error message |

### Function Calling (Tool Use)

Send a `session.update` with tools to enable function-calling mode:

```json
{
  "type": "session.update",
  "session": {
    "instructions": "You are a helpful assistant with access to tools.",
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "execute",
          "description": "Delegate a task to an external agent",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "The task to execute"}
            },
            "required": ["query"]
          }
        }
      }
    ]
  }
}
```

Built-in tools:
- **`execute`** — forwards a query to the agent backend (`AGENT_URL`)
- **`identify_person`** — sends the current camera frame to the face recognition API (`PERSON_URL`)

### Escalation (No-Tools Mode)

When no tools are declared, the LLM is instructed to output `[ESCALATE] <description>` for tasks it can't handle (web search, emails, etc.). The gateway detects this marker and forwards the request to the agent backend.

## Project Structure

```
local-multimodal-llm-simulation/
├── gateway.py          # Main gateway — FastAPI WebSocket server
├── web/
│   └── index.html      # Test console (served at /)
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
