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
git clone https://github.com/<your-user>/local-multimodal-llm-simulation.git
cd local-multimodal-llm-simulation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
