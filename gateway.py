#!/usr/bin/env python3
"""
Local Multimodal LLM Simulation — Realtime voice + vision assistant

A local-first alternative to cloud-based realtime AI APIs (e.g. Gemini Live,
OpenAI Realtime). Processes voice and vision input through a pipeline of
local models:
  Silero VAD → Whisper STT (transformers) → Gemma 4 LLM → Kokoro TTS

Features:
  - Real-time voice conversation with barge-in interruption
  - Camera frame vision (multimodal LLM input)
  - Function calling (tool use) via WebSocket protocol
  - Agent escalation (forward queries to an external OpenAI-compatible API)
  - Person identification (via external face recognition API, tool-based)

WebSocket Protocol (ws://host:8001/realtime):
  Client → Server:
    Binary frames: raw PCM audio (16kHz, 16-bit, mono, little-endian)
    JSON frames: {"type": "video_frame", "data": "<base64 JPEG>"}
                 {"type": "config", "sample_rate": 16000}
                 {"type": "session.update", "session": {"instructions": "...", "tools": [...]}}
  Server → Client:
    Binary frames: raw PCM audio (24kHz, 16-bit, mono) from TTS
    JSON frames: {"type": "transcript", "text": "...", "is_final": bool}
                 {"type": "response_start"}
                 {"type": "response_text", "text": "..."}  (accumulated)
                 {"type": "response_end"}
                 {"type": "interrupted"}
                 {"type": "error", "message": "..."}
                 {"type": "toolCall", ...}       (tool execution notification)
                 {"type": "toolResponse", ...}   (tool result notification)
"""
import os, sys, json, time, asyncio, logging, base64, re, traceback
from collections import deque
from io import BytesIO
from typing import Optional, AsyncGenerator

import numpy as np

# --- .env loading (optional, for local development with secrets) ---
for _env_path in ("./.env", os.path.expanduser("~/.env")):
    if os.path.exists(_env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path, override=False)
            break
        except ImportError:
            pass

# --- Config ---
HOST = os.environ.get("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("GATEWAY_PORT", "8001"))
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "gemma-4-12b")
STT_MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
STT_DEVICE = os.environ.get("STT_DEVICE", "cuda")
STT_BACKEND = os.environ.get("STT_BACKEND", "transformers")  # transformers (GPU) or faster_whisper (CPU)
TTS_MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "./models/kokoro")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")
KOKORO_ONNX = os.path.join(TTS_MODEL_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(TTS_MODEL_DIR, "voices-v1.0.bin")

INPUT_SAMPLE_RATE = 16000   # client sends 16kHz
OUTPUT_SAMPLE_RATE = 24000  # Kokoro outputs 24kHz

# VAD params
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "400"))
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "250"))
VAD_CHUNK_SAMPLES = 512  # Silero VAD expects 512 samples at 16kHz

# Vision
ENABLE_VISION = os.environ.get("ENABLE_VISION", "1") == "1"

# Conversation
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "10"))

# --- Person recognition (external face recognition API, tool-based) ---
PERSON_URL = os.environ.get("PERSON_URL", "http://localhost:8765/v1/identify")
PERSON_TOKEN = os.environ.get("PERSON_TOKEN", "")
PERSON_MIN_CONFIDENCE = float(os.environ.get("PERSON_MIN_CONFIDENCE", "0.55"))

# --- Agent backend (for execute tool / marker-based escalation) ---
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8642/v1/chat/completions")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", os.environ.get("API_SERVER_KEY", ""))
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4")
ENABLE_ESCALATION = os.environ.get("ENABLE_ESCALATION", "1") == "1"

# Escalation marker — LLM outputs this prefix when it needs the agent backend
ESCALATE_MARKER = "[ESCALATE"
_ESCALATE_RE = re.compile(r'\[ESCALATE\]')

# --- System prompt ---
_BASE_PROMPT = (
    "You are a helpful AI assistant with real-time vision capabilities. "
    "You can see what the user sees through their camera. "
    "Be concise, natural, and conversational — respond as if speaking aloud. "
    "Avoid markdown formatting. Keep responses brief unless asked for detail.\n\n"
)

_ESCALATION_RULES = (
    "\n\nESCALATION RULES (CRITICAL — always follow these):\n"
    "You are a small on-device model. You can ONLY converse, describe what you see, "
    "and answer from your own knowledge.\n"
    "You CANNOT: search the web, send emails/messages, set reminders, control devices, "
    "read files, run code, check the weather, access real-time info, or call any API.\n\n"
    "When the user asks for ANYTHING you cannot do, you MUST respond with ONLY this line:\n"
    "[ESCALATE] <one sentence describing what the user wants>\n\n"
    "RULES:\n"
    "- Output the marker and NOTHING else. No explanation.\n"
    "- NEVER output JSON, function calls, tool names, or code.\n"
    "- NEVER invent tool names like 'call_execute', 'send_email', etc.\n"
    "- NEVER output curly braces, angle brackets with function names, or XML tags.\n"
    "- Just the marker + plain English description. Examples:\n"
    "  User: 'Check my emails' → [ESCALATE] Check my unread emails\n"
    "  User: 'What's the weather?' → [ESCALATE] Check the current weather\n"
    "  User: 'Set a timer for 5 minutes' → [ESCALATE] Set a timer for 5 minutes\n"
    "  User: 'What is 2+2?' → 4 (no escalation needed)\n"
    "  User: 'Tell me a joke' → (respond normally, no escalation)"
)

# Track app-provided instructions + tools
_app_instructions = None
_app_tools = None

def build_system_prompt(base_instructions: str = None) -> str:
    """Build the system prompt. When tools are active (function-calling mode),
    use the app's instructions as-is. Otherwise append escalation rules."""
    base = base_instructions if base_instructions else _BASE_PROMPT
    if _app_tools:
        return base
    return base + _ESCALATION_RULES

SYSTEM_PROMPT = build_system_prompt()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gateway")

# --- Lazy-loaded models (global singletons) ---
_stt = None
_vad = None
_tts = None
_http = None

def get_http():
    global _http
    if _http is None:
        import httpx
        _http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _http

def init_models():
    """Initialize all models. Called once at startup."""
    global _stt, _vad, _tts

    log.info("Loading STT (%s backend: %s on %s)...", STT_BACKEND, STT_MODEL, STT_DEVICE)
    if STT_BACKEND == "transformers":
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
        _stt_dtype = torch.float16 if STT_DEVICE == "cuda" else torch.float32
        _stt_processor = AutoProcessor.from_pretrained(STT_MODEL)
        _stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            STT_MODEL,
            torch_dtype=_stt_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(STT_DEVICE)
        _stt = pipeline(
            "automatic-speech-recognition",
            model=_stt_model,
            tokenizer=_stt_processor,
            device=0 if STT_DEVICE == "cuda" else -1,
        )
        log.info("  ✓ STT ready (transformers/%s/%s)", STT_MODEL, STT_DEVICE)
    else:
        from faster_whisper import WhisperModel
        _stt_compute = os.environ.get("STT_COMPUTE", "int8")
        _stt = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=_stt_compute)
        log.info("  ✓ STT ready (faster-whisper/%s/%s/%s)", STT_MODEL, STT_DEVICE, _stt_compute)

    log.info("Loading VAD (Silero)...")
    from silero_vad import load_silero_vad, VADIterator
    _vad = load_silero_vad()
    log.info("  ✓ VAD ready")

    log.info("Loading TTS (Kokoro ONNX)...")
    from kokoro_onnx import Kokoro
    _tts = Kokoro(KOKORO_ONNX, KOKORO_VOICES)
    log.info("  ✓ TTS ready")

    log.info("All models loaded.")

def transcribe(audio_float32: np.ndarray) -> str:
    """Run STT on a complete speech segment. Blocking — call via to_thread."""
    if STT_BACKEND == "transformers":
        result = _stt(audio_float32, generate_kwargs={"language": "en"})
        return result["text"].strip()
    else:
        segments, _info = _stt.transcribe(audio_float32, beam_size=1, language="en", vad_filter=False)
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting that TTS would read as literal characters."""
    # Code blocks → "code block" placeholder (skip entirely, too garbled)
    text = re.sub(r'```[\s\S]*?```', ' (code omitted) ', text)
    # Inline code `like this`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Bold/italic markers **text**, __text__, *text*, _text_
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Headings # → remove entirely
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strikethrough ~~text~~
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # Images ![alt](url) → remove (must come before links!)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    # Links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Blockquotes >
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # List markers - * + → remove
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Numbered lists 1. → remove
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Horizontal rules --- *** ___
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    # Emojis with skin tones or ZWJ sequences — keep simple emoji, strip variation selectors
    text = re.sub(r'\uFE0F', '', text)
    # Multiple spaces → single
    text = re.sub(r' {2,}', ' ', text)
    # Clean up empty parentheses left by removed images
    text = re.sub(r'\(\s*\)', '', text)
    return text.strip()


def synthesize(text: str) -> Optional[np.ndarray]:
    """Run TTS on text. Returns int16 PCM at 24kHz. Blocking — call via to_thread."""
    text = clean_text_for_speech(text)
    if not text.strip():
        return None
    try:
        samples, sr = _tts.create(text, voice=TTS_VOICE, speed=1.0, lang="en-us")
        # Kokoro returns float32; convert to int16
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767).astype(np.int16)
    except Exception as e:
        log.error("TTS error: %s", e)
        return None


async def _stream_llm_raw(messages: list, tools: Optional[list], cancel_event: asyncio.Event):
    """Stream raw chunks from the LLM. Yields parsed deltas: {"role", "content"} or {"tool_calls": [...]}.
    Handles tool_call chunk accumulation for streaming tool calls."""
    client = get_http()
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    tool_call_buf: dict[int, dict] = {}  # index → {id, name, arguments}
    try:
        async with client.stream("POST", VLLM_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if cancel_event.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Text content
                    content = delta.get("content")
                    if content:
                        yield {"role": "assistant", "content": content}
                    # Tool calls
                    tc_deltas = delta.get("tool_calls") or []
                    for tc in tc_deltas:
                        idx = tc.get("index", 0)
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                        buf = tool_call_buf[idx]
                        if "id" in tc and tc["id"]:
                            buf["id"] = tc["id"]
                        if "function" in tc:
                            fn = tc["function"]
                            if "name" in fn and fn["name"]:
                                buf["name"] += fn["name"]
                            if "arguments" in fn:
                                buf["arguments"] += fn.get("arguments", "")
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error("LLM stream error: %s", e)
        yield {"role": "assistant", "content": f"[Error: {e}]"}

    # After stream ends, flush accumulated tool calls
    if tool_call_buf:
        yield {"tool_calls": [
            {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": b["arguments"]}}
            for b in tool_call_buf.values()
        ]}


async def stream_llm(messages: list, cancel_event: asyncio.Event) -> AsyncGenerator[str, None]:
    """Stream text-only tokens from the LLM."""
    async for event in _stream_llm_raw(messages, tools=None, cancel_event=cancel_event):
        if "content" in event:
            yield event["content"]


async def generate_llm(messages: list, cancel_event: asyncio.Event) -> str:
    """Non-streaming LLM call. Returns full text."""
    client = get_http()
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    try:
        resp = await client.post(VLLM_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""
    except Exception as e:
        log.error("LLM generate error: %s", e)
        return f"[Error: {e}]"


async def _stream_from_backend(
    url: str, token: str, model: str, task: str, conversation_history: list,
    extra_headers: dict = None,
) -> AsyncGenerator[str, None]:
    """Stream chunks from an OpenAI-compatible backend. Yields text as it arrives.
    Raises on connection/auth failure so the caller can handle it."""
    client = get_http()

    messages = []
    for msg in conversation_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": task})

    payload = {"model": model, "messages": messages, "stream": True}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers:
        headers.update(extra_headers)

    async with client.stream("POST", url, json=payload, headers=headers, timeout=180.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except json.JSONDecodeError:
                pass


async def escalate_to_agent_stream(
    task: str, conversation_history: list,
) -> AsyncGenerator[str, None]:
    """Forward a task to the agent backend. Yields text chunks as they arrive."""
    try:
        log.info("Escalating to agent: %s", task[:100])
        async for chunk in _stream_from_backend(
            AGENT_URL, AGENT_TOKEN, AGENT_MODEL, task, conversation_history,
        ):
            yield chunk
    except Exception as e:
        log.error("Agent escalation failed: %s", e)
        yield "Sorry, I couldn't reach the agent."


async def escalate_to_agent(task: str, conversation_history: list) -> str:
    """Forward a task to the agent backend. Returns the full text response."""
    full_text = ""
    async for chunk in escalate_to_agent_stream(task, conversation_history):
        full_text += chunk

    if full_text.strip():
        log.info("Agent response (%d chars): %s", len(full_text), full_text[:100])
        return full_text.strip()
    return "Sorry, I couldn't get a response from the agent."


async def identify_person(frame_b64: str) -> str:
    """Send the latest video frame to a face recognition API.
    Returns a natural-language string to speak."""
    client = get_http()
    headers = {"Content-Type": "application/json"}
    if PERSON_TOKEN:
        headers["Authorization"] = f"Bearer {PERSON_TOKEN}"

    payload = {
        "image": frame_b64,
        "min_confidence": PERSON_MIN_CONFIDENCE,
        "top_k": 3,
    }

    try:
        log.info("Identifying person via face recognition API...")
        resp = await client.post(PERSON_URL, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()

        matches = data.get("matches", [])
        no_match = data.get("no_match", False)

        if no_match or not matches:
            log.info("Person not recognized")
            return "I don't recognize them."

        top = matches[0]
        name = top.get("name", "")
        details = top.get("details", "")
        conf = top.get("confidence", 0.0)

        log.info("Person identified: %s (%.0f%%)", name, conf * 100)

        if details and details.strip():
            return f"This is {name}. {details}"
        return f"This is {name}."

    except Exception as e:
        log.error("Person identification error: %s", e)
        return "I couldn't reach the face recognition service."


# --- Sentence boundary detection ---
SENTENCE_END = re.compile(r'[.!?]\s|\n')

def extract_sentences(text: str):
    """Extract complete sentences from text. Returns (sentences_list, remaining_text)."""
    sentences = []
    pos = 0
    while True:
        match = SENTENCE_END.search(text, pos)
        if not match:
            break
        end = match.end()
        sentence = text[pos:end].strip()
        if sentence:
            sentences.append(sentence)
        pos = end
    return sentences, text[pos:]


# --- Session State Machine ---
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"

class VoiceSession:
    def __init__(self, ws):
        self.ws = ws
        self.state = STATE_IDLE
        self.audio_buffer = []  # accumulated float32 speech samples
        self.latest_frame_b64 = None  # latest video frame (base64 JPEG)
        self.cancel_event = asyncio.Event()
        self.vad_iterator = None
        self.client_sample_rate = INPUT_SAMPLE_RATE
        self.response_full_text = ""
        self.history = []  # conversation history: [{"role": "user/assistant", "content": "..."}]

    def reset_vad(self):
        from silero_vad import VADIterator
        self.vad_iterator = VADIterator(
            _vad,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_SILENCE_MS,
            speech_pad_ms=200,
        )
        self.audio_buffer = []

    async def send_json(self, data: dict):
        try:
            await self.ws.send_text(json.dumps(data))
        except Exception as e:
            log.warning("send_json failed: %s", e)

    async def send_audio(self, pcm_int16: np.ndarray):
        try:
            await self.ws.send_bytes(pcm_int16.tobytes())
        except Exception as e:
            log.warning("send_audio failed: %s", e)

    async def process_audio_chunk(self, raw_bytes: bytes):
        """Process a raw PCM audio chunk from the client."""
        if self.state == STATE_THINKING:
            return  # ignore audio while processing

        # Convert bytes to float32 numpy
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Process VAD in 512-sample chunks
        if not hasattr(self, '_vad_input_buf'):
            self._vad_input_buf = deque()

        self._vad_input_buf.extend(samples.tolist())

        import torch
        while len(self._vad_input_buf) >= VAD_CHUNK_SAMPLES:
            chunk = torch.FloatTensor([self._vad_input_buf.popleft() for _ in range(VAD_CHUNK_SAMPLES)])

            # Feed to VAD
            vad_result = self.vad_iterator(chunk, return_seconds=False)

            if vad_result is not None:
                if 'start' in vad_result:
                    # Speech detected
                    if self.state == STATE_SPEAKING:
                        # INTERRUPT: user started talking while AI is speaking
                        log.info("Interrupt detected — cancelling response")
                        self.cancel_event.set()
                        await self.send_json({"type": "interrupted"})
                        self.state = STATE_LISTENING
                        self.reset_vad()
                    elif self.state == STATE_IDLE:
                        self.state = STATE_LISTENING
                        log.info("Speech started")
                elif 'end' in vad_result:
                    # End of speech detected
                    if self.state == STATE_LISTENING:
                        self.audio_buffer.extend(chunk.tolist())
                        log.info("Speech ended — processing (%d samples)",
                                 len(self.audio_buffer))

                        # Trigger processing in background
                        audio_data = np.array(self.audio_buffer, dtype=np.float32)
                        self.audio_buffer = []
                        self.state = STATE_THINKING
                        self.reset_vad()

                        # Process asynchronously
                        asyncio.create_task(self.process_utterance(audio_data))
                        continue  # skip accumulation — already in THINKING
                    else:
                        self.reset_vad()
                        if self.state != STATE_SPEAKING:
                            self.state = STATE_IDLE
                        continue

            # Accumulate ALL audio during listening state (including start chunk)
            if self.state == STATE_LISTENING:
                self.audio_buffer.extend(chunk.tolist())

    async def process_utterance(self, audio_data: np.ndarray):
        """Process a complete speech segment: STT → LLM → TTS."""
        try:
            # 1. STT
            log.info("Running STT on %.1fs of audio...", len(audio_data) / INPUT_SAMPLE_RATE)
            text = await asyncio.to_thread(transcribe, audio_data)

            if not text:
                log.info("STT returned empty — ignoring")
                self.state = STATE_IDLE
                self.reset_vad()
                return

            log.info("STT: %s", text)
            await self.send_json({"type": "transcript", "text": text, "is_final": True})

            if self.cancel_event.is_set():
                self.state = STATE_IDLE
                self.reset_vad()
                return

            # 2. Build LLM messages (system + history + current turn)
            if ENABLE_VISION and self.latest_frame_b64:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.latest_frame_b64}"}},
                    {"type": "text", "text": text},
                ]
            else:
                content = text

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.history)  # previous turns (text-only)
            messages.append({"role": "user", "content": content})

            # 3. LLM → TTS pipeline
            await self._run_llm_pipeline(messages, text)

        except Exception as e:
            log.error("Process utterance error: %s\n%s", e, traceback.format_exc())
            await self.send_json({"type": "error", "message": str(e)})
        finally:
            self.state = STATE_IDLE
            self.reset_vad()
            log.info("Turn complete. State → idle")

    async def process_text_input(self, text: str):
        """Process text input (from app, not voice). Same pipeline minus STT."""
        try:
            log.info("Processing text input: %s", text[:80])
            await self.send_json({"type": "transcript", "text": text, "is_final": True})

            if self.cancel_event.is_set():
                self.state = STATE_IDLE
                self.reset_vad()
                return

            # Build LLM messages
            if ENABLE_VISION and self.latest_frame_b64:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.latest_frame_b64}"}},
                    {"type": "text", "text": text},
                ]
            else:
                content = text

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.history)
            messages.append({"role": "user", "content": content})

            await self._run_llm_pipeline(messages, text)

        except Exception as e:
            log.error("Text input error: %s\n%s", e, traceback.format_exc())
            await self.send_json({"type": "error", "message": str(e)})
        finally:
            self.state = STATE_IDLE
            self.reset_vad()
            log.info("Text turn complete. State → idle")

    async def _run_llm_pipeline(self, messages: list, user_text: str):
        """LLM → TTS pipeline. Two modes:
        - Function calling: when the app sends tools via session.update.
        - Marker routing: fallback when no tools configured."""

        async def _stream_tts(token_gen, initial_buffer=""):
            """Consume a token generator, extract sentences, and TTS in real-time."""
            text_buffer = initial_buffer
            self.response_full_text += initial_buffer
            await self.send_json({"type": "response_text", "text": self.response_full_text})
            sentences, text_buffer = extract_sentences(text_buffer)
            for sentence in sentences:
                if self.cancel_event.is_set():
                    break
                audio = await asyncio.to_thread(synthesize, sentence)
                if audio is not None and not self.cancel_event.is_set():
                    await self.send_audio(audio)
            async for token in token_gen:
                if self.cancel_event.is_set():
                    break
                text_buffer += token
                self.response_full_text += token
                await self.send_json({"type": "response_text", "text": self.response_full_text})
                sentences, text_buffer = extract_sentences(text_buffer)
                for sentence in sentences:
                    if self.cancel_event.is_set():
                        break
                    audio = await asyncio.to_thread(synthesize, sentence)
                    if audio is not None and not self.cancel_event.is_set():
                        await self.send_audio(audio)
            if text_buffer.strip() and not self.cancel_event.is_set():
                audio = await asyncio.to_thread(synthesize, text_buffer.strip())
                if audio is not None:
                    await self.send_audio(audio)

        def _save_history():
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": self.response_full_text})
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

        async def _execute_tool_on_backend(name: str, args: dict) -> str:
            """Execute a single tool call. Returns result text."""
            if name == "identify_person":
                query = args.get("query", "")
                log.info("Tool: identify_person(%s)", query[:60])
                if not self.latest_frame_b64:
                    return "No camera frame available. Ask the user to enable their camera."
                return await identify_person(self.latest_frame_b64)
            elif name == "execute":
                query = args.get("query", "") or user_text  # fallback to original user speech if LLM omitted args
                log.info("Tool: execute(%s)", query[:80])
                result = await escalate_to_agent(query, self.history)
                return result
            else:
                log.warning("Unknown tool: %s", name)
                return f"Unknown tool: {name}"

        # --- Start of pipeline ---
        self.state = STATE_SPEAKING
        self.cancel_event.clear()
        self.response_full_text = ""
        await self.send_json({"type": "response_start"})

        tools = _app_tools  # module-level, set by session.update

        if not ENABLE_ESCALATION and not tools:
            # No escalation, no tools — pure streaming
            await _stream_tts(stream_llm(messages, self.cancel_event))
            if not self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                _save_history()
            return

        if tools:
            # ── FUNCTION CALLING MODE ──
            # Send to LLM with tools, collect events
            events = []
            async for event in _stream_llm_raw(messages, tools, self.cancel_event):
                if self.cancel_event.is_set():
                    break
                events.append(event)

            tool_calls = []
            text_content = ""
            for event in events:
                if "tool_calls" in event:
                    tool_calls = event["tool_calls"]
                elif "content" in event:
                    text_content += event["content"]

            if tool_calls:
                ack = "Let me check that for you."
                self.response_full_text = ack
                await self.send_json({"type": "response_text", "text": ack})
                audio = await asyncio.to_thread(synthesize, ack)
                if audio is not None and not self.cancel_event.is_set():
                    await self.send_audio(audio)
                if self.cancel_event.is_set():
                    await self.send_json({"type": "response_end"})
                    return

                # Parse tool calls
                import uuid as _uuid
                fc_list = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    call_id = f"tool-{_uuid.uuid4().hex[:8]}"
                    fc_list.append({"name": name, "args": args, "call_id": call_id})

                # Notify client: tools are running (for UI)
                await self.send_json({
                    "toolCall": {
                        "functionCalls": [
                            {"id": fc["call_id"], "name": fc["name"], "args": fc["args"]}
                            for fc in fc_list
                        ]
                    }
                })
                log.info("Sent toolCall to client: %s", [fc["name"] for fc in fc_list])

                # Execute tools immediately
                tool_results = []
                for fc in fc_list:
                    result = await _execute_tool_on_backend(fc["name"], fc["args"])
                    log.info("Executed '%s' → %s", fc["name"], str(result)[:80])
                    # Notify client: tool done
                    await self.send_json({
                        "toolResponse": {
                            "functionResponses": [
                                {"id": fc["call_id"], "name": fc["name"],
                                 "response": {"result": result[:500]}}
                            ]
                        }
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": fc["call_id"],
                        "content": result,
                    })

                # Build a clean follow-up that avoids tool-message format entirely.
                # Some LLMs (e.g. Gemma) can't handle tool-role messages well: they
                # either loop (calling execute again) or refuse ("I cannot see anyone")
                # for visual tools. Reframe tool results as a plain user message so
                # the LLM just relays the information conversationally.
                tool_summary = " ".join(r["content"] for r in tool_results)
                followup_messages = [
                    {"role": "system", "content": "You are a helpful voice assistant. Relay information from your tools to the user naturally and concisely."},
                    *self.history,  # previous turns so pronouns/context resolve across tool calls
                    {"role": "user", "content": f'The user asked: "{user_text}". Here is the result: "{tool_summary}". Respond to the user naturally.'},
                ]

                # Final LLM call for the spoken response (NO tools)
                # Accumulate text in a buffer across chunks so sentences that
                # span multiple streaming chunks are detected correctly.
                final_text = ""
                first_chunk = True
                text_buffer = ""  # accumulates incomplete sentence fragments
                async for event in _stream_llm_raw(followup_messages, None, self.cancel_event):
                    if self.cancel_event.is_set():
                        break
                    content = event.get("content", "")
                    if content:
                        # Some LLMs prepend "thought\n" — strip from first chunk
                        if first_chunk:
                            first_chunk = False
                            content = re.sub(r'^thought\n', '', content)
                        if not content.strip():
                            continue
                        final_text += content
                        self.response_full_text = ack + " " + final_text
                        await self.send_json({"type": "response_text", "text": self.response_full_text})
                        text_buffer += content
                        sentences, text_buffer = extract_sentences(text_buffer)
                        for sentence in sentences:
                            if self.cancel_event.is_set():
                                break
                            audio = await asyncio.to_thread(synthesize, sentence)
                            if audio is not None and not self.cancel_event.is_set():
                                await self.send_audio(audio)

                # Flush any remaining unspoken text
                if text_buffer.strip() and not self.cancel_event.is_set():
                    log.info("Flushing remaining TTS: %s", text_buffer.strip()[:80])
                    audio = await asyncio.to_thread(synthesize, text_buffer.strip())
                    if audio is not None and not self.cancel_event.is_set():
                        await self.send_audio(audio)

                if not self.cancel_event.is_set():
                    await self.send_json({"type": "response_end"})
                    _save_history()
                return

            # No tool calls — stream text response directly
            if text_content:
                async def _iter(s):
                    yield s
                await _stream_tts(_iter(text_content))
            if not self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                _save_history()
            return

        # ── MARKER-BASED ROUTING (fallback, no tools) ──
        if not ENABLE_ESCALATION:
            await _stream_tts(stream_llm(messages, self.cancel_event))
            if not self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                _save_history()
            return

        token_gen = stream_llm(messages, self.cancel_event)
        PEEK_THRESHOLD = 22
        peek_buf = ""
        async for token in token_gen:
            if self.cancel_event.is_set():
                break
            peek_buf += token
            if len(peek_buf) >= PEEK_THRESHOLD:
                break

        if self.cancel_event.is_set():
            await self.send_json({"type": "response_end"})
            return

        if ESCALATE_MARKER in peek_buf:
            m = _ESCALATE_RE.search(peek_buf)
            if m:
                task_desc = peek_buf[m.end():].strip()
            else:
                task_desc = peek_buf
                async for token in token_gen:
                    if self.cancel_event.is_set():
                        break
                    task_desc += token
                    m2 = _ESCALATE_RE.search(task_desc)
                    if m2:
                        task_desc = task_desc[m2.end():].strip()
                        break
            # Consume remaining tokens
            if m:
                async for token in token_gen:
                    if self.cancel_event.is_set():
                        break
                    task_desc += token
            if self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                return
            log.info("Escalating: %s", task_desc[:80])
            await self.send_json({"type": "escalating", "task": task_desc[:100]})
            ack = "Let me check that for you."
            self.response_full_text = ack + " "
            await self.send_json({"type": "response_text", "text": ack})
            audio = await asyncio.to_thread(synthesize, ack)
            if audio is not None and not self.cancel_event.is_set():
                await self.send_audio(audio)
            if self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                return
            agent_text = ""
            tts_buffer = ""
            async for chunk in escalate_to_agent_stream(task_desc, self.history):
                if self.cancel_event.is_set():
                    break
                tts_buffer += chunk
                agent_text += chunk
                self.response_full_text = ack + " " + agent_text
                await self.send_json({"type": "response_text", "text": self.response_full_text})
                sentences, tts_buffer = extract_sentences(tts_buffer)
                for sentence in sentences:
                    if self.cancel_event.is_set():
                        break
                    audio = await asyncio.to_thread(synthesize, sentence)
                    if audio is not None and not self.cancel_event.is_set():
                        await self.send_audio(audio)
            if tts_buffer.strip() and not self.cancel_event.is_set():
                audio = await asyncio.to_thread(synthesize, tts_buffer.strip())
                if audio is not None:
                    await self.send_audio(audio)
            if not self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                _save_history()
            return

        else:
            # No marker — normal streaming
            log.info("LLM streaming (no escalation marker)")
            await _stream_tts(token_gen, initial_buffer=peek_buf)
            if not self.cancel_event.is_set():
                await self.send_json({"type": "response_end"})
                _save_history()
            else:
                log.info("Turn interrupted — not saving to history")


# --- FastAPI App ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path

async def warmup_agent():
    """Send a tiny request to warm up the agent session (avoids cold start)."""
    try:
        log.info("Warming up agent session...")
        result = await escalate_to_agent("Hello", [])
        log.info("Agent warm-up complete: %s", result[:60])
    except Exception as e:
        log.warning("Agent warm-up failed (will retry on first use): %s", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Initializing models...")
    init_models()
    log.info("Gateway ready!")

    if ENABLE_ESCALATION:
        asyncio.create_task(warmup_agent())

    yield
    log.info("Shutting down...")

app = FastAPI(title="Local Multimodal LLM Simulation", lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "stt_model": STT_MODEL,
        "tts_voice": TTS_VOICE,
        "llm_model": VLLM_MODEL,
        "llm_endpoint": VLLM_URL,
        "escalation": "enabled" if ENABLE_ESCALATION else "disabled",
        "agent_url": AGENT_URL if ENABLE_ESCALATION else None,
        "agent_model": AGENT_MODEL if ENABLE_ESCALATION else None,
    }

WEB_APP_DIR = Path(__file__).parent / "web"

@app.get("/", response_class=HTMLResponse)
async def web_app():
    """Serve the test web console."""
    html_path = WEB_APP_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Web app not found</h1>", status_code=404)

@app.get("/info")
async def root():
    return {
        "service": "Local Multimodal LLM Simulation",
        "websocket": f"ws://{HOST}:{PORT}/realtime",
        "protocol": {
            "audio_in": "16kHz 16-bit mono PCM (binary frames)",
            "audio_out": "24kHz 16-bit mono PCM (binary frames)",
            "video": "JSON: {type: video_frame, data: base64 JPEG}",
            "events": ["transcript", "response_start", "response_text", "response_end", "interrupted", "error", "toolCall", "toolResponse"],
        }
    }

@app.websocket("/realtime")
async def realtime_endpoint(ws: WebSocket):
    await ws.accept()
    session = VoiceSession(ws)
    session.reset_vad()
    session._vad_input_buf = deque()
    log.info("Client connected")

    try:
        # Send hello
        await session.send_json({
            "type": "connected",
            "audio_format": {"sample_rate": OUTPUT_SAMPLE_RATE, "bits": 16, "channels": 1},
            "message": "Gateway ready. Send audio (16kHz PCM) and video frames."
        })

        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.receive":
                if "bytes" in msg and msg["bytes"]:
                    # Binary audio frame
                    await session.process_audio_chunk(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    # JSON control message
                    try:
                        data = json.loads(msg["text"])
                        msg_type = data.get("type")

                        if msg_type == "video_frame":
                            session.latest_frame_b64 = data.get("data")
                            log.debug("Video frame updated")

                        elif msg_type == "config":
                            session.client_sample_rate = data.get("sample_rate", INPUT_SAMPLE_RATE)
                            log.info("Client config: %s", data)

                        elif msg_type == "ping":
                            await session.send_json({"type": "pong", "t": time.time()})

                        # --- Client session events ---
                        elif msg_type == "session.update":
                            sess = data.get("session", {})
                            instructions = sess.get("instructions")

                            app_tools = sess.get("tools", [])
                            if app_tools:
                                global _app_tools
                                _app_tools = app_tools
                                log.info("Accepted %d tool(s) from client for function-calling mode", len(app_tools))
                            elif "tools" in sess:
                                _app_tools = None
                                log.info("Client cleared tools — reverting to marker-based routing")

                            if instructions:
                                global SYSTEM_PROMPT, _app_instructions
                                _app_instructions = instructions
                                SYSTEM_PROMPT = build_system_prompt(instructions)
                                log.info("Session instructions updated from client")

                        elif msg_type == "conversation.item.create":
                            item = data.get("item", {})
                            item_type = item.get("type")

                            if item_type == "message" and item.get("role") == "user":
                                # Text input from client (not voice)
                                content_parts = item.get("content", [])
                                text_input = " ".join(
                                    c.get("text", "") for c in content_parts if c.get("type") == "text"
                                )
                                if text_input.strip():
                                    log.info("Text input from client: %s", text_input[:80])
                                    session.state = STATE_THINKING
                                    asyncio.create_task(session.process_text_input(text_input))

                        elif msg_type == "response.cancel":
                            # Client-requested interruption
                            session.cancel_event.set()
                            await session.send_json({"type": "interrupted"})
                            session.state = STATE_IDLE
                            session.reset_vad()
                            log.info("Response cancelled by client")

                        elif msg_type == "input_audio_buffer.append":
                            # Alternative audio format from client (base64 in JSON)
                            audio_b64 = data.get("audio", "")
                            if audio_b64:
                                raw = base64.b64decode(audio_b64)
                                await session.process_audio_chunk(raw)

                        elif msg_type == "input_video_frame.append":
                            # Client sends video frame (base64 JPEG in JSON)
                            img = data.get("image", "")
                            if img:
                                session.latest_frame_b64 = img
                                log.debug("Video frame from client")

                    except json.JSONDecodeError:
                        log.warning("Invalid JSON from client")

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error("WebSocket error: %s\n%s", e, traceback.format_exc())
    finally:
        session.cancel_event.set()
        log.info("Session ended")


if __name__ == "__main__":
    import uvicorn
    ssl_cert = os.environ.get("SSL_CERT", "./certs/cert.pem")
    ssl_key = os.environ.get("SSL_KEY", "./certs/key.pem")
    print(f"\n{'='*60}", flush=True)
    print(f"🎙️  Local Multimodal LLM Simulation", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Web App:    https://0.0.0.0:{PORT}/", flush=True)
    print(f"  WebSocket:  wss://0.0.0.0:{PORT}/realtime", flush=True)
    print(f"  Health:     https://0.0.0.0:{PORT}/health", flush=True)
    print(f"  ─────────────────────────────────────────", flush=True)
    print(f"  LLM:        {VLLM_MODEL} @ {VLLM_URL}", flush=True)
    print(f"  STT:        {STT_BACKEND}/{STT_MODEL} ({STT_DEVICE})", flush=True)
    print(f"  TTS:        Kokoro ({TTS_VOICE})", flush=True)
    print(f"  VAD:        Silero (threshold={VAD_THRESHOLD}, silence={VAD_SILENCE_MS}ms)", flush=True)
    print(f"  Vision:     {'enabled' if ENABLE_VISION else 'disabled'}", flush=True)
    print(f"  History:    {MAX_HISTORY_TURNS} turns", flush=True)
    print(f"  Escalation: {'→ agent @ ' + AGENT_URL if ENABLE_ESCALATION else 'disabled'}", flush=True)
    print(f"{'='*60}\n", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info",
                ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)
