"""FastAPI service for the Gemini LLM Chatbot."""

import json
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure local directory is in path and server loads .env privately
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
load_dotenv(CURRENT_DIR / ".env")

import llm
from llm import ask_llm, ask_llm_stream

STATIC_DIR = CURRENT_DIR / "static"

# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================
#
# WHY TRIMMING IS NECESSARY — THE CONTEXT WINDOW
# ───────────────────────────────────────────────
# Every LLM has a fixed "context window": the maximum number of tokens it can
# read in a single call.  For Gemini 1.5 Flash the limit is ~1 million tokens,
# but even generous limits matter because:
#   • Longer prompts cost more (latency + API quota).
#   • At some point the model starts to "forget" or confuse early turns.
#   • Unbounded history means unbounded memory usage on the client.
#
# TRADE-OFF
# ─────────
#   Keeping more history → better coherence, the model remembers older facts.
#   Dropping old messages → cheaper, faster, avoids hitting the context limit.
#
# STRATEGY USED HERE
# ──────────────────
# If the number of messages sent by the client exceeds MAX_HISTORY_MESSAGES,
# we drop the OLDEST messages from the front of the list.  We always remove
# in pairs (user + model) to avoid leaving the history in an invalid state
# (Gemini requires the first message to have role "user").
#
# The system prompt is passed separately and is NEVER touched by this trim.

MAX_HISTORY_MESSAGES = 20  # maximum number of messages (user + model turns)

app = FastAPI(
    title="Gemini Chatbot API",
    description="A multi-turn chat API powered by Google Gemini with conversation history, streaming, and token usage.",
    version="1.0.0",
)

# Mount static files directory if present
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class ChatMessage(BaseModel):
    """Single message in a conversation."""
    role: str = Field(
        ...,
        description="Role of the message sender ('user', 'model', or 'assistant')",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Text content of the message",
        examples=["Hello! What is an API?"],
    )


class ChatRequest(BaseModel):
    """Request payload for /chat and /chat/stream."""
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of messages representing conversation history",
        examples=[
            [
                {"role": "user", "content": "Hello, my name is Alice."},
                {"role": "model", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ]
        ],
    )
    system_prompt: Optional[str] = Field(
        default="You are a helpful and concise assistant.",
        description="Overarching system instruction / persona for the model",
        examples=["You are a helpful and concise assistant."],
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Creativity / randomness parameter (0.0 = deterministic, 2.0 = highly creative)",
        examples=[0.7],
    )
    max_tokens: Optional[int] = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate in the reply",
        examples=[1000],
    )


class UsageMetadata(BaseModel):
    """Token usage breakdown."""
    prompt_tokens: Optional[int] = Field(None, description="Tokens used in prompt/input")
    completion_tokens: Optional[int] = Field(None, description="Tokens generated in output")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed")


class ChatResponse(BaseModel):
    """Response payload returned by POST /chat."""
    reply: str = Field(..., description="Generated text response from the model")
    usage: UsageMetadata = Field(..., description="Token usage details")
    messages_dropped: int = Field(
        0,
        description=(
            "Number of messages removed from the front of the history before "
            "calling Gemini because the conversation exceeded MAX_HISTORY_MESSAGES."
        ),
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Server status indicator")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_messages_for_gemini(messages: List[ChatMessage]) -> list:
    """Map incoming [{role, content}] to Gemini's [{'role': ..., 'parts': [{'text': ...}]}]."""
    gemini_messages = []
    for msg in messages:
        # Normalize "assistant" to "model" for Gemini
        role = "model" if msg.role.lower() == "assistant" else msg.role.lower()
        gemini_messages.append({
            "role": role,
            "parts": [{"text": msg.content}],
        })
    return gemini_messages


def _trim_history(messages: List[ChatMessage]) -> tuple[List[ChatMessage], int]:
    """Drop the oldest messages when the history exceeds MAX_HISTORY_MESSAGES.

    Rules
    ─────
    • The system prompt is passed separately and is never touched here.
    • We drop from the FRONT of the list (oldest messages first).
    • We always drop in pairs (user + following model) so the resulting list
      still starts with a "user" role, which Gemini requires.  If the list
      has an odd length after a pair-wise trim we drop one more message to
      enforce the invariant.

    Returns
    ───────
    (trimmed_messages, number_of_messages_dropped)
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages, 0

    excess = len(messages) - MAX_HISTORY_MESSAGES

    # Round excess up to the nearest even number so we always remove whole
    # user+model pairs and never strand a lone model reply at the front.
    if excess % 2 != 0:
        excess += 1

    trimmed = messages[excess:]

    # Paranoia guard: if the first remaining message is not a user turn, drop
    # one more to restore the invariant (can happen with malformed history).
    while trimmed and trimmed[0].role.lower() not in ("user",):
        trimmed = trimmed[1:]
        excess += 1

    return trimmed, excess


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", summary="Serve Chat Web UI", tags=["UI"])
def index():
    """Serve the single-file chat web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "UI file static/index.html not found."},
        )
    return FileResponse(str(index_file), media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        429: {"description": "Rate limit exceeded or quota exhausted"},
        500: {"description": "Server configuration / API key error"},
        502: {"description": "Gemini API error or bad response"},
    },
    tags=["Chat"],
)
def chat(request: ChatRequest):
    """Send conversation messages to Gemini and return the full reply with token usage."""
    # Trim history before formatting; record how many messages were dropped.
    trimmed_messages, messages_dropped = _trim_history(request.messages)
    gemini_messages = _format_messages_for_gemini(trimmed_messages)

    try:
        reply_text = ask_llm(
            messages=gemini_messages,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except RuntimeError as error:
        err_str = str(error)
        cause = getattr(error, "__cause__", None)
        cause_code = getattr(cause, "code", None)
        cause_str = str(cause) if cause else ""
        combined_err = f"{err_str} {cause_str}".lower()

        # Handle rate limiting (HTTP 429) / quota exhaustion
        if (
            cause_code == 429
            or "429" in combined_err
            or "resource_exhausted" in combined_err
            or "quota" in combined_err
            or "rate limit" in combined_err
        ):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": "Gemini rate limit or quota exceeded. Please wait a moment and try again.",
                    "detail": err_str,
                },
            )

        # Handle missing or invalid API key configuration
        if "gemini_api_key is missing" in combined_err or "api_key not valid" in combined_err:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "ConfigurationError",
                    "message": "GEMINI_API_KEY is not properly configured on the server. Check the .env file.",
                    "detail": err_str,
                },
            )

        # Handle any other Gemini API error
        status_code = status.HTTP_502_BAD_GATEWAY
        if isinstance(cause_code, int) and 400 <= cause_code < 600:
            status_code = cause_code

        return JSONResponse(
            status_code=status_code,
            content={
                "error": "GeminiError",
                "message": "An error occurred while calling the Gemini API.",
                "detail": err_str,
            },
        )

    # Extract token usage metadata from the global last_response recorded in llm.py
    last_resp = getattr(llm, "last_response", None)
    metadata = getattr(last_resp, "usage_metadata", None)

    usage = UsageMetadata(
        prompt_tokens=getattr(metadata, "prompt_token_count", None),
        completion_tokens=getattr(metadata, "candidates_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
    )

    return ChatResponse(reply=reply_text, usage=usage, messages_dropped=messages_dropped)


@app.post(
    "/chat/stream",
    summary="Stream chat responses using Server-Sent Events (SSE)",
    responses={
        200: {
            "description": "Server-Sent Events (SSE) stream of text chunks followed by final token usage.",
            "content": {
                "text/event-stream": {
                    "example": (
                        "data: {\"chunk\": \"Hello\"}\n\n"
                        "data: {\"chunk\": \" world!\"}\n\n"
                        "event: usage\n"
                        "data: {\"prompt_tokens\": 5, \"completion_tokens\": 3, \"total_tokens\": 8, \"messages_dropped\": 0}\n\n"
                    )
                }
            },
        },
        429: {"description": "Rate limit exceeded or quota exhausted"},
        500: {"description": "Server configuration / API key error"},
        502: {"description": "Gemini API error or bad response"},
    },
    tags=["Chat"],
)
def chat_stream(request: ChatRequest):
    """Stream conversation responses from Gemini using Server-Sent Events (SSE).

    Each text chunk is yielded in real-time as an SSE event:
        data: {"chunk": "..."}

    When the response finishes, a final event is emitted containing token usage
    and the number of messages that were trimmed from the history:
        event: usage
        data: {"prompt_tokens": 12, "completion_tokens": 20, "total_tokens": 32,
               "messages_dropped": 0}
    """
    # Trim history before formatting; record how many messages were dropped.
    trimmed_messages, messages_dropped = _trim_history(request.messages)
    gemini_messages = _format_messages_for_gemini(trimmed_messages)

    def event_generator():
        try:
            for chunk in ask_llm_stream(
                messages=gemini_messages,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                payload = json.dumps({"chunk": chunk})
                yield f"data: {payload}\n\n"

            # Emit final event containing token usage and trim metadata
            last_resp = getattr(llm, "last_response", None)
            metadata = getattr(last_resp, "usage_metadata", None)
            usage_data = {
                "prompt_tokens": getattr(metadata, "prompt_token_count", None),
                "completion_tokens": getattr(metadata, "candidates_token_count", None),
                "total_tokens": getattr(metadata, "total_token_count", None),
                "messages_dropped": messages_dropped,
            }
            yield f"event: usage\ndata: {json.dumps(usage_data)}\n\n"

        except RuntimeError as error:
            err_str = str(error)
            cause = getattr(error, "__cause__", None)
            cause_code = getattr(cause, "code", None)
            cause_str = str(cause) if cause else ""
            combined_err = f"{err_str} {cause_str}".lower()

            if (
                cause_code == 429
                or "429" in combined_err
                or "resource_exhausted" in combined_err
                or "quota" in combined_err
                or "rate limit" in combined_err
            ):
                err_payload = {
                    "error": "RateLimitExceeded",
                    "message": "Gemini rate limit or quota exceeded. Please wait a moment and try again.",
                    "detail": err_str,
                }
            elif "gemini_api_key is missing" in combined_err or "api_key not valid" in combined_err:
                err_payload = {
                    "error": "ConfigurationError",
                    "message": "GEMINI_API_KEY is not properly configured on the server.",
                    "detail": err_str,
                }
            else:
                err_payload = {
                    "error": "GeminiError",
                    "message": "An error occurred while streaming from Gemini API.",
                    "detail": err_str,
                }
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
