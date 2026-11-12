"""Small helper for sending chat messages to Gemini."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

# test_api.py reads this after ask_llm finishes to display token usage.
last_response = None


def ask_llm(messages, system_prompt, temperature, max_tokens):
    """Send messages to Gemini and return only the generated reply text."""

    global last_response

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file and try again."
        )

    # The SDK accepts the conversation as dictionaries with user/model roles.
    try:
        client = genai.Client(api_key=api_key)
        last_response = client.models.generate_content(
            model=model_name,
            contents=messages,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
    except errors.APIError as error:
        raise RuntimeError(
            f"Gemini could not complete the request. Please check your API key "
            f"and connection. Details: {error}"
        ) from error

    if not last_response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return last_response.text


def ask_llm_timed(
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Wrap a single-turn Gemini call and return reply + timing + token stats.

    Parameters
    ----------
    system_prompt:
        Text for the system role.  Pass an empty string (or None) for
        strategies that don't use one.
    user_message:
        Text for the user role.
    temperature:
        Sampling temperature forwarded to the model.
    max_tokens:
        Maximum output tokens forwarded to the model.

    Returns
    -------
    On success::

        {
            "reply":           str,   # model's text response
            "latency_seconds": float, # wall-clock time for the API call only
            "prompt_tokens":   int,
            "output_tokens":   int,
            "total_tokens":    int,
        }

    On failure::

        {
            "error": str,  # human-friendly message; never raises
        }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

    if not api_key:
        return {"error": "GEMINI_API_KEY is missing. Add it to your .env file and try again."}

    client = genai.Client(api_key=api_key)
    contents = [{"role": "user", "parts": [{"text": user_message}]}]
    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    # Only attach system_instruction when the strategy actually provides one.
    if system_prompt:
        config["system_instruction"] = system_prompt

    try:
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        latency = time.perf_counter() - t0
    except errors.APIError as exc:
        # 429 → rate-limited; other codes → surface a friendly message.
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status == 429:
            return {
                "error": (
                    "Rate limit reached (HTTP 429). "
                    "Wait a moment and try again, or reduce request frequency."
                )
            }
        return {
            "error": (
                f"The API returned an error (status {status}). "
                f"Check your key/connection. Details: {exc}"
            )
        }
    except Exception as exc:  # network timeouts, etc.
        return {"error": f"Unexpected error during API call: {exc}"}

    reply = getattr(response, "text", None) or ""
    if not reply:
        return {"error": "Gemini returned an empty response."}

    # Token counts live under usage_metadata; default to 0 if absent.
    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens  = getattr(usage, "prompt_token_count",     0) or 0
    output_tokens  = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens   = getattr(usage, "total_token_count",      0) or 0

    return {
        "reply":           reply,
        "latency_seconds": round(latency, 3),
        "prompt_tokens":   prompt_tokens,
        "output_tokens":   output_tokens,
        "total_tokens":    total_tokens,
    }



def ask_llm_stream(messages, system_prompt, temperature, max_tokens):
    """Send messages to Gemini and yield generated text chunks as they arrive."""

    global last_response

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file and try again."
        )

    try:
        client = genai.Client(api_key=api_key)
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=messages,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        for chunk in response_stream:
            last_response = chunk
            if chunk.text:
                yield chunk.text
    except errors.APIError as error:
        raise RuntimeError(
            f"Gemini could not complete the request. Please check your API key "
            f"and connection. Details: {error}"
        ) from error
