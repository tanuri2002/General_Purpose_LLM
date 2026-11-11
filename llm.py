"""Small helper for sending chat messages to Gemini."""

import os

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
