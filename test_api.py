"""Make one test request and display its reply and token usage."""

import llm


try:
    reply = llm.ask_llm(
        messages=[
            {
                "role": "user",
                "parts": [{"text": "Explain what an API is in one sentence."}],
            }
        ],
        system_prompt="Answer clearly and briefly.",
        temperature=0.2,
        max_tokens=100,
    )

    print("Reply:")
    print(reply)

    usage = llm.last_response.usage_metadata
    print("\nToken usage:")
    print(f"Prompt tokens: {usage.prompt_token_count}")
    print(f"Output tokens: {usage.candidates_token_count}")
    print(f"Total tokens: {usage.total_token_count}")
except RuntimeError as error:
    print(f"Could not call Gemini: {error}")
