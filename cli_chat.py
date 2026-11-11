"""Terminal chat interface using Gemini with conversation history management."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure the script's directory is in sys.path and .env is loaded regardless of CWD
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
load_dotenv(CURRENT_DIR / ".env")

from llm import ask_llm

# =============================================================================
# CONFIGURATION SETTINGS (Edit these to experiment with model behavior)
# =============================================================================

# 1. USE_HISTORY (bool)
#    - True : Sends full conversation history every turn, enabling multi-turn memory.
#    - False: Sends only the latest user message, demonstrating that the API is
#             stateless and has no memory of prior turns.
USE_HISTORY = True

# 2. SYSTEM_PROMPT (str)
#    Sets the overarching persona, tone, rules, and role for the model before it
#    reads any user messages.
#
# --- Example System Prompts (Swap or uncomment one below): ---
#
# [Example A: Coding Tutor]
# SYSTEM_PROMPT = "You are a patient, encouraging coding tutor. Help the user learn by providing clear explanations, guiding questions, and concise, well-commented code examples."
#
# [Example B: Concise Assistant]
# SYSTEM_PROMPT = "You are an ultra-concise assistant. Answer questions directly in as few words or bullet points as possible without pleasantries or filler."
#
# [Example C: Pirate]
SYSTEM_PROMPT = "You are a salty 17th-century pirate captain aboard the high seas. Speak entirely in pirate slang, call the user 'matey', and weave seafaring metaphors into all your answers, arrr!"
#
#SYSTEM_PROMPT = "You are a patient, encouraging coding tutor. Help the user learn by providing clear explanations, guiding questions, and concise, well-commented code examples."

# 3. TEMPERATURE (float)
#    Controls the randomness and creativity of the generated text (typically 0.0 to 2.0).
#    - Low (0.0 - 0.3): Deterministic, focused, and factual. Ideal for coding, math, and data extraction.
#    - Medium (0.7): Balanced creativity and coherence; standard for general conversation.
#    - High (1.0+): Highly creative, diverse, and unpredictable; higher risk of hallucinations.
TEMPERATURE = 1

# 4. MAX_TOKENS (int)
#    The maximum number of output tokens (word fragments) the model can generate in a single reply.
#    - Controls response length, saves tokens/cost, and prevents runaway responses.
#    - If set too low, answers may be cut off mid-sentence.
MAX_TOKENS = 10

# =============================================================================
# WHY WE MUST RE-SEND THE CONVERSATION HISTORY:
# =============================================================================
# LLM APIs (such as Gemini, OpenAI, and Anthropic) are completely stateless.
# They operate over standard REST/HTTP requests and do not maintain any session
# state, memory, or context on their servers between calls.
#
# Every time ask_llm() is called:
#   1. The API processes only the messages sent in that individual request.
#   2. Once the response is generated, the server discards the interaction.
#
# Therefore, for a multi-turn conversation to feel continuous and coherent,
# the client application (this script) is responsible for storing the conversation
# history and re-sending the full list of past user and model messages with each
# new prompt.
# =============================================================================


def main():
    """Run interactive terminal chat loop."""
    history = []

    print("==================================================")
    print("           Interactive Gemini CLI Chat            ")
    print("==================================================")
    print(f"USE_HISTORY   : {USE_HISTORY}")
    print(f"TEMPERATURE   : {TEMPERATURE}")
    print(f"MAX_TOKENS    : {MAX_TOKENS}")
    prompt_preview = (
        f'"{SYSTEM_PROMPT[:65]}..."'
        if len(SYSTEM_PROMPT) > 65
        else f'"{SYSTEM_PROMPT}"'
    )
    print(f"SYSTEM_PROMPT : {prompt_preview}")
    if not USE_HISTORY:
        print("\n[Notice: History disabled. The model will not remember previous messages.]")
    print("\nType 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Exiting chat. Goodbye!")
            break

        # Format user message for the API
        user_message = {
            "role": "user",
            "parts": [{"text": user_input}],
        }
        history.append(user_message)

        # When USE_HISTORY is True, send the full history.
        # When False, send only the latest user message so the model has no memory.
        messages_to_send = history if USE_HISTORY else [user_message]

        try:
            reply = ask_llm(
                messages=messages_to_send,
                system_prompt=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except RuntimeError as error:
            print(f"\n[Error calling Gemini: {error}]\n")
            continue

        print(f"\nModel: {reply}\n")

        # Append model response to keep the conversation history complete
        model_message = {
            "role": "model",
            "parts": [{"text": reply}],
        }
        history.append(model_message)


if __name__ == "__main__":
    main()
