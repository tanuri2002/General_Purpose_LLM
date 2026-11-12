"""
compare_cli.py – Run every prompting strategy against a user task and
print a side-by-side summary in the terminal.

Usage:
    python compare_cli.py

The script prompts for a task interactively, then runs each strategy in
STRATEGIES order with a short pause between calls to respect rate limits.
"""

import sys
import textwrap
import time

from prompts import STRATEGIES, build_prompt
from llm import ask_llm_timed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEMPERATURE   = 0.7
MAX_TOKENS    = 512
PAUSE_SECONDS = 2        # pause between strategy calls (free-tier rate limits)
REPLY_WIDTH   = 90       # characters for wrapped reply text


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 100) -> str:
    return char * width


def _wrap(text: str, indent: str = "  ") -> str:
    """Wrap long reply text to REPLY_WIDTH, with a consistent left indent."""
    return textwrap.fill(
        text,
        width=REPLY_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _print_result(index: int, strategy: str, result: dict) -> None:
    """Pretty-print one strategy's result block."""
    total = len(STRATEGIES)
    label = strategy.replace("_", " ").title()

    print()
    print(_divider("═"))
    print(f"  [{index}/{total}]  Strategy: {label!r}")
    print(_divider("─"))

    if "error" in result:
        print(f"  ⚠  ERROR: {result['error']}")
    else:
        print(f"  ⏱  Latency : {result['latency_seconds']:.3f} s")
        print(
            f"  🔢  Tokens  : "
            f"prompt={result['prompt_tokens']}  "
            f"output={result['output_tokens']}  "
            f"total={result['total_tokens']}"
        )
        print(f"  💬  Reply:")
        print(_wrap(result["reply"].strip()))

    print(_divider("─"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(_divider("═"))
    print("  LLM Strategy Comparator")
    print(_divider("═"))
    print(f"  Strategies to run : {', '.join(STRATEGIES)}")
    print(f"  Temperature       : {TEMPERATURE}")
    print(f"  Max output tokens : {MAX_TOKENS}")
    print(f"  Pause between calls: {PAUSE_SECONDS} s")
    print(_divider("─"))

    # --- gather task ---
    print()
    print("Enter your task / question (press Enter twice or Ctrl-Z/Ctrl-D to finish):")
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass  # Ctrl-Z / Ctrl-D on Windows/Unix

    task = "\n".join(lines).strip()
    if not task:
        print("\n⚠  No task entered. Exiting.")
        sys.exit(1)

    print()
    print(f"  Task: {task!r}")
    print()

    # --- run strategies ---
    results: list[tuple[str, dict]] = []

    for idx, strategy in enumerate(STRATEGIES, start=1):
        label = strategy.replace("_", " ").title()
        print(f"  ▶  Running '{label}' ({idx}/{len(STRATEGIES)}) …", end="", flush=True)

        system_prompt, user_message = build_prompt(strategy, task)
        result = ask_llm_timed(
            system_prompt=system_prompt or "",
            user_message=user_message,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        results.append((strategy, result))
        print(" done." if "error" not in result else " ⚠ error.")

        # Pause between calls — skip after the last one.
        if idx < len(STRATEGIES):
            print(f"  ⏸  Waiting {PAUSE_SECONDS}s before next call …")
            time.sleep(PAUSE_SECONDS)

    # --- summary ---
    print()
    print(_divider("═"))
    print("  RESULTS SUMMARY")
    print(_divider("═"))

    for idx, (strategy, result) in enumerate(results, start=1):
        _print_result(idx, strategy, result)

    print()
    print("  All strategies completed.")
    print(_divider("═"))


if __name__ == "__main__":
    main()

