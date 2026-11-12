"""Prompt strategies for comparing different LLM prompting techniques."""

from typing import Optional, Tuple

# Supported prompting strategies. Keeping this list makes it easy to iterate
# over strategies, validate inputs, and add new ones in future stages.
STRATEGIES = [
    "zero_shot",
    "few_shot",
    "role_based",
    "step_by_step",
]


def build_prompt(strategy: str, task: str) -> Tuple[Optional[str], str]:
    """Build a (system_prompt, user_message) tuple for a given strategy and task.

    Args:
        strategy: One of the strategies in STRATEGIES.
        task: The core user task or question to be solved.

    Returns:
        A tuple of (system_prompt, user_message).
        system_prompt is None if the strategy does not use system instructions.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in STRATEGIES:
        valid = ", ".join(f"'{s}'" for s in STRATEGIES)
        raise ValueError(f"Unknown strategy '{strategy}'. Supported strategies: {valid}")

    clean_task = task.strip()

    # =========================================================================
    # 1. ZERO-SHOT
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT CAN IMPROVE / WHEN TO USE:
    # Zero-shot is the raw baseline. It provides no extra context, formatting
    # demonstrations, or persona guidance. It tests the model's innate pre-trained
    # knowledge and basic instruction-following ability while minimizing prompt
    # token usage and latency.
    # =========================================================================
    if strategy == "zero_shot":
        system_prompt = None
        user_message = clean_task
        return system_prompt, user_message

    # =========================================================================
    # 2. FEW-SHOT
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT CAN IMPROVE / WHEN TO USE:
    # Few-shot prompting uses "in-context learning". By showing the model concrete
    # input-output demonstrations before the actual query, we establish a clear
    # pattern for tone, brevity, and structure. This significantly reduces
    # hallucination, eliminates ambiguous formatting, and aligns the response style.
    # =========================================================================
    if strategy == "few_shot":
        system_prompt = None
        user_message = (
            "Here are a few examples of how to approach and answer tasks clearly:\n\n"
            "Example 1:\n"
            "Task: Explain what DNS does in simple terms.\n"
            "Output: DNS acts as the internet's phonebook, translating human-readable "
            "domain names (like example.com) into machine-readable IP addresses.\n\n"
            "Example 2:\n"
            "Task: Convert 25 Celsius to Fahrenheit.\n"
            "Output: Use the formula (C * 9/5) + 32. (25 * 9/5) + 32 = 45 + 32 = 77°F.\n\n"
            "Example 3:\n"
            "Task: What is the main benefit of writing automated tests?\n"
            "Output: Automated tests catch bugs early in development, giving developers "
            "confidence to refactor code without causing regressions.\n\n"
            f"Now, complete the following task:\n{clean_task}"
        )
        return system_prompt, user_message

    # =========================================================================
    # 3. ROLE-BASED
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT CAN IMPROVE / WHEN TO USE:
    # Role prompting primes the model's attention toward domain-specific knowledge,
    # terminology, and analytical standards. Asking the model to assume an expert
    # persona encourages comprehensive, authoritative, and professionally structured
    # answers rather than casual or surface-level summaries.
    # =========================================================================
    if strategy == "role_based":
        system_prompt = (
            "You are a leading domain expert and senior technical advisor. "
            "Provide rigorous, accurate, and insightful answers with clear structure, "
            "practical explanations, and professional attention to detail."
        )
        user_message = clean_task
        return system_prompt, user_message

    # =========================================================================
    # 4. STEP-BY-STEP (Chain-of-Thought)
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT CAN IMPROVE / WHEN TO USE:
    # Chain-of-Thought (CoT) prompting encourages the model to generate intermediate
    # reasoning tokens before committing to an answer. Because LLMs predict the next
    # token conditioned on prior tokens, working through intermediate steps prevents
    # arithmetic and logical errors, making the final answer much more reliable.
    # =========================================================================
    if strategy == "step_by_step":
        system_prompt = None
        user_message = (
            f"{clean_task}\n\n"
            "Please think through this problem step by step. Show your reasoning and "
            "intermediate steps clearly, and then provide your final conclusion on a "
            "separate line starting with 'Answer:'."
        )
        return system_prompt, user_message

    # Fallback (should not be reached due to validation check above)
    return None, clean_task

