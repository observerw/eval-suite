def instruction_messages(
    instruction: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": instruction}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages
