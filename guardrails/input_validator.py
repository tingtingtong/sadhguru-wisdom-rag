from __future__ import annotations


def validate_input(query: str) -> tuple[bool, str | None]:
    """Validate user input. Returns (is_valid, error_message_or_none)."""
    if not query or not query.strip():
        return False, "Please provide a question."

    if len(query) > 2000:
        return False, "Question is too long. Please keep it under 2000 characters."

    if len(query.strip()) < 3:
        return False, "Question is too short. Please provide more detail."

    return True, None
