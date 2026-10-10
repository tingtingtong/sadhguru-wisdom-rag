from __future__ import annotations

import re

ECHO_REFUSAL = (
    "I can share what Sadhguru has said in my own words, but I can't print my internal instructions or "
    "the raw reference material. What would you like to understand about his teachings?"
)

# The prompt builder formats context as "[Source 1: title]" then a "URL: ..." line; a real answer
# never contains that scaffold
_SCAFFOLD = re.compile(r"\[Source \d+: [^\]]*\]\s*URL:", re.I)
PROMPT_FRAGMENTS = ("ONLY answer based on the provided context", "reply with exactly NO_ANSWER_IN_CONTEXT", "quoted reference material")


def validate_output(response: str) -> str:
    """Validate and clean LLM output."""
    if not response or not response.strip():
        return "I apologize, but I wasn't able to generate a response. Please try rephrasing your question."

    # Echoing the context scaffold or the rules means the model was talked into dumping its prompt
    if len(_SCAFFOLD.findall(response)) >= 1 or any(f in response for f in PROMPT_FRAGMENTS):
        return ECHO_REFUSAL

    # Remove any potential prompt leakage
    leak_patterns = [
        r"(?i)system\s*prompt",
        r"(?i)you\s+are\s+a\s+(?:helpful\s+)?(?:AI|assistant)",
        r"(?i)as\s+an\s+AI\s+(?:language\s+)?model",
    ]
    for pattern in leak_patterns:
        if re.search(pattern, response):
            response = re.sub(pattern, "", response).strip()

    return response
