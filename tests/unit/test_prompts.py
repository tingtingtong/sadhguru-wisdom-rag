from generation.prompts import build_system_prompt, build_no_context_response


def test_build_system_prompt_with_docs():
    docs = [
        {"title": "Karma", "url": "https://example.com/karma", "text": "Karma is action."},
        {"title": "Yoga", "url": "https://example.com/yoga", "text": "Yoga is union."},
    ]
    prompt = build_system_prompt(docs)
    assert "Karma is action" in prompt
    assert "Yoga is union" in prompt
    assert "[Source 1: Karma]" in prompt
    assert "[Source 2: Yoga]" in prompt


def test_build_system_prompt_empty():
    prompt = build_system_prompt([])
    assert "No relevant context" in prompt


def test_no_context_response():
    response = build_no_context_response()
    assert "isha.sadhguru.org" in response
