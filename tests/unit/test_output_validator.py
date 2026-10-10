from guardrails.output_validator import validate_output


def test_validate_empty():
    result = validate_output("")
    assert "apologize" in result.lower() or "unable" in result.lower()


def test_validate_normal():
    result = validate_output("Karma is the memory of life.")
    assert result == "Karma is the memory of life."


def test_validate_strips_prompt_leak():
    result = validate_output("As an AI language model, karma means action.")
    assert "AI language model" not in result
