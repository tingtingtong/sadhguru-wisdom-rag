from guardrails.input_validator import validate_input


def test_validate_empty():
    is_valid, error = validate_input("")
    assert not is_valid
    assert error is not None


def test_validate_too_long():
    is_valid, error = validate_input("a" * 2001)
    assert not is_valid


def test_validate_too_short():
    is_valid, error = validate_input("ab")
    assert not is_valid


def test_validate_ok():
    is_valid, error = validate_input("What is karma?")
    assert is_valid
    assert error is None
