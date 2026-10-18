import pytest

from evaluations.runner import estimate_cost, validate_cases


def test_validate_cases_requires_twenty_per_task():
    cases = [{"task": task} for task in ("knowledge_brain", "inbox_agent", "meeting_to_sow")]
    with pytest.raises(ValueError, match="At least 20"):
        validate_cases(cases)


def test_validate_cases_accepts_complete_distribution():
    cases = [{"task": task} for task in ("knowledge_brain", "inbox_agent", "meeting_to_sow") for _ in range(20)]
    validate_cases(cases)


def test_estimate_cost_requires_token_usage():
    assert estimate_cost(None, 20, 3.0, 15.0) is None
    assert estimate_cost(1_000_000, 1_000_000, 3.0, 15.0) == 18.0
