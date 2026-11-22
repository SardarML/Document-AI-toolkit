"""Small evaluation helpers that do not require external models."""


def exact_match(answer: str, reference: str) -> bool:
    """Compare exact normalized strings, not semantic correctness."""
    return answer.strip().casefold() == reference.strip().casefold()


def success_rate(outcomes: list[bool]) -> float:
    """Compute the fraction of successful tasks or tool calls."""
    return sum(outcomes) / len(outcomes) if outcomes else 0.0
