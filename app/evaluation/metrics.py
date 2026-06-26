from typing import Optional


def reciprocal_rank(matched_rank: Optional[int]) -> float:
    if not matched_rank or matched_rank <= 0:
        return 0.0
    return 1.0 / matched_rank


def safe_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
