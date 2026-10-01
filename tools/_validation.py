"""
Small input checks shared by all calculation tools.

Every tool validates its own inputs so that bad values fail loudly with a
clear message instead of silently producing a wrong number.
"""

import math
from numbers import Real

import config


def ensure_number(value, name: str) -> float:
    """Return value as float. Reject non numbers, booleans, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def ensure_non_negative(value, name: str) -> float:
    """Return value as float. Reject negative values."""
    number = ensure_number(value, name)
    if number < 0:
        raise ValueError(f"{name} cannot be negative (got {number}).")
    return number


def ensure_return_pct(value, name: str = "Annual return %") -> float:
    """Annual return in percent, e.g. 12 means 12%."""
    number = ensure_number(value, name)
    if not config.MIN_RETURN_PCT <= number <= config.MAX_RETURN_PCT:
        raise ValueError(
            f"{name} must be between {config.MIN_RETURN_PCT} and "
            f"{config.MAX_RETURN_PCT} (got {number})."
        )
    return number


def ensure_inflation_pct(value, name: str = "Annual inflation %") -> float:
    """Annual inflation in percent, e.g. 6 means 6%."""
    number = ensure_number(value, name)
    if not config.MIN_INFLATION_PCT <= number <= config.MAX_INFLATION_PCT:
        raise ValueError(
            f"{name} must be between {config.MIN_INFLATION_PCT} and "
            f"{config.MAX_INFLATION_PCT} (got {number})."
        )
    return number


def ensure_months(value, name: str = "Months") -> int:
    """Whole number of months, 0 or more. 12.0 is accepted, 12.5 is not."""
    number = ensure_non_negative(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number (got {number}).")
    return int(number)


def ensure_horizon_years(value, name: str = "Goal horizon (years)") -> int:
    """Whole number of years between 1 and MAX_HORIZON_YEARS."""
    number = ensure_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number of years (got {number}).")
    years = int(number)
    if not 1 <= years <= config.MAX_HORIZON_YEARS:
        raise ValueError(
            f"{name} must be between 1 and {config.MAX_HORIZON_YEARS} (got {years})."
        )
    return years
