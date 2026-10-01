"""Indian rupee formatting helpers (lakh / crore grouping)."""


def format_inr(amount: float, decimals: int = 0, symbol: bool = True) -> str:
    """
    Format a number with Indian digit grouping.
    format_inr(1234567)   -> '₹12,34,567'
    format_inr(-12123.1)  -> '-₹12,123'
    """
    rounded = round(float(amount), decimals)
    negative = rounded < 0
    text = f"{abs(rounded):.{decimals}f}"
    integer_part, _, fraction = text.partition(".")

    if len(integer_part) > 3:
        head, tail = integer_part[:-3], integer_part[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        integer_part = ",".join(groups) + "," + tail

    body = integer_part + (f".{fraction}" if fraction else "")
    return ("-" if negative else "") + ("₹" if symbol else "") + body


def format_lakh_crore(amount: float) -> str:
    """Short form: 12500000 -> '₹1.25 Cr', 450000 -> '₹4.50 L', 9500 -> '₹9,500'."""
    value = float(amount)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_00_00_000:
        return f"{sign}₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"{sign}₹{value / 1_00_000:.2f} L"
    return sign + format_inr(value)
