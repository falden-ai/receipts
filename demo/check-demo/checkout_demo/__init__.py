"""A public demo package. All amounts are integer cents."""


def total_cents(quantity: int, unit_price_cents: int) -> int:
    """Apply a 10 percent discount at five items or more, rounding the discount down."""
    if quantity < 0 or unit_price_cents < 0:
        raise ValueError("quantity and unit price must be nonnegative")
    subtotal = quantity * unit_price_cents
    if quantity >= 5:
        return subtotal - subtotal // 10
    return subtotal
