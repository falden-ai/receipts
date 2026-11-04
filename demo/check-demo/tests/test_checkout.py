"""The correct base behavior before the deliberately planted demo regression."""

import pytest
from checkout_demo import total_cents


@pytest.mark.parametrize(("quantity", "expected"), [(0, 0), (4, 400), (5, 450), (6, 540)])
def test_discount_boundary(quantity: int, expected: int) -> None:
    assert total_cents(quantity, 100) == expected


def test_discount_rounds_down_to_whole_cents() -> None:
    assert total_cents(5, 101) == 455


@pytest.mark.parametrize(("quantity", "unit_price_cents"), [(-1, 100), (1, -100)])
def test_negative_input_is_refused(quantity: int, unit_price_cents: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        total_cents(quantity, unit_price_cents)
