"""A share-of-total column must not be presented as a change between two periods."""
from src.agent.markdown_utils import format_markdown_table

ROWS = [
    {"restaurant_name": "KFC", "revenue": 42466314.0, "pct_of_total": 10.42},
    {"restaurant_name": "Domino's Pizza", "revenue": 35671213.0, "pct_of_total": 8.75},
    {"restaurant_name": "Pizza Hut", "revenue": 32128653.0, "pct_of_total": 7.88},
]


def test_share_column_is_labelled_and_formatted_as_a_share():
    md = format_markdown_table(ROWS)
    assert "% of Total" in md
    assert "% Change" not in md
    assert "10.42%" in md and "+10.42%" not in md   # no sign: a share is not a change


def test_period_change_column_keeps_its_label_and_sign():
    rows = [{"restaurant_name": n, "sales_cur": 90.0, "sales_prev": 100.0, "pct_change": -10.0} for n in "ABC"]
    md = format_markdown_table(rows)
    assert "% Change" in md and "-10.00%" in md
