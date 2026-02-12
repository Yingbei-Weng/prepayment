from prepayment.datasets import load_quarterly_rates


def test_rate_loader_scales_to_decimal() -> None:
    series = load_quarterly_rates("10y_yahoo_quarter_avg.csv")
    med = float(series.df["market_rate"].median())
    assert 0.005 < med < 0.15

