import pandas as pd

from prepayment.datasets import load_loan_panel
from prepayment.features import prepare_loan_quarterly_data


def test_load_loan_panel_derives_year_q_sort_key_from_quarter_date(tmp_path) -> None:
    path = tmp_path / "loans.csv"
    pd.DataFrame(
        [
            {
                "lnno": 1,
                "quarter": "12/31/11",
                "mrtg_status": 100,
                "amt_upb_endg": 100.0,
                "liq_upb_amt": None,
                "dt_fund": "11/9/11",
                "amt_upb_pch": 100.0,
                "rate_int": 0.05,
            },
            {
                "lnno": 1,
                "quarter": "3/31/12",
                "mrtg_status": 100,
                "amt_upb_endg": 100.0,
                "liq_upb_amt": None,
                "dt_fund": "11/9/11",
                "amt_upb_pch": 100.0,
                "rate_int": 0.05,
            },
        ]
    ).to_csv(path, index=False)

    out = load_loan_panel(path)
    assert out.loc[0, "year"] == 2011
    assert out.loc[0, "q"] == 4
    assert out.loc[0, "sort_key"] == 20114
    assert out.loc[0, "period"] == "2011Q4"
    assert out.loc[1, "year"] == 2012
    assert out.loc[1, "q"] == 1
    assert out.loc[1, "sort_key"] == 20121
    assert out.loc[1, "period"] == "2012Q1"


def test_prepare_loan_quarterly_data_uses_is_prepaid_flag() -> None:
    loans = pd.DataFrame(
        [
            {
                "lnno": 1,
                "year": 2020,
                "q": 1,
                "sort_key": 20201,
                "period": "2020Q1",
                "dt_fund": "2019-01-01",
                "loan_age_quarters": 0,
                "amt_upb_pch": 100.0,
                "amt_upb_endg": 100.0,
                "liq_upb_amt": None,
                "mrtg_status": 100,
                "rate_int": 0.05,
                "is_prepaid": 0,
            },
            {
                "lnno": 1,
                "year": 2020,
                "q": 2,
                "sort_key": 20202,
                "period": "2020Q2",
                "dt_fund": "2019-01-01",
                "loan_age_quarters": 1,
                "amt_upb_pch": 100.0,
                "amt_upb_endg": None,
                "liq_upb_amt": None,
                "mrtg_status": 500,
                "rate_int": 0.05,
                "is_prepaid": 1,
            },
        ]
    )
    loans["dt_fund"] = pd.to_datetime(loans["dt_fund"], errors="coerce")

    rates = pd.DataFrame({"period": ["2020Q1", "2020Q2"], "market_rate": [0.04, 0.04]})
    out = prepare_loan_quarterly_data(loans, rates)

    assert out["is_prepay"].tolist() == [False, True]
    assert out["age_quarters"].tolist() == [0, 1]
    assert out.loc[out["sort_key"] == 20202, "prepay_upb"].item() == 100.0
    assert "market_rate_rates" not in out.columns
