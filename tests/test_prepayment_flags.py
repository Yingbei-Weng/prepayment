import pandas as pd

from prepayment.features import PrepayDefinition, add_prepayment_flags


def test_add_prepayment_flags_early_payoff_and_exclusions() -> None:
    df = pd.DataFrame(
        [
            {
                "lnno": 1,
                "sort_key": 20201,
                "mrtg_status": 100,
                "amt_upb_endg": 100.0,
                "begin_upb": 100.0,
                "liq_upb_amt": None,
                "liq_dte": None,
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 1,
                "sort_key": 20202,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 100.0,
                "liq_upb_amt": 100.0,
                "liq_dte": "2020-05-15",
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 2,
                "sort_key": 20221,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 200.0,
                "liq_upb_amt": 200.0,
                "liq_dte": "2022-01-01",
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 3,
                "sort_key": 20211,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 300.0,
                "liq_upb_amt": None,
                "liq_dte": None,
                "dt_mty": "2022-10-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 4,
                "sort_key": 20201,
                "mrtg_status": 300,
                "amt_upb_endg": 400.0,
                "begin_upb": 400.0,
                "liq_upb_amt": None,
                "liq_dte": None,
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 4,
                "sort_key": 20202,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 400.0,
                "liq_upb_amt": 400.0,
                "liq_dte": "2020-05-15",
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 5,
                "sort_key": 20202,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 500.0,
                "liq_upb_amt": 500.0,
                "liq_dte": "2020-05-15",
                "dt_mty": "2022-01-01",
                "credit_loss": 1.0,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 6,
                "sort_key": 20202,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 600.0,
                "liq_upb_amt": 600.0,
                "liq_dte": "2020-05-15",
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": "2020-06-01",
                "Sales_Price": None,
                "flag_defeased": None,
            },
            {
                "lnno": 7,
                "sort_key": 20202,
                "mrtg_status": 500,
                "amt_upb_endg": 0.0,
                "begin_upb": 700.0,
                "liq_upb_amt": 700.0,
                "liq_dte": "2020-05-15",
                "dt_mty": "2022-01-01",
                "credit_loss": None,
                "dt_sold": None,
                "Sales_Price": None,
                "flag_defeased": 1.0,
            },
        ]
    )

    for col in ["liq_dte", "dt_mty", "dt_sold"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    out = add_prepayment_flags(df, PrepayDefinition())

    prepay = out[out["is_prepay"]].copy()
    assert set(prepay["lnno"].tolist()) == {1, 3}

    row_1 = out[(out["lnno"] == 1) & (out["sort_key"] == 20202)].iloc[0]
    assert bool(row_1["is_terminated"]) is True
    assert bool(row_1["is_early_payoff"]) is True
    assert row_1["prepay_upb"] == 100.0

    row_2 = out[(out["lnno"] == 2) & (out["sort_key"] == 20221)].iloc[0]
    assert bool(row_2["is_terminated"]) is True
    assert bool(row_2["is_prepay"]) is False

    row_3 = out[(out["lnno"] == 3) & (out["sort_key"] == 20211)].iloc[0]
    assert bool(row_3["is_prepay"]) is True
    assert row_3["prepay_upb"] == 300.0

    row_4_term = out[(out["lnno"] == 4) & (out["sort_key"] == 20202)].iloc[0]
    assert bool(row_4_term["is_early_payoff"]) is True
    assert bool(row_4_term["is_prepay"]) is False
