from pathlib import Path

from prepayment.datasets import load_loan_panel


def test_load_loan_panel_derives_year_quarter_from_quarter_date(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        "\n".join(
            [
                "lnno,quarter,mrtg_status,amt_upb_endg,liq_upb_amt,dt_fund,amt_upb_pch,rate_int",
                "1,12/31/11,100,100000,,11/9/11,100000,0.05",
                "1,3/31/12,100,99000,,11/9/11,100000,0.05",
            ]
        )
        + "\n"
    )

    df = load_loan_panel(csv_path)
    assert df.loc[0, "year"] == 2011
    assert df.loc[0, "q"] == 4
    assert df.loc[0, "sort_key"] == 20114
    assert df.loc[0, "period"] == "2011Q4"

