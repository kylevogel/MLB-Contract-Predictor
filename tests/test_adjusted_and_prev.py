import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pandas as pd
from adjusted_aav import attach_adjusted_aav, deferral_pv
from prev_contract import add_previous_contract


def _contracts(**kw):
    base = dict(spotrac_id=24661, years_new=10, total_value_m=700.0, aav_nominal_m=70.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_ohtani_adjusted_aav_near_mlb_official_figure():
    d = pd.DataFrame([dict(spotrac_id=24661, years=10, total=700e6, orig_years="24-33", deferred=680e6, def_years="34-43")])
    out = attach_adjusted_aav(_contracts(), d, pd.Series([pd.Timestamp("2023-12-09")]))
    assert abs(out.aav_adjusted.iloc[0] - 46.08) / 46.08 < 0.05   # MLB's luxury-tax AAV for Ohtani
    assert bool(out.has_deferral.iloc[0])


def test_no_deferral_means_adjusted_equals_nominal():
    d = pd.DataFrame(columns=["spotrac_id", "years", "total", "orig_years", "deferred", "def_years"])
    out = attach_adjusted_aav(_contracts(spotrac_id=1, total_value_m=100.0, years_new=5, aav_nominal_m=20.0), d,
                              pd.Series([pd.Timestamp("2020-01-01")]))
    assert out.aav_adjusted.iloc[0] == 20.0 and not bool(out.has_deferral.iloc[0])


def test_impossible_deferral_is_not_used():
    d = pd.DataFrame([dict(spotrac_id=7335, years=4, total=62e6, orig_years="17-20", deferred=80e6, def_years="21-28")])
    out = attach_adjusted_aav(_contracts(spotrac_id=7335, years_new=4, total_value_m=62.0, aav_nominal_m=15.5), d,
                              pd.Series([pd.Timestamp("2016-12-01")]))
    assert out.aav_adjusted.iloc[0] == 15.5 and out.aav_adjusted_method.iloc[0].endswith("invalid")


def test_pv_is_less_than_face_value():
    assert deferral_pv(100e6, "30-39", pd.Timestamp("2020-01-01")) < 100e6


def test_previous_contract_uses_only_earlier_classes():
    rows = []
    for mid, off, typ, tot, yrs in [(1, 2015, "free_agent", 50.0, 5), (1, 2020, "extension_veteran", 100.0, 4),
                                   (1, 2020, "free_agent", 20.0, 1), (2, 2018, "free_agent", 10.0, 2)]:
        rows.append(dict(mlb_id=mid, offseason=off, contract_type=typ, total_value_nominal=tot, years_new=yrs,
                         aav_nominal=tot / yrs, aav_adjusted=tot / yrs))
    out = add_previous_contract(pd.DataFrame(rows))
    first, second, third, other = (out.iloc[i] for i in range(4))
    assert not first.has_prev_contract and not other.has_prev_contract
    assert second.has_prev_contract and second.prev_contract_total_m == 50.0 and second.years_since_prev_contract == 5
    assert third.prev_contract_total_m == 50.0          # the same-class $100M deal must not leak into it
