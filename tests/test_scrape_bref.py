import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pytest
from scrape_bref_stats import parse_season_table

# Hand-made sample in Baseball-Reference's layout (made-up players and numbers, no real page content).
ROW = ('<tr{cls}><th data-stat="ranker">{rk}</th>'
       '<td data-stat="name_display"{pid}>{name}</td><td data-stat="age">30</td>'
       '<td data-stat="team_name_abbr">{tm}</td><td data-stat="comp_name_abbr">AL</td>'
       '<td data-stat="b_war">{war}</td><td data-stat="b_games">100</td><td data-stat="b_pa">{pa}</td>'
       '<td data-stat="b_r">50</td><td data-stat="b_batting_avg">.270</td>'
       '<td data-stat="b_onbase_plus_slugging_plus">{ops}</td><td data-stat="pos">*9</td>'
       '<td data-stat="awards">{aw}</td></tr>')


def row(rk="1", pid="playa01", name="Player A", tm="NYY", war="2.0", pa="500", ops="110", aw="", cls=""):
    pid = f' data-append-csv="{pid}"' if pid else ""
    cls = f' class="{cls}"' if cls else ""
    return ROW.format(rk=rk, pid=pid, name=name, tm=tm, war=war, pa=pa, ops=ops, aw=aw, cls=cls)


def page(*rows, commented=False):
    table = f'<table id="players_standard_batting"><thead></thead><tbody>{"".join(rows)}</tbody></table>'
    return f"<html><body>{f'<!--{table}-->' if commented else table}</body></html>"


SAMPLE = page(
    row(name="Player A*", aw="AS,MVP-1,SS"),
    row(rk="2", pid="tradeb01", name="Traded B", tm="2TM", war="3.0", pa="600"),
    row(rk="2", pid="tradeb01", name="Traded B", tm="BOS", war="1.0", pa="250", cls="partial_table"),
    row(rk="2", pid="tradeb01", name="Traded B", tm="SEA", war="2.0", pa="350", cls="partial_table"),
    row(rk="3", pid="pitchc01", name="Pitcher C", pa="0", ops=""),
    row(rk="", pid="", name="MLB Average", war="", cls="norank"),
)


def test_one_row_per_player_using_combined_traded_row():
    df = parse_season_table(SAMPLE, "batting", 2023)
    assert list(df.bbref_id) == ["playa01", "tradeb01", "pitchc01"]
    traded = df.set_index("bbref_id").loc["tradeb01"]
    assert traded.team == "2TM" and traded.war == 3.0 and traded.pa == 600


def test_values_types_and_name_markers():
    df = parse_season_table(SAMPLE, "batting", 2023).set_index("bbref_id")
    a = df.loc["playa01"]
    assert a["name"] == "Player A" and a.season == 2023
    assert a.ba == pytest.approx(0.270) and a.ops_plus == 110 and a.awards == "AS,MVP-1,SS"
    assert df.loc["pitchc01"].ops_plus != df.loc["pitchc01"].ops_plus  # blank OPS+ -> NaN


def test_table_inside_html_comment():
    df = parse_season_table(page(row(), commented=True), "batting", 2023)
    assert list(df.bbref_id) == ["playa01"]


def test_missing_table_raises():
    with pytest.raises(ValueError):
        parse_season_table("<html></html>", "batting", 2023)


def test_duplicate_player_raises():
    with pytest.raises(ValueError):
        parse_season_table(page(row(), row()), "batting", 2023)
