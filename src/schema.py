"""Dataset schema and validation for the MLB contract dataset.

One row = one guaranteed major-league contract signed after a player's rookie
deal (free agency or extension). See DATA_SPEC.md for definitions.
"""
import pandas as pd

CONTRACT_TYPES = [
    "free_agent",            # signed after reaching free agency
    "extension_pre_arb",     # extension before arbitration eligibility
    "extension_arb",         # extension signed during arbitration years
    "extension_veteran",     # Spotrac "Veteran Extension" (past arbitration, or buying out FA years)
]
POS_GROUPS = ["hitter", "pitcher"]  # two-way players are excluded (D-08)
FIRST_OFFSEASON, LAST_OFFSEASON = 2014, 2025  # offseason 2025 = Nov 1 2025 .. Oct 31 2026


def offseason_of(date) -> int:
    """Class rule: signing window Nov 1 (Y) .. Oct 31 (Y+1) belongs to offseason Y."""
    return date.year if date.month >= 11 else date.year - 1

COLUMNS = {
    "player_name": "str",
    "mlb_id": "Int64",            # Chadwick key_mlbam
    "fg_id": "Int64",             # FanGraphs id (key_fangraphs)
    "bbref_id": "str",
    "offseason": "Int64",         # year the offseason starts (2025 = 2025-26)
    "signing_date": "datetime",
    "contract_type": "str",
    "team": "str",
    "prev_team": "str",
    "is_resign": "bool",
    "position": "str",
    "pos_group": "str",
    "age_at_signing": "float",
    "service_years_at_signing": "float",
    "years_new": "Int64",         # total new years counted from signing date
    "total_value_nominal": "float",   # $M
    "aav_nominal": "float",           # $M, total / years_new (TARGET)
    "aav_adjusted": "float",          # $M, deferrals discounted / CBT-style
    "deferred_amount_m": "float",   # $M of the contract paid later (Spotrac deferral list)
    "aav_adjusted_method": "str",
    "at_league_minimum": "bool",    # AAV within 1% of that season's MLB minimum salary (D-07)
    "has_opt_out": "bool",
    "has_deferral": "bool",
    "has_incentives": "bool",
    "source_url": "str",
    "notes": "str",
    # previous-contract features (src/prev_contract.py)
    "has_prev_contract": "bool",
    "prev_contract_type": "str",
    "prev_contract_years": "float",
    "prev_contract_total_m": "float",
    "prev_contract_aav": "float",
    "prev_contract_aav_adjusted": "float",
    "years_since_prev_contract": "float",
}


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = clean)."""
    problems = []
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    bad = df[~df.contract_type.isin(CONTRACT_TYPES)]
    if len(bad):
        problems.append(f"{len(bad)} rows with unknown contract_type: {bad.contract_type.unique()[:5]}")
    bad = df[~df.pos_group.isin(POS_GROUPS)]
    if len(bad):
        problems.append(f"{len(bad)} rows with unknown pos_group")
    bad = df[~df.offseason.between(FIRST_OFFSEASON, LAST_OFFSEASON)]
    if len(bad):
        problems.append(f"{len(bad)} rows outside offseasons {FIRST_OFFSEASON}-{LAST_OFFSEASON}")
    bad = df[(df.years_new < 1) | (df.total_value_nominal <= 0)]
    if len(bad):
        problems.append(f"{len(bad)} rows with non-positive years or value")
    calc = df.total_value_nominal / df.years_new
    bad = df[(calc - df.aav_nominal).abs() > 0.05]
    if len(bad):
        problems.append(f"{len(bad)} rows where aav_nominal != total/years (>$0.05M)")
    bad = df[df.aav_adjusted > df.aav_nominal + 0.05]
    if len(bad):
        problems.append(f"{len(bad)} rows where aav_adjusted > aav_nominal (check deferral math)")
    dated = df[df.signing_date.notna()]
    bad = dated[dated.signing_date.map(offseason_of) != dated.offseason]
    if len(bad):
        problems.append(f"{len(bad)} rows where offseason != class implied by signing_date")
    dup = df.duplicated(["mlb_id", "offseason", "contract_type", "signing_date", "total_value_nominal", "years_new"], keep=False)
    if dup.any():
        problems.append(f"{dup.sum()} duplicate contract rows")
    if df.mlb_id.isna().any():
        problems.append(f"{df.mlb_id.isna().sum()} rows without an mlb_id (ID join failed)")
    return problems
