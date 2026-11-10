# Data Specification: MLB Contract Dataset

Decisions agreed by the team on 2026-11-10.

## What counts as a row
One row per **guaranteed major-league contract signed after a player's rookie deal**, from the
**2014-15 through 2025-26 offseasons** (12 classes).

**Class rule (decided 2026-11-10):** the class is set by signing date. Nov 1 (Y) to Oct 31 (Y+1)
is offseason Y, so offseason 2025 runs Nov 1 2025 to Oct 31 2026 and is the final test class.
It includes in-season 2026 extensions. Anything signed Nov 1 2026 or later is out
(three extensions already: Young, Webb, Waguespack). Soto's December 2024 deal is offseason 2024.
Free agents have no signing date on Spotrac, so their class is the contract start year minus 1.
Extensions use the transaction date when Spotrac has one (289 of 340 in scope), otherwise
the signed year minus 1 (51 rows; `date_source` records which).

## Included
- Free-agent signings (new team or re-signing; `is_resign` flags re-signings)
- All extensions, tagged by `contract_type`:
  `free_agent`, `extension_pre_arb`, `extension_arb`, `extension_veteran` (Spotrac's three extension types)

## Excluded (kept in `contracts_dropped.csv` with a reason, so the filtering is auditable)
- Arbitration settlements and awards
- Minor-league deals and non-roster invitations
- International signings with no MLB history (NPB/KBO/CPBL etc.). Spotrac's free-agent `From` column shows JPN, KOR, MEX, DOM, OS for these.
- Extensions of players with no MLB history (`no_mlb_history`, mostly 19-21 year-old prospects). Kept in `contracts_dropped.csv` for case studies. Needs the Chadwick `mlb_played_first` check.
- Free-agent rows with a $0 value (164 found, treated as minor-league or unknown terms)
- Rookie / pre-arbitration standard contracts

## Targets
| Column | Definition |
| --- | --- |
| `years_new` | Total new years counted from the signing date. For an extension, this is the whole new commitment from that date. |
| `aav_nominal` | `total_value_nominal / years_new` in $M. **The model predicts this.** |
| `aav_adjusted` | AAV with deferrals discounted (luxury-tax style). Stored for analysis and error studies, not predicted. |

Flags (`has_opt_out`, `has_deferral`, `has_incentives`) are stored but not used as inputs until we decide how.

## Features (built in `fetch_stats.py`)
- Last **3 completed seasons** before the signing date (no future information).
  Hitters: WAR, PA, wRC+. Pitchers: WAR, IP, FIP.
- Plus: age, position group, service time, contract_type, is_resign.
- Hitters and pitchers are modelled separately. Two-way players are treated as hitters for now
  (open question).

## Split (by offseason)
Train 2014-2021, validate 2022-2023, test 2024-2025.

## First cleaning run (2026-11-10): 2,512 scraped rows -> 1,909 kept
Every dropped row is in `data/processed/prepare_dropped.csv` with a reason.

| Reason | Rows |
| --- | --- |
| Undated pre-2013 extension (outside window) | 206 |
| Minor-league deal ($0 value) | 161 |
| Outside the class window (extensions before Nov 2014 or after Oct 2026) | 147 |
| No completed MLB season before signing (international imports, pre-debut prospects) | 57 |
| Follow-up deal in the same class (< 50% of the player's biggest deal) | 15 |
| Exact duplicate | 10 |
| Not in the MLB register (no MLB history) | 3 |
| Zero contract years | 2 |
| Below that season's MLB minimum salary | 2 |

Kept: 1,588 free agent, 147 arbitration extension, 113 veteran extension, 61 pre-arbitration extension.
Hitters 874, pitchers 1,034, two-way 1 (Ohtani).

### Verification against independent sources
24 headline contracts from MLB.com, CBS Sports, MLB Trade Rumors and Boardroom were compared with the dataset:
21 match exactly on years and total value. The 3 differences are explained:
- Mike Trout (extension): Spotrac counts 12 yrs / $426.5M (includes the 2 years already owed); MLB.com and MLBTR count new money only (10 yrs / $360M).
- Julio Rodriguez (extension signed in his debut season): dropped by the no-completed-season rule.
- Yoshinobu Yamamoto: dropped as an international signing (intended).
Name-to-ID matching: 9 players needed manual overrides (`data/raw/manual_id_overrides.csv`), 3 not in the MLB register were dropped.

## Open items
0. See the questions list in the team chat for extension-overlap years, in-season debut extensions, and Spotrac redistribution.
1. The 2020 shortened season: add a flag or scale counting stats to 162 games.
2. Two-way players (Ohtani): hitter model, pitcher model, or drop.
3. Pre-arb extensions are a small group. Check counts before deciding whether to model contract_type as a feature or as separate models.
4. Spotrac data licensing: confirm what we can republish in the public dataset link the course requires.

## Collection (Spotrac, scraped Nov 10 2026)
`src/scrape_spotrac_console.js` runs in the browser console on a Spotrac MLB page (about 60 requests,
0.5 s apart) and downloads `spotrac_raw_contracts.csv`.
- Free-agent rows: 1,820 across classes 2014-2025 (Soto $765M/15, Ohtani $700M/10, Judge $360M/9 spot-checked).
  Includes 164 $0 rows and 206 under $1M rows to review, 8 apparent duplicates, 304 re-signings (From == To).
- Extension rows: 693 all-time; 340 in scope (289 dated, 51 year-only); 206 are pre-2013 and out of scope.
- Spotrac ignores `?query=` filters. It needs path URLs (`/_/year/2024`, `/_/type/...`, `/_/start/.../end/.../page/N`).
- `aav_adjusted` (luxury-tax AAV) and `service_years_at_signing` appear on each player's own page
  ("Annual Luxury Tax", "YOS @ Signing") and are NOT scraped yet. That is about 2,000 page loads.
