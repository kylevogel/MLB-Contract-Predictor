# Data Specification: MLB Contract Dataset

Decisions agreed by the team on 2026-11-10; performance features added 2026-11-17 (D-15 to D-17).

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

## Excluded (kept in `data/processed/prepare_dropped.csv` with a reason, so the filtering is auditable)
- Arbitration settlements and awards
- Minor-league deals and non-roster invitations
- International signings with no MLB history (NPB/KBO/CPBL etc.). Spotrac's free-agent `From` column shows JPN, KOR, MEX, DOM, OS for these.
- Extensions of players with no MLB history (`no_mlb_history`, mostly 19-21 year-old prospects). Kept in `prepare_dropped.csv` for case studies.
- Free-agent rows with a $0 value (164 found, treated as minor-league or unknown terms)
- Rookie / pre-arbitration standard contracts

## Targets
| Column | Definition |
| --- | --- |
| `years_new` | Total new years counted from the signing date. For an extension, this is the whole new commitment from that date. |
| `aav_nominal` | `total_value_nominal / years_new` in $M. **The model predicts this.** |
| `aav_adjusted` | AAV with deferrals discounted (luxury-tax style). Stored for analysis and error studies, not predicted. |

Flags (`has_opt_out`, `has_deferral`, `has_incentives`) are stored but not used as inputs until we decide how.

## Features (built in `src/build_features.py`, D-15 and D-16)
- Source: Baseball-Reference season tables (`src/scrape_bref_stats.py`, seasons 2012-2025).
- Window: the 3 completed seasons before signing (offseason-2 .. offseason), so no future information.
  Each stat is the equal-weight average over the seasons played. Seasons under 50 PA (hitters) or
  10 IP (pitchers) are skipped unless every window season is under that. 2020 WAR and runs are scaled x 162/60.
  Hitters: `war_3yr`, `r_3yr`, `ba_3yr`, `ops_plus_3yr`. Pitchers: `war_3yr`, `era_plus_3yr`, `fip_3yr`, `whip_3yr`.
- Awards (both groups): `won_mvp`, `won_cy_young`, `won_gold_glove`, `won_silver_slugger` = 1 if WON in a window season.
- `seasons_in_window` = number of seasons used. Contracts with none are dropped and logged in
  `data/processed/features_dropped.csv` (11 rows).
- Outputs: `data/processed/contracts_features_hitters.csv` (862), `contracts_features_pitchers.csv` (1,031).
- Plus: age, position group, contract_type, is_resign, previous-contract features. (`service_years_at_signing` is still empty, so it is not a usable feature.)
- Hitters and pitchers are modelled separately. Two-way players: Ohtani excluded (D-08).

## Split (by offseason)
Train 2014-2021, validate 2022-2023, test 2024-2025.

## Cleaning run (2026-11-10): 2,512 scraped rows -> 1,904 kept
Every dropped row is in `data/processed/prepare_dropped.csv` with a reason. 2,512 = 1,904 kept + 608 dropped.

| Reason | Rows |
| --- | --- |
| Undated pre-2013 extension (outside window) | 206 |
| Minor-league deal ($0 value) | 161 |
| Outside the class window (extensions before Nov 2014 or after Oct 2026) | 147 |
| No completed MLB season before signing (international imports, pre-debut prospects, debut-season extensions) | 57 |
| Follow-up league-minimum deal in the same class | 15 |
| Exact duplicate | 10 |
| Not in the MLB register (no MLB history) | 3 |
| Zero contract years | 2 |
| Below that season's MLB minimum salary | 2 |
| Two-way player (Ohtani, free agency) | 1 |
| Ohtani, other contract (2020 arbitration extension) | 1 |
| Double-counted or superseded row, manually verified (D-14: Abreu QO, López renegotiation, Moustakas duplicate) | 3 |

Kept: 1,586 free agent, 146 arbitration extension, 111 veteran extension, 61 pre-arbitration extension.
Hitters 863, pitchers 1,041. Split sizes: train (2014-2021) 1,192, validation (2022-2023) 346, test (2024-2025) 366.

### New columns
`aav_adjusted` and `deferred_amount_m` (Spotrac deferral list, 3% discount, see DECISIONS.md D-12), `at_league_minimum`,
and previous-contract features `has_prev_contract`, `prev_contract_type`, `prev_contract_years`, `prev_contract_total_m`,
`prev_contract_aav`, `prev_contract_aav_adjusted`, `years_since_prev_contract` (D-10). Qualifying-offer flags are not used.
`service_years_at_signing`, `has_opt_out`, `has_incentives` are still empty.

### Verification against independent sources
24 headline contracts from MLB.com, CBS Sports, MLB Trade Rumors and Boardroom were compared with the dataset:
21 matched exactly on years and total value. The 3 differences: Mike Trout (corrected to 10 yrs / $360M new money by majority of sources, D-03),
Julio Rodriguez (dropped, no completed season, D-06) and Yoshinobu Yamamoto (international, dropped).
Minimum-salary floors and the follow-up rule were verified (D-07). Name matching needed 9 manual overrides.

## Open items
0. All decisions and open items are tracked in `DECISIONS.md`.
1. (Resolved) The 2020 shortened season: counting stats scaled to 162 games, D-15.
2. (Resolved) Two-way players: Ohtani excluded, D-08.
3. Pre-arb extensions are a small group. Check counts before deciding whether to model contract_type as a feature or as separate models.
4. (Resolved) Spotrac data licensing (D-13): raw Spotrac files stay out of the repo; the cleaned dataset is published with the professor's approval.
5. `service_years_at_signing`, `has_opt_out`, `has_incentives` are 100% empty: fill, derive or drop.
6. `position` needs cleaning into buckets (C, IF, OF, DH, SP, RP) before modeling.
7. Four same-class free-agent pairs (McGowan 2014, Murphy 2015, Detwiler 2020, Guillorme 2023): keep or drop.
8. The Baseball-Reference season tables used by `build_features.py` are not committed; the scrape must be re-run to rebuild the feature files.

## Collection (Spotrac, scraped Nov 10 2026)
`src/scrape_spotrac_console.js` runs in the browser console on a Spotrac MLB page (about 60 requests,
0.5 s apart) and downloads `spotrac_raw_contracts.csv`.
- Free-agent rows: 1,820 across classes 2014-2025 (Soto $765M/15, Ohtani $700M/10, Judge $360M/9 spot-checked).
  Includes 164 $0 rows and 206 under $1M rows to review, 8 apparent duplicates, 304 re-signings (From == To).
- Extension rows: 693 all-time; 340 in scope (289 dated, 51 year-only); 206 are pre-2013 and out of scope.
- Spotrac ignores `?query=` filters. It needs path URLs (`/_/year/2024`, `/_/type/...`, `/_/start/.../end/.../page/N`).
- Deferrals: `src/scrape_spotrac_deferrals_console.js` downloads Spotrac's all-time deferred-money list (104 contracts); `adjusted_aav.py` turns it into `aav_adjusted` (D-12).
- Spotrac player pages also list each contract's signing date and 'YOS @ Signing' (service time). Not scraped (about 1,100 page loads); a possible later improvement for free-agent dates and service time.
