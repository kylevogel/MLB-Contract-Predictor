# Decision log: MLB contract dataset

Purpose: every choice that shapes the dataset, who made it, what else was considered, and the evidence.
Dates are 2026-11-10 (Copenhagen time). "User" = the team member in the chat. "AI" = Claude (planning agent).
Status: **Decided** (user said so), **Default** (AI proposed, user did not object, confirm later), **Open**.

## Decisions

| ID | Decision | Made by | Alternatives considered | Why / evidence | Status |
| --- | --- | --- | --- | --- | --- |
| D-01 | Scope: every guaranteed MLB contract after a player's rookie deal, free agents and extensions, offseasons 2014 to 2025. Targets: contract length (`years_new`) and nominal AAV. | User (10:15, 10:19) | Free agents only (original proposal) | Extensions add about 320 rows and a second market. | Decided |
| D-02 | Excluded: arbitration contracts, minor-league / non-roster deals, international signings. | User (10:19) | Keep arbitration as its own type | Arbitration salaries are set by a hearing process, not negotiated. | Decided |
| D-03 | Extensions are kept and tagged with Spotrac's three types: `extension_pre_arb`, `extension_arb`, `extension_veteran`. For extensions, `years_new` and AAV count only NEW years and NEW money from the signing date. Where Spotrac disagrees with the majority of independent sources, the majority is used. Applied once: Trout 2019 is 10 yrs / $360M (Spotrac says 12 / $426.5M because it includes 2 years already owed). | User (10:19, 10:51) | Keep Spotrac's full span and flag it | MLB Trade Rumors, FanGraphs, CBS Sports, NBC Sports, SABR/The Ringer and MLB.com all describe it as 10 yrs / $360M of new money; Spotrac and a few headlines use the 12-year total. Only Trout was checked this way. Other extensions with overlapping years may exist (see Open items). | Decided |
| D-04 | Store both nominal and adjusted AAV; models predict nominal. | User (10:19) | Predict adjusted | Nominal is what is announced; adjusted differs only for deferrals. | Decided |
| D-05 | A contract belongs to the offseason class of its signing date: Nov 1 (Y) to Oct 31 (Y+1) is class Y. The last class is 2025 (Nov 1 2025 to Oct 31 2026), so in-season 2026 extensions count and anything from Nov 1 2026 on does not. Free agents have no signing date on Spotrac, so class = contract start year minus 1. | User chose the last class; AI proposed the window | Class by Spotrac's year | Spotrac's year is the contract START year, not the signing year (found when the 2027 page already held deals signed in 2026). | Default, confirm |
| D-06 | A player needs at least one completed MLB season before the signing date. Pre-debut prospects, international imports, and extensions signed in a debut season (e.g. Julio Rodriguez, Michael Harris II) are dropped. Partial-season stats are NOT used. | User (10:51 for debut-season and partial seasons); AI for the rest | Use partial-season stats up to the signing date | Keeps features simple and leak-free. Costs 57 rows, most of them pre-arbitration extensions (pre-arb falls to 61 rows). | Decided |
| D-07 | Drop later same-class free-agent rows worth under 50% of the player's biggest deal (15 rows). Drop deals below that season's MLB minimum salary. Flag (do not drop) contracts at the league minimum (`at_league_minimum`, 68 rows). | AI proposed; user asked for verification (10:51) | Keep one row per signing event | Verified: all 15 dropped rows are one-year deals at exactly that season's minimum, signed after the player's first deal that class (release-and-pickup, not a market price). Minimum salaries verified against AP/MLBPA, Baseball-Reference and Spotrac (2015: $507.5K, 2016: $507.5K, 2017: $535K, 2018: $545K, 2019: $555K, 2020: $563.5K, 2021: $570.5K, 2022: $700K, 2023: $720K, 2024: $740K, 2025: $760K, 2026: $780K). | Decided |
| D-08 | Ohtani is excluded entirely (both his 2020 arbitration extension and 2023 free agency). No two-way players remain. | User (10:51) | Model as hitter; keep as outlier | Only one two-way player; his $700M deal is 97% deferred and would distort everything. | Decided |
| D-09 | Time split by class: train 2014-2021, validate 2022-2023, test 2024-2025 (used once). Hitters and pitchers modelled separately. The neural network is compared honestly with linear and boosted-tree baselines; if trees win, we report that. Success metric: open. | Proposal; user confirmed baseline comparison (10:51) | | With about 1,900 rows, trees may well beat the network. | Metric open |
| D-10 | Add previous-contract features (previous type, years, total, AAV, adjusted AAV, years since). Do NOT add qualifying-offer flags. | User (10:51) | Qualifying-offer flag (AI suggested; rejected) | The previous contract is the strongest single price anchor. Only contracts in this dataset count, so 42% of rows have one (rookie/arbitration salaries and pre-2014 deals are missing). Only strictly earlier classes are used, so no same-class leakage. | Decided |
| D-11 | Names are matched to MLB player IDs through the Chadwick register, with 9 manual overrides (same-name players, nicknames) and 3 players dropped as never in the MLB register. | AI, user not asked | Match by Spotrac ID only | No ID crosswalk exists between Spotrac and the MLB data. | Default |
| D-12 | Adjusted AAV: contracts with no deferral have adjusted = nominal. For the 69 contracts with a valid Spotrac deferral, deferred money is discounted to the signing date at 3% a year (equal payments each Nov 30). It is an estimate, not the official luxury-tax figure. | User asked for it (10:51); AI chose the method | Skip it; scrape player pages (the luxury-tax tables are hidden behind a login for expired contracts) | Published present values imply: MLB's own luxury-tax AAV for Ohtani 2.9%, MLBPA 3.2%, Betts 3.6%, Strasburg 2.5%, Tucker 3.3%, Lindor 1.2% (outlier). The MLBPA's average-salary convention uses 5%. Test: Ohtani at 3% is within 5% of MLB's $46.08M. Two Spotrac deferrals larger than the contract itself are ignored. | Decided (rate: confirm) |
| D-13 | Publishing Spotrac data in a public repo. Spotrac's terms allow downloading for personal non-commercial use but say not to "further reproduce, publish or distribute". Raw Spotrac files are kept out of the repo; the scraper and code are in. Whether to publish the cleaned dataset is open. | Open | Publish everything; publish only facts-only data; ask the professor | The course needs a dataset link for reproducibility. | Open |

## AI mistakes caught and fixed (process evidence)

| # | What went wrong | How it was caught | Fix |
| --- | --- | --- | --- |
| A-1 | Rows with no date or year were silently dropped (206 pre-2013 extensions) because a missing value made the filter keep nothing. | Row accounting: 2,512 in, 1,882 + 424 logged out did not add up. | Missing values now drop with a logged reason; the accounting check is run every time. |
| A-2 | Ages came out a year too high (Soto 27.1 instead of 26.1). | Test on known players. | Reference date corrected. |
| A-3 | Early spec defined the class by offseason but never said how in-season deals map; Spotrac's "year" was assumed to be the signing year. | The 2027 Spotrac page already held 2026 signings. | Date-based rule (D-05). |
| A-4 | Extension type `extension_pre_fa` was invented before seeing Spotrac's categories. | Spotrac's dropdown showed Pre-Arbitration / Arbitration / Veteran. | Renamed `extension_veteran`. |
| A-5 | Zero-value minor-league rows were logged as "zero years" because of rule order. | Drop counts changed unexpectedly (161 vs 55). | Rule order fixed. |
| A-6 | Name matching lost 46 real MLB veterans (initials, nicknames, common names). | Reviewed the unmatched list. | Alias keys plus 9 manual overrides; 3 confirmed not in the MLB register. |
| A-7 | Validation flagged 49 false duplicates because free agents share an empty signing date. | Validation output. | Duplicate key now includes class and contract type. |
| A-8 | The first Ohtani drop removed only his 2023 row; a 2020 arbitration extension stayed. | Final check on the cleaned file. | Player-level exclusion. |
| A-9 | The proposal slides call Soto a "2025 signing"; he signed December 2024 (class 2024). | Spotrac and MLBTR both date it Dec 2024. | Use class, not calendar year, in the blog. |

## Open items
1. Success metric (D-09): AI suggests MAE, share within 20% of real AAV, and R-squared on log AAV, instead of "within 5%".
2. Extensions other than Trout may include overlapping years; only the 24 headline contracts were cross-checked (21 exact matches, 3 explained). A larger random check is a good first issue.
3. Spotrac redistribution (D-13).
4. The saved scraper (`src/scrape_spotrac_console.js`) is untested end to end.
5. `fetch_stats.py` has not been run on real data (FanGraphs is unreachable from the AI workspace).
6. 68 free-agent rows sit exactly at the league minimum; they are kept and flagged.

## Sources used for verification
MLB.com "largest free-agent contracts" (Feb 2026); MLB Trade Rumors "Largest contracts in MLB history" (Dec 2025) and its 2019 Trout posts; FanGraphs "Mike Trout Leaves Money on the Table Again"; CBS Sports and NBC Sports Trout reports; SABR/The Ringer (Lindbergh); Boardroom top-40 list; CBS Sports largest contracts; AP/MLBPA salary tables (theScore); Baseball-Reference Bullpen "Minimum salary"; Spotrac "Breaking down the new MLB CBA"; Spotrac terms of service.
