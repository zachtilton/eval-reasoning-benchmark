# Anchor Lenient-Error Consensus Fragments — 2026-09-05

Fragments where `rater_1` and `rater_2` agreed the fragment was **"Not
sound"**, but the anchor called it **"sound"**. Identified in
`reliability/turing_test_report_2026-09-05.md` as the dominant error pattern
behind the anchor's failed Multi-Rater Agreement Statistical Turing Test
(source data: `reliability/multi_rater_output_data.csv`).

| report_id | rater_1 | rater_2 | anchor |
|---|---|---|---|
| 703 | Not sound | Not sound | sound |
| 939 | Not sound | Not sound | sound |
| 1001 | Not sound | Not sound | sound |
| 1058 | Not sound | Not sound | sound |
| 1367 | Not sound | Not sound | sound |
| 2517 | Not sound | Not sound | sound |
| 4116 | Not sound | Not sound | sound |
| 5946 | Not sound | Not sound | sound |
| 6279 | Not sound | Not sound | sound |

## Suggested use

Cross-reference these `report_id`s against the main evaluation database to
pull the fragment text and the anchor's rationale, to check for a shared
pattern in what the anchor's coding rule is letting slide relative to the
human raters' standard for "sound."
