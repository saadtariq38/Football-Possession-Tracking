# SpellF1 experiment v3: retain v1 fallback

The competition ranks possession spellF1. V3 improved that metric on a fresh
external match but failed the previously fixed restart guard. **V3 remains
experimental; v1 remains the conservative submission fallback.** No challenge
CSV was generated for v3 and no settings changed after fresh validation.

## Selection before validation

`spell_protocol_v3.json` fixed the objective, grid, split, tie breaks and acceptance
checks. Only IDSSE J03WPY and J03WOH were used for selection. Previously seen
J03WQQ and J03WR9 were excluded from fresh validation. J03WMX was the new validation
match. Original and degraded scenarios share references and are not independent
matches. Fault generation receives no labels.

The search retained frozen v1 dead decisions and compared frozen legacy/learned
team evidence, observation-dependent blending, smoothing, switching costs and
resets at predicted dead boundaries. It fitted no new classifier. There were 121
coarse candidates at 5 Hz; the top eight admissible candidates plus exact v1 were
checked at 25 Hz. The full development record is saved locally under
`data/processed/external/idsse/spell_v3_development.json`.

Selection chose learned team evidence, switch penalty 2 at the 5 Hz convention,
no smoothing, and separate decoding of predicted live blocks. Mean development
spellF1 improved from **0.47738 to 0.52292**, passing development guards.
`config_spell_v3.json` records the selected settings and protocol, inference code,
official metric and base-model hashes. It was saved before fresh scoring. Its
historical "validation pending" status records that freeze point; the final
validation report records the completed result without rewriting the configuration.

## Fresh external result

J03WMX has 145,967 explicit references: H=39,111, A=46,258, D=60,598. Halves
are separate sequences. All predictions were saved before the scoring routine
opened reference arrays. Reference status/possession never enters input features.

| Metric | v1 original | v3 original | v1 degraded | v3 degraded |
| --- | ---: | ---: | ---: | ---: |
| Official spellF1, mean IoU 0.3–0.7 | 0.50188 | 0.53748 | 0.39523 | 0.43963 |
| H/A/D macro F1 | 0.91074 | 0.92072 | 0.85817 | 0.87072 |
| Predicted possessions | 447 | 292 | 328 | 246 |
| Correct missing-ball restart-offset H/A/D | n/a | n/a | 59/136 | 56/136 |

Mean scenario spellF1 improved **0.04000**, exceeding the required 0.02. Each
scenario passed spellF1 and macro F1 noninferiority. However, missing-ball restart
accuracy fell from 43.38% to 41.18%: **2.21 percentage points** against a maximum
permitted loss of 2 points. The complete gate failed. Offset samples include
multiple observations of the same restarts and are not independent events.

For context only, v2a scored 0.49218 original / 0.36341 degraded spellF1 on this
match; it was not adjusted or reselected. V3 favours longer possessions and still
misses many reference spells. One fresh match cannot establish transfer to the
challenge tracking or its generated tags.

The full report, per-IoU scores, bootstrap intervals, counts, provenance and saved
predictions are local ignored artifacts under
`data/processed/external/idsse/spell_v3_validation/`. Do not rerun parameter
selection against this now-seen match. A next experiment needs its own external
development work and untouched validation split.

## Reproduction and attribution

Download and adapt J03WPY, J03WOH and J03WMX using the existing IDSSE fetcher and
adapter. On a fresh experiment checkout, `python -m possession.calibrate_spell`
freezes development selection; `python -m possession.validate_spell` performs the
fresh check. These commands refuse to overwrite completed experiment artifacts.
They neither generate nor score challenge submissions.

Bassek, Rein, Weber & Memmert (2025), DFL / Sportec Solutions, *An integrated
dataset of spatiotemporal and event data in elite soccer*. Dataset DOI
10.6084/m9.figshare.28196177.v1; paper DOI 10.1038/s41597-025-04505-y; CC BY 4.0.
Retain attribution with derived materials.
