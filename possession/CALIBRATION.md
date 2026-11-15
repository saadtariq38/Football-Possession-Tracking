# External control calibration v1

## Scope and frozen protocol

Only external SkillCorner observations and known home/away provider references
were used. Development matches: 2017461, 2016236. Held-out validation: 2015213.
The split and 17 candidates were written before scoring. Candidate 0 is the
original prototype; the 16 alternatives cross radius [1.5, 2.5] m, centred support
[0.2, 0.6] s, linked forward/bridge decay [1, 3] s and switch cost [2, 4]. All
other settings remain fixed. Objective: equal-match average known-H/A macro F1.
Ties select the first declared candidate. No additional search followed validation.

Source revision: `4340d274572876239c154c90bc507a9b3250a656`.
Frozen configuration SHA-256: `28d702b8f687a8b4296d334d7ff72219110a21113f2712058e3cbbcab2bea429`.
Protocol SHA-256: `612f2f7b8c190b9318e664e19b3f1267241d459b3551b8b9abe2a47d2178d2fd`.

The production decoder has an optional Viterbi-only path for search; synthetic
checks confirm it returns the same states as the full marginal computation.
Emissions are weighted by 25/source_fps to integrate evidence on a common temporal
scale; decoding and motion reset at each period. Inputs exclude provider possession,
phases, restart tags, extrapolated positions and altitude. Unknown references are
excluded, never relabelled D. No ground-truth references enter feature generation.

## Results

| Metric | Original prototype settings | Frozen selected settings |
| --- | --- | --- |
| Development mean known-H/A macro F1 | 0.7651 | 0.7807 |
| Held-out known-H/A macro F1 | 0.7703 | 0.7942 |
| Held-out known-H/A accuracy | 0.7711 | 0.7946 |
| Held-out macro F1, ball observed subset | 0.7676 | 0.7909 |
| Held-out macro F1, ball missing subset | 0.7965 | 0.8264 |

These are comparisons against **our original prototype settings**, not the supplied
competition baselines. Validation used 31,349 known-reference
frames out of 72,101 total frames (43.5% coverage).
The ball-missing subset contains 2,829 reference
frames. D predictions on known H/A frames count as misses. Unknown frames are not
used to measure accuracy. These are frame metrics, not possession-spell F1.

## Selected parameter changes

| Parameter | Original | Selected |
| --- | --- | --- |
| `control_radius_m` | 2.0 | 1.5 |
| `control_support_seconds` | 0.4 | 0.6 |
| `continuity_decay_seconds` | 2.0 | 3.0 |
| `bridge_decay_seconds` | 2.0 | 3.0 |
| `switch_penalty` | 3.0 | 4.0 |

The selected candidate is index 8 (zero-based). Its configuration is
`possession/config_external_control_v1.json`. The default `config.json` and existing
challenge submission were left unchanged. Full protocol, all development scores,
selection and one held-out evaluation are under ignored
`data/processed/calibration/control_v1/`.

## Interpretation and limits

The held-out result is encouraging for team-control inference, but does not establish
challenge accuracy, calibrated uncertainty or statistical significance across matches.
There are only two development matches and one validation match, and references are
provider-generated with substantial missingness. Label availability may favour easier
situations; extrapolated coordinates were excluded. Definitions and tracking noise may
differ from Metrica. Unknown intervals prevent reliable complete-spell evaluation here.

Dead-ball parameters, Smart Tagging weights, motion reliability scales and other fixed
assumptions were **not** calibrated. This data has no explicit D reference. H/A-only
validation cannot establish dead-ball quality and should not justify suppressing D.
The candidate grid did not vary dead-ball weights. The fps weighting is a time-integration
convention, not proof that settings transfer perfectly from 10 fps to 25 fps.

No challenge labels or video were used, and no challenge-score selection occurred.
The held-out match must not now become a tuning set. Further tuning needs new external
development data and a fresh untouched validation split.

## Commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m possession.calibrate
```

The calibration command refuses to overwrite an existing protocol/configuration,
preventing accidental repeated validation. Do not delete the protocol to retry a
search in response to held-out scores. Source fetch and adaptation instructions
are in EXTERNAL_DATA.md. Challenge regeneration, if desired later, must explicitly
select the frozen configuration and a separate output file; it is not performed
by the calibration command.
