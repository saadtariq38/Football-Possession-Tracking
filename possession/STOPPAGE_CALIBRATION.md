# External stoppage calibration v1

The subsequent frozen combined H/A/D and restart-team audit is in
[JOINT_VALIDATION.md](JOINT_VALIDATION.md). It measures the existing candidate;
it does not change the weights or submission saved by this experiment.

The motion-only live/dead model was fitted on external IDSSE observations. No
challenge possession labels, challenge video, manual challenge annotations or
public challenge scores were used for fitting, selecting settings or validation.
The previous challenge candidates remain separate.

## Locked experiment

The match split and search were saved in `stoppage_protocol.json` before scoring:

| Purpose | Match |
| --- | --- |
| Fit initial models | J03WPY |
| Choose ridge and switch penalty | J03WOH |
| Final independent check | J03WQQ |

Ridge candidates were 0.001, 0.01 and 0.1; binary Viterbi switch penalties were
2, 8 and 20 at 5 Hz. The objective was the mean live/dead macro F1 of the original
and synthetically degraded selection scenarios. Ridge **0.001**, penalty **8**
won with **0.9782407501**. The final logistic model was refitted on both development
matches and saved before accessing the held-out reference. Held-out results did
not cause any parameter change.

The first evaluation attempt reached validation before its adapter had finished
writing inputs and stopped with FileNotFoundError before reading validation
labels. Validation resumed using the identical saved model, without fitting or
selection again. The detailed nine-candidate table was not persisted by that
interrupted run; the winning objective was recovered from stdout. This is recorded
explicitly in `data/processed/external/idsse/stoppage_fit_record_v1.json`. The script
now preflights all prepared files and saves the fitting record before validation.

## Inputs and reference separation

`python -m possession.idsse --match ID` streams verified raw XML into separate
`inputs.npz` and `reference.npz` files. Only x/y enters model features. Sportec's
centre-origin metre convention was checked against the installed Kloppy 3.18
`SportecTrackingDataCoordinateSystem`; it is converted to normalised pitch
coordinates using each match's dimensions. Raw speed S, altitude Z, BallPossession,
BallStatus and event fields are excluded from features. Velocity is derived from
adjacent observed coordinates. No positions are interpolated. Period boundaries,
frame ID gaps and timestamp gaps split independent sequences.

BallStatus 1 supplies the external alive reference; 0 supplies dead. Missing or
unexpected status is unknown and masked, never converted to dead. Adapter tests
confirm that changing only status leaves all input arrays identical, and altitude
and provider speed cannot leak into the position/speed channel.

The 48 features are eight quantities averaged over centred 0.4, 2 and 6 second
windows, each accompanied by a missing indicator: ball speed, outside distance,
nearest player distance, mean player speed, fraction of quiet players, ball
observation fraction, reliable-ball fraction and player observation fraction.
Unavailable ball observations leave useful player-movement evidence. The model
does not treat missing ball coordinates as a dead-state label.

Medians, means and standard deviations are fitted on development samples only.
Logistic fitting uses natural class prevalence and L2 coefficient regularisation;
the intercept is not penalised. The binary decoder scales evidence by 5/fps, so
the 5 Hz development search and full 25 Hz inference have comparable timing cost.
Each half is processed separately on external data.

Each match has an original and deterministic degraded scenario. Degradation masks
about 43% of ball frames in randomly placed 1, 3 and 8 second blocks, drops 15% of
individual player observations and perturbs 1% of surviving ball positions with
Gaussian 25 metre offsets. Fault generation never receives reference labels.
This is a simulated robustness check, not a reconstruction of challenge noise.

## Held-out results

J03WQQ contains 142,345 frames: 84,352 alive and 57,993 dead, all explicitly labelled.

| Tracking scenario | Live/dead accuracy | Live/dead macro F1 | Dead F1 |
| --- | ---: | ---: | ---: |
| Original | 98.58% | 98.53% | 98.26% |
| Synthetic degradation | 95.40% | 95.23% | 94.30% |

For context, the previous control candidate's geometry-based dead detector scored
57.13% / 51.36% macro F1 on these same original/degraded external observations.
That baseline received no Smart Tagging evidence because IDSSE does not provide
the challenge's tags. These are binary external comparisons, **not challenge
H/A/D scores**.

On original observations, 94 of 99 stop transitions and 94 of 97 resume transitions
were matched within 2 seconds. There were 102 predicted stops and 100 predicted
resumes. With synthetic degradation, the respective matched counts were 76/99 and
78/97, with 99 predicted stops and 97 predicted resumes. Matching is one-to-one
with a fixed two-second tolerance; segment reports retain matched timing errors
and unmatched counts rather than hiding failures behind a mean timing value.

Full confusion matrices, per-half transition diagnostics and source hashes are in
`data/processed/external/idsse/stoppage_report_v1.json`. The frozen fitted weights,
preprocessing, feature schema, source attribution and protocol hash are in
`model_external_stoppage_v1.json`.

## Challenge integration and limits

The new candidate uses the frozen external control configuration for H/A and the
external binary detector for live/dead. D overrides the independent team sequence;
H/A decoding currently runs across predicted stoppages. Smart Tagging contributes
to the unchanged team evidence but is not an input to the learned stoppage model.
No challenge period boundaries or manually identified breaks are supplied.

Internal team marginals and logistic scores are diagnostic model support, not
calibrated probabilities of correctness. One held-out match and simulated faults
cannot establish transfer to the challenge camera, track identities, observation
patterns, half-time or post-match footage. No challenge video comparisons or
public-score tuning were performed. This is a separately saved candidate, not a
claim of perfect labels or final challenge accuracy.

The saved `evaluation/results/tactical_external_stoppage_v1.csv` passes the official
format validator: 199,319 rows (frames 0–199,318), H=74,426, A=51,435, D=73,458,
X=0. These counts describe its output and are not an accuracy check. Per-frame
motion evidence, raw dead logits, separate team states and final states are saved
under `data/processed/tactical_external_stoppage_v1/`. The earlier prototype and
external-control CSV hashes were rechecked and remain unchanged.

## Reproduction

```bash
.venv/bin/python scripts/fetch_idsse_sample.py --match J03WPY
.venv/bin/python scripts/fetch_idsse_sample.py --match J03WOH
.venv/bin/python scripts/fetch_idsse_sample.py --match J03WQQ
.venv/bin/python -m possession.idsse --match J03WPY
.venv/bin/python -m possession.idsse --match J03WOH
.venv/bin/python -m possession.idsse --match J03WQQ
# Fresh experiment only; refuses to overwrite the frozen model or held-out report.
.venv/bin/python -m possession.calibrate_stoppage
.venv/bin/python -m possession.pipeline \
  --config possession/config_external_control_v1.json \
  --stoppage-model possession/model_external_stoppage_v1.json \
  --out evaluation/results/tactical_external_stoppage_v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_external_stoppage_v1.csv
.venv/bin/python -m unittest discover -s tests
```

## Attribution

Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*. DFL / Sportec Solutions. Dataset version 1:
https://doi.org/10.6084/m9.figshare.28196177.v1 . Paper:
https://doi.org/10.1038/s41597-025-04505-y . Licensed CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . Derived weights and diagnostics
here use the external x/y and status data as described above; raw downloads and
generated per-frame files remain ignored by Git. Retain this attribution with
distributed derived materials.
