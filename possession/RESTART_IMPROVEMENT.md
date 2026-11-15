# External restart candidate v2a

**Submission-readiness amendment:** the later official spellF1 diagnostic below
found a ranking-metric regression under simulated faults. Passing the original
frame/restart gate does not establish submission-ranking readiness. Do not treat
v2a as an established improvement over v1 for the competition's primary metric.

The new candidate improves whole-match and overall restart decisions on a fresh
external match. It also reduces delayed D release on missing-ball restart samples.
It is **not a uniform improvement**: correct H/A/D decisions on that small
missing-ball subset decreased. Both the earlier and new challenge candidates are
preserved; no challenge accuracy or perfect submission is claimed.

## Fresh comparison

All numbers below compare the frozen v1 candidate with the frozen v2a candidate
on the **same new external match J03WR9**, excluded from fitting and parameter
selection. These numbers must not be compared directly with percentages on the
previous audit match J03WQQ.

| Metric | Original v1 | Original v2a | Faulted v1 | Faulted v2a |
| --- | ---: | ---: | ---: | ---: |
| Whole-match H/A/D accuracy | 92.59% | 93.42% | 87.30% | 88.40% |
| Whole-match H/A/D macro F1 | 91.11% | 92.12% | 85.43% | 86.77% |
| Accuracy in +/-5s restart windows | 88.08% | 90.06% | 79.94% | 81.10% |
| Correct H/A at 0.5/1/2s after restart, pooled | 201/266 (75.56%) | 221/266 (83.08%) | 165/266 (62.03%) | 175/266 (65.79%) |
| Correct H/A at exactly +1s | 68/89 (76.40%) | 76/89 (85.39%) | 55/89 (61.80%) | 59/89 (66.29%) |
| Dead F1 | 98.77395% | 98.77395% | 95.33889% | 95.33769% |

The equal-scenario gain at the three main restart offsets is **5.64 percentage
points**. The original and faulted scenarios share one match's reference labels;
they are not two independent validation matches. Faults use the previously fixed,
label-independent block gaps, player dropouts and ball jumps.

## What changed

**Team evidence:** a small logistic model uses signed home-minus-away proximity,
relative motion, closest approach under constant relative velocity, prior/future
control anchors and past/future context. Every input reverses sign when teams are
swapped. Paired team-swapped training examples enforce zero team intercept and
zero feature mean. Neither team identity, absolute match time, altitude nor
possession labels enters inference.

Development selected ridge **0.01**, **zero additional Viterbi switch penalty** and
continuous decoding. The input context still smooths evidence over time. Explicitly
resetting the team decoder at predicted stoppages was tested and was not selected;
the code does not blindly reset ownership at every restart.

**Dead release:** the broad replacement live/dead models all failed development
guards, so the frozen v1 stoppage model was retained. A separately specified,
bounded refinement was then developed before any fresh-match scoring. It looks
only near an existing predicted D->live boundary and only starts a correction
when the ball is missing and there is sustained evidence of play resuming.

The refinement uses the original dead score, directional player speed/active
fractions, future ball availability and time to the original predicted restart.
Development selected ridge **0.01**, a maximum **2-second** advance, dead probability
below **0.25**, and **0.4 seconds** of consecutive qualifying missing-ball evidence.
It can only remove D near an existing predicted restart. It cannot create D,
invent a new restart, fill coordinates, or clear a terminal dead run without a
predicted live successor. The two-second limit is externally selected, not a
manual challenge boundary.

Challenge Smart Tagging retains its prior fixed 1.2 contribution to home log-odds;
that strength was not calibrated here. All external comparisons use zero tag
evidence because IDSSE does not contain the challenge's Smart Tagging.

## Missing-ball results and tradeoffs

The 266 main offset samples include **116 missing-ball samples** in the faulted
scenario. Offset samples from one restart are correlated, not independent trials.

| On those 116 missing-ball samples | v1 | Unrefined new team model | Final v2a |
| --- | ---: | ---: | ---: |
| Incorrectly still D | 37 (31.90%) | 37 (31.90%) | 34 (29.31%) |
| Correct H/A/D decision | 49 (42.24%) | 42 (36.21%) | 44 (37.93%) |

The release patch fixes three delayed-D samples, two of which become correct team
decisions. However, the learned team model has seven fewer correct decisions than
v1 in this subset before the patch, so the final net change remains **five fewer
correct decisions**. It would be misleading to describe missing-ball team
selection as solved. The next development target is improving team decisions
during gaps without discarding the gains on observed-ball restarts.

The release patch also has a small whole-match tradeoff. Compared with the same
new team model without refinement, faulted whole-match accuracy decreases from
88.4136% to 88.4000%, a difference of 20 frames. It removes 72 false-D frames but
adds 67 missed-dead frames. Of the 72 released reference-alive frames, 47 get the
correct team and 25 get the wrong team. Binary accuracy therefore slightly rises
while combined H/A/D accuracy slightly falls. The primary candidate was fixed
before validation; no ablation was retrospectively selected as the winner.

At the exact restart instant, original-track accuracy also falls from 47/90 to
46/90. The prespecified primary restart metric uses 0.5/1/2 seconds; the report
retains 0 and 5 seconds so this limitation remains visible.

## External calibration and split

| Role | Match |
| --- | --- |
| Initial fitting | J03WPY |
| Development settings selection | J03WOH |
| Previous audit, excluded from this iteration | J03WQQ |
| Fresh final validation | J03WR9 |

`restart_protocol_v2.json` specifies the first experiment. Two ridge values,
two stop penalties and two missing-score shifts produced eight new stop options
plus the frozen baseline. Two team ridge values, three team penalties and two
sequence modes produced twelve new team options plus the legacy path.

The team objective combines whole-alive accuracy and restart accuracy, guarded
against whole-match regression. The broad stop options were rejected on those
prespecified development safeguards despite some improvements in their targeted
objective. The retained v1 stop model had previously been refitted on both
development matches; these development comparisons therefore favour that
established baseline and are not independent validation claims.

`restart_release_protocol_v2a.json` then fixed the targeted refinement before fresh
scoring. Two ridge values, three advance limits and two probability thresholds
produced twelve release options plus no change. It selects reduced missing-ball
false-D rate at restart offsets, with guards on whole-match accuracy, dead F1 and
dead recall. All parameter selection used only the two development matches.

Selected learned heads were refitted on both development matches. The base model
and refinement were saved and hashed before fresh scoring. J03WR9 was downloaded
from official version-1 Figshare files; official MD5 and size plus local SHA-256
were checked. Its joint references were built separately after the models were
frozen. Predictions for all four scenario/half combinations were saved before
opening the fresh joint reference arrays for scoring.

J03WR9 has 146,810 explicit references: H=53,239, A=32,077, D=61,494, unknown=0.
Raw BallStatus=0 supplies D; alive status with BallPossession=1/2 supplies H/A.
Other cases would be masked. Original inputs and binary reference files were not
rewritten. No manual labels or challenge ground-truth/video tuning were used.

## Prespecified fresh gate

The candidate passed the gate fixed before validation:

- Mean original/faulted restart-offset accuracy improves by at least 1 point:
  measured +5.64 points.
- Whole-match H/A/D accuracy is no worse than v1 minus 0.2 points in either
  scenario: measured gains of +0.83 and +1.10 points.
- Dead F1 is no worse than v1 minus 0.5 points in either scenario: unchanged on
  original tracking, -0.0012 points with faults.
- Missing-ball restart false-D rate does not increase where eligible: decreases
  by 2.59 points with faults; no eligible original-track missing-ball samples.

This gate did not require noninferiority of missing-ball **team accuracy**. Passing
it is not evidence that every subset improved. The regression above is retained
as a material limitation, not hidden by the gate.

The full report includes v1, team-only change, stop-only change, combined v2a and
unrefined v2. It also retains per-half transition matching, confusion matrices,
owner-change strata, missing-frame metrics and censored recovery diagnostics.

## Subsequent official ranking-metric diagnostic

A submission-readiness review identified that the challenge ranks by **spellF1**,
not the frame and restart objectives used for this development iteration. That
should have been checked before describing the candidate's external gate as a
general improvement. The models and previous reports were left unchanged.

`possession.audit_spell_readiness` applies the actual repository
`evaluation/possession_metrics.py` spell functions to the **saved external**
predictions. It preserves half/gap boundaries, matches possessions over the five
official IoU thresholds and records 1,000-draw possession bootstrap intervals.
No challenge ground truth or video is read. This is a post-hoc diagnostic on the
same external validation match, not a new independent test or parameter search.

| External scenario | v1 spellF1 | v2a spellF1 | Reference spells | v1 / v2a predicted spells |
| --- | ---: | ---: | ---: | ---: |
| Original observations | 0.5483 | 0.5610 | 370 | 388 / 363 |
| Simulated faults | 0.4344 | 0.3806 | 370 | 281 / 436 |

The official scorer's 5th–95th percentile bootstrap intervals are 0.5124–0.5826
and 0.5250–0.5975 on original observations, and 0.3914–0.4750 and 0.3454–0.4169
under faults, for v1 and v2a respectively. These intervals are not a paired test
of the difference and do not turn one match into a broad generalisation claim.

The degraded ranking score and excess predicted spells reveal a fragmentation
problem that improved frame accuracy did not capture. **V2a is not established as
the best submission candidate.** Any further development should use spellF1 on
external development data as the primary objective and reserve another untouched
external match before making a stronger ranking claim. No ablation or new
hyperparameter has been selected from this diagnostic.

Full diagnostic: `data/processed/external/idsse/restart_v2a_validation/spell_readiness.json`.
It records the official metric source hash, saved validation hash, per-IoU scores,
spell counts, durations and bootstrap intervals. The existing CSVs remain intact.

## Saved candidate and artifacts

`evaluation/results/tactical_external_restart_v2a.csv` passes the official format
validator: 199,319 rows, H=73,546, A=52,326, D=73,447, X=0. The release refinement
changes 11 frames relative to its unrefined base on the challenge inputs. These
are output diagnostics, not a correctness check. All earlier candidates remain
unchanged.

Candidate SHA-256:
`75c6bd33fd8e35356314d8d74cfa21806302dacf05b5fb5ee7cb8d4d977e1654`.

| Artifact | Location |
| --- | --- |
| Frozen team/base model | `possession/model_external_restart_v2.json` |
| Frozen bounded release model | `possession/model_external_release_v2a.json` |
| Base development search | `data/processed/external/idsse/restart_v2_fit.json` |
| Release development search | `data/processed/external/idsse/restart_release_v2a_fit.json` |
| Fresh validation and ablations | `data/processed/external/idsse/restart_v2a_validation/report.json` |
| Saved fresh per-half predictions | Same ignored validation folder |
| Challenge evidence and run hashes | `data/processed/tactical_external_restart_v2a/` |

Base model SHA-256:
`08b8b8291f6e78107d5a4ce91bdd413373e7c3ae9b02f8d0db1e533e9e82a9c7`.
Release model SHA-256:
`3b07162577b217f8ea856772125e9ea627023051ea0e2106d50f704421dba3f4`.

The weights suffice to regenerate the challenge candidate without external
training downloads:

```bash
.venv/bin/python -m possession.run_restart
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_external_restart_v2a.csv
.venv/bin/python -m unittest discover -s tests
```

The runner refuses to overwrite an existing candidate. External calibration and
fresh validation can be reproduced in a fresh experiment checkout using
`possession.calibrate_restart`, `possession.calibrate_release`, then
`possession.validate_restart --release-model possession/model_external_release_v2a.json`.
They refuse to replace frozen experiment artifacts. Do not repeat selection on
the freshly evaluated J03WR9 match. A subsequent model iteration needs another
untouched external validation match.

## Scope and attribution

One fresh match and simulated faults do not establish transfer to challenge
tracking, camera coverage, possession conventions, half-time or post-match data.
Raw model logits and conditional release probabilities are diagnostic support,
not calibrated correctness probabilities. No challenge reference score or
algorithm-output/video comparison was used. Nothing was submitted or pushed.

External source: Bassek, Rein, Weber & Memmert (2025), DFL / Sportec Solutions,
*An integrated dataset of spatiotemporal and event data in elite soccer*.
[Paper](https://doi.org/10.1038/s41597-025-04505-y),
[IDSSE version 1](https://doi.org/10.6084/m9.figshare.28196177.v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Derived weights and diagnostics use external x/y and explicit status/team fields
as described above. Retain attribution with distributed derived materials.
Per-frame external data, challenge assets and generated submissions remain
ignored by Git.
