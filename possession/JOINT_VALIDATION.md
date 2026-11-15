# Frozen external H/A/D and restart audit

The subsequent development iteration and fresh-match comparison are documented
in [RESTART_IMPROVEMENT.md](RESTART_IMPROVEMENT.md). The frozen audit below remains
unchanged; it was not used for selecting the new model's parameters.

The combined candidate reaches **91.73% H/A/D accuracy** on original external
tracking and **86.18%** under the existing simulated faults. Accuracy in five-second
windows on each side of restarts is lower: **87.38%** and **79.21%**. Team selection
immediately after restarts is a material remaining weakness; binary live/dead
accuracy alone did not expose it.

This audit changed **no model weights, control settings, inference code or challenge
submission**. It does not estimate challenge accuracy. The challenge candidate
remains `evaluation/results/tactical_external_stoppage_v1.csv`.

## Dataset and reference

We used IDSSE match **J03WQQ**, the match excluded from all previous model fitting
and parameter selection. Its binary live/dead results were already inspected in
the stoppage experiment. This is a new prespecified diagnostic endpoint on that
same held-out match, **not a newly independent validation dataset**.

The reference-only extension reads raw ball attributes as follows:

| Raw fields | Reference |
| --- | --- |
| BallStatus=0 | D, regardless of possession field |
| BallStatus=1 and BallPossession=1 | H |
| BallStatus=1 and BallPossession=2 | A |
| Other/missing status or alive without explicit team | Unknown, excluded |

These meanings are documented in the [dataset paper](https://www.nature.com/articles/s41597-025-04505-y).
Provider possession follows control semantics, which may differ from challenge
ownership during loose balls or passes. No manual annotation, forward fill or
absent-possession-to-dead conversion was used.

There are 142,345 explicit references: H=37,030, A=47,322, D=57,993, unknown=0.
The new `reference_had_v1.npz` is separate from both the existing binary reference
and observed-coordinate inputs. Their original SHA-256 hashes were preserved.
Source XML checksums and frame/timestamp alignment were verified.

## Fixed procedure

`joint_validation_protocol.json` fixed the metrics, windows, scenarios and frozen
file hashes before computing joint results. Both methods use the production
functions at 25 fps with each half processed separately. A synthetic integration
test verifies that the external inference function produces exactly the CSV
decisions from both production run modes.

Both methods receive zero Smart Tagging evidence because that challenge-specific
input is unavailable in IDSSE. The current method decodes the frozen H/A evidence
and then applies the frozen binary D mask. The legacy control method uses its old
three-state geometry-based dead evidence. The original and degraded scenarios
reuse the exact severity and deterministic seeds from the earlier stoppage audit;
these are two observations of one match, not independent matches.

Inference for both scenarios completed and was saved before opening the joint
reference arrays. Labels only select scoring windows, strata and target states.
Neither model receives status, possession, reference transitions or restart anchors.
No training, parameter search, public challenge scoring or challenge video check
was performed.

## Results

| Metric, current combined model | Original | Simulated faults |
| --- | ---: | ---: |
| Whole-match H/A/D accuracy | 91.73% | 86.18% |
| Whole-match H/A/D macro F1 | 90.77% | 85.07% |
| H F1 | 85.18% | 78.03% |
| A F1 | 88.89% | 82.86% |
| D F1 | 98.26% | 94.30% |
| Restart-window H/A/D accuracy | 87.38% | 79.21% |
| Restart-window H/A/D macro F1 | 83.88% | 75.63% |
| Stop-window H/A/D accuracy | 84.11% | 75.37% |
| Stop-window H/A/D macro F1 | 80.24% | 72.14% |

The legacy method's whole-match H/A/D accuracy is 61.11% / 55.81%, and its restart
window accuracy is 53.94% / 48.21%. Much of the improvement comes from replacing
its weak dead detector. This comparison lacks challenge Smart Tagging for both
methods and must not be represented as a measured challenge improvement.

Restart and stop windows are half-open [-5,+5) seconds around adjacent explicit
D->alive and alive->D transitions, clipped to each source segment. Overlapping
windows are counted once in frame metrics. There are 97 reference restarts and
99 stops, with 24,250 restart-window and 24,632 stop-window frames. Unknown status,
period boundaries and recording edges never create a transition. Macro F1 always
uses the three fixed H/A/D classes, with absent-class F1=0; accuracy and explicit
support counts are also retained for windows where class balance differs.

Whole-match confusion matrices, **reference rows / predicted columns**, order H/A/D:

```text
Original                     Simulated faults
          H      A      D              H      A      D
H     31764   4598    668     H     29889   5547   1594
A      5160  41918    244     A      7585  38626   1111
D       629    477  56887     D      2104   1732  54157
```

## Team selection after play resumes

Each entry is correct H/A decisions divided by eligible reference restarts at that
offset. A prediction of D counts as wrong. Only offsets inside the same reference
live run qualify; the target is the actual explicit team at that offset, not a
team label carried forward from the restart.

| Time after restart | Original | Simulated faults | Eligible restarts |
| --- | ---: | ---: | ---: |
| 0 seconds | 58/97 (59.79%) | 47/97 (48.45%) | 97 |
| 0.5 seconds | 73/97 (75.26%) | 52/97 (53.61%) | 97 |
| 1 second | 76/97 (78.35%) | 61/97 (62.89%) | 97 |
| 2 seconds | 76/97 (78.35%) | 65/97 (67.01%) | 97 |
| 5 seconds | 77/87 (88.51%) | 75/87 (86.21%) | 87 |

At one second, the clean-track failures are **19 wrong-team decisions and 2 false
D decisions**. Under faults, they are **21 wrong-team decisions and 15 false D
decisions**. This supports two separate development targets: H/A control evidence
around restarts and releasing the dead state when the ball is not observed.

The previous owner is the last explicit team in the last second before the dead
run, and restart owner is the first explicit team in the first second of the new
live run. These labels are used only to stratify evaluation:

| Restart stratum | Events | Correct at +1s, original | Correct at +1s, faults |
| --- | ---: | ---: | ---: |
| Same owner across stoppage | 42 | 29/42 (69.05%) | 24/42 (57.14%) |
| Changed owner across stoppage | 55 | 47/55 (85.45%) | 37/55 (67.27%) |

Six clean-track wrong-team decisions at +1s predict the previous owner on a
changed-owner restart; the fault scenario has nine. Previous-owner carryover is
one observed failure pattern, but it does not explain all restart errors. The
same-owner stratum performs worse here, so simply resetting continuity cannot be
assumed to fix the problem.

A further diagnostic asks whether there is **any** continuously correct 0.4-second
H/A run inside the first five seconds after each restart, before the next stop.
95/97 original and 90/97 degraded events meet it; 2 and 7 are censored. Mean latency
among recovered events is 0.368s / 0.750s. This permissive first-correct-burst metric
does **not** imply that the subsequent prediction stays correct; the fixed-offset
accuracy table above remains essential. Unknown reference frames cannot qualify.

## Error attribution

On 84,352 reference-alive frames, the team sequence before applying the D mask is
correct on 88.41% / 84.07%. After the D mask, combined alive accuracy is
87.35% / 81.23%.

| Error on the whole match | Original | Simulated faults |
| --- | ---: | ---: |
| Wrong H/A on a reference-alive frame | 9,758 | 13,132 |
| D predicted on a reference-alive frame | 912 | 2,705 |
| H/A predicted on a reference-dead frame | 1,106 | 3,836 |
| Previously correct team decision overridden by D | 893 | 2,400 |

The last row is a subset of the false-D errors, not an additional error category.
These counts make clear that strong D detection leaves substantial H/A work.

The binary confusion matrices and every per-half stop/resume matching statistic
exactly reproduce the earlier frozen stoppage audit. Saved current states were
checked against the saved H/A sequence plus D override. All frozen input, model,
configuration, inference-code, prior-report and challenge-candidate hashes still
match the protocol.

## Artifacts and reproduction

- Protocol: `possession/joint_validation_protocol.json`.
- Reference-only adapter: `possession/idsse_joint_reference.py`.
- Reference-side metrics: `possession/joint_metrics.py`.
- Frozen audit runner: `possession/validate_joint.py`.
- Full report: `data/processed/external/idsse/joint_audit_v1/report.json`.
- Four saved scenario/half prediction archives: the same ignored audit folder.
- Tests cover unknown labels, metric aggregation, segment boundaries, deduplicated
  windows, offsets crossing stops, censored recovery and parity with production.

```bash
# On an already prepared external match; each command refuses to overwrite its result.
.venv/bin/python -m possession.idsse_joint_reference --match J03WQQ
.venv/bin/python -m possession.validate_joint
.venv/bin/python -m unittest discover -s tests
```

The external source remains CC BY 4.0: Bassek, Rein, Weber & Memmert (2025),
DFL / Sportec Solutions; [dataset version 1](https://doi.org/10.6084/m9.figshare.28196177.v1).
Retain attribution with distributed derived materials. Per-frame external data
and challenge outputs remain ignored by Git.

## Next development target

Develop restart-aware H/A evidence and missing-ball live/dead transitions on the
existing external development matches, with a new, untouched external match
reserved for the next model's final validation. Do not select parameters on this
audit match. Any proposed change should show fixed-offset restart gains without
hiding extra false D decisions or losses in whole-match H/A/D accuracy.
