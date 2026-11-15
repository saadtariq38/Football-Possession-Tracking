# GunnersFC: football possession technical submission

Team: **GunnersFC**. Submission: `GunnersFC.csv`, a byte-identical copy of
`tactical_external_stoppage_v1.csv`. Later experimental models did not pass all
replacement checks and are not used in this retained candidate. Supply the final
public repository URL in the submission form.

## Approach and assets

We infer H/A/D from the supplied automatic tactical tracking, its team/slot
metadata and generated Smart Tagging. Normalised x/y is scaled to the documented
105.3 × 68 metre pitch. Velocity comes from adjacent observed coordinates; the
provider speed channel is unused. The final inference path uses neither video,
broadcast tracking, homography, corrected tracking nor possession references.
Starter video-derived kickoff/alignment anchors do not construct our labels.

Soft ball-player proximity provides per-team control evidence. Relative movement
strengthens nearby control; large ball-coordinate jumps reduce reliability.
Centred context, decaying previous control and bridges between matching earlier
and later observations support passes and short gaps. Generated attacking and
defending tags supply bounded team evidence. Viterbi decoding with a switching
cost resolves H/A across the recording, including predicted dead intervals.

A separate logistic live/dead model uses 48 motion/context features. Eight
quantities—ball speed, outside distance, nearest-player distance, collective
player speed, quiet-player fraction, ball observation fraction, reliable-ball
fraction and player observation fraction—are averaged over centred 0.4, 2 and
6 second windows with missing indicators. A binary Viterbi decoder stabilises
live/dead decisions. Its D mask overrides the team sequence. Generated tags do
not enter this learned detector.

## Operational possession rules

H/A include team continuity during passes. Proximity is evidence of control,
not a guaranteed touch. In challenges, relative evidence and the sequence model
choose a side. Each predicted live frame gets a team; there is no explicit neutral
live-ball class. Missing observations remain missing and do not automatically
mean D. Pitch y does not measure ball altitude.

D means inferred dead play from the independent motion detector. Set-piece tags
do not directly create D in the retained candidate. Restarts are predicted
D-to-live transitions; continuous team decoding can initially retain the previous
side. No manually identified half-time/post-match boundaries are supplied. The
candidate emits no X; off-game ground-truth X masking is left to the official
scorer. All submission labels are automatic.

## External labelled data

No challenge ground truth, manually produced challenge labels, provided-video
training/tuning, public challenge scores, or prediction/video comparisons were
used in our development. The organisers' later clarification takes precedence
over older repository suggestions to evaluate on the public window.

**SkillCorner Open Data:** three complete external matches from
[SkillCorner Open Data](https://github.com/SkillCorner/opendata), revision
`4340d274572876239c154c90bc507a9b3250a656`. Provider-generated team possession is
used where known. Matches 2017461 and 2016236 select radius, context, continuity
and switching cost from 17 predefined candidates; 2015213 is held out. Unknown
possession is excluded, never D. Only explicitly detected x/y enters features.
The upstream MIT licence, source material and provenance are retained locally;
credit SkillCorner and PySport. No new human labels were made.

**IDSSE:** five complete external matches downloaded from Bassek, Rein, Weber &
Memmert (2025), DFL / Sportec Solutions, *An integrated dataset of spatiotemporal
and event data in elite soccer*.
[Dataset version 1](https://doi.org/10.6084/m9.figshare.28196177.v1),
[publication](https://doi.org/10.1038/s41597-025-04505-y),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
References are the published provider's explicit ball-status/team-possession
fields. No new manual labelling was performed. Raw observed x/y and reference
labels are stored separately. Provider altitude, speed, possession, status and
events are excluded from feature inputs.

The retained dead model initially fits on J03WPY and selects ridge/switch penalty
on J03WOH, then refits on both. J03WQQ is held out, with 142,345 explicit live/dead
frames. The combined model is also audited against that match's separate team
references. Later experiments train/select on the same two development matches;
J03WR9 and then J03WMX supply successive fresh restart/spell validation. Their
results do not change fitted parameters. All five are disclosed even though the
later experimental models are not used in the retained submission.

Robustness scenarios mask about 43% of ball frames in blocks, drop 15% of player
observations and perturb 1% of remaining ball coordinates. Fault generation sees
no reference labels. Original and degraded scenarios share one match's references
and are not independent validation matches.

## Validation and limitations

SkillCorner held-out known-H/A macro F1 is 0.7942, at 43.5% reference coverage.
On IDSSE J03WQQ, live/dead macro F1 is 0.9853 original / 0.9523 degraded, and
combined H/A/D macro F1 is 0.9077 / 0.8507. These external metrics do not establish
challenge accuracy. Official ranking uses spellF1. On fresh J03WMX, retained v1
spellF1 is 0.50188 / 0.39523. Later v3 improves it to 0.53748 / 0.43963 but fails
the fixed missing-ball restart guard; it remains experimental. See
`possession/SPELL_CALIBRATION.md` for that tradeoff and the fixed protocol.

Limits include different provider possession definitions, occlusion, tracking
noise, airborne projections, long gaps, challenges, restart-team errors and
continuous team carry through D. Challenge tag weights remain fixed assumptions;
external tests lack equivalent tags. Few complete matches support selection and
validation. Internal scores are not calibrated correctness probabilities. We
make no perfect-result or challenge-score claim.

## Reproduction

Use Python 3.11 at the repository root. Place the authorised bundle under
`data/tactical/` as documented in `data/README.md`. Saved weights are included;
external training downloads are unnecessary for inference. Do not include match
data in the public code repository.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-submission.txt
.venv/bin/python -m possession.pipeline \
  --config possession/config_external_control_v1.json \
  --stoppage-model possession/model_external_stoppage_v1.json \
  --out evaluation/results/reproduced_v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/reproduced_v1.csv
```

Expected SHA-256:
`791099fb5195ed0fc8a2c9a60de7fcc5a4cfae9d5899a196819b1c56b4518c3f`.
The CSV contains 199,319 ordered frames: H=74,426, A=51,435, D=73,458, X=0.
Official format validation checks structure and does not load ground truth.

Run `scripts/verify_reproduction.py --python /path/to/clean/environment/bin/python`
to export required source into a temporary directory, expose only the three
automatic input assets, regenerate the CSV, validate it and compare exact hashes.
It saves a local reproduction report. Frozen protocols, configurations, fitted
weights and provenance hashes are under `possession/`; its calibration documents
describe training and validation reproduction.

The reproduction check passed on Python 3.11.16 with NumPy 2.4.6 and pandas 3.1.1
in a separate minimal environment. The exported source had only the three
automatic tactical input files available in its asset directory. The regenerated
CSV was byte-identical and passed the official format validator. The local report
is `data/processed/reproduction_v1.json`.
