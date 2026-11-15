# Tactical possession prototype

This is a deterministic, **uncalibrated prototype**. It generates a full-match
submission but has no established accuracy advantage over the supplied baseline.
No parameter fitting, score-based selection, or video-based checking was performed.

## Run

From the repository root, after installing the dependencies and extracting assets:

```bash
.venv/bin/python possession/pipeline.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_prototype.csv
.venv/bin/python -m unittest discover -s tests -v
```

Options: `--data-dir` (folder containing tactical/), `--config`, and `--out`.
The script honours `MTAH_DATA_DIR` through games.py when no data-dir is specified.
The pinned environment is in requirements-lock.txt. Given identical input files,
configuration and environment, output generation is deterministic.

## Inputs and indexing

Only tactical ATD metadata, raw tracking and automatic Smart Tagging are read.
No videos, public/private possession labels, corrected tracking, human match events,
or manually observed match boundaries are loaded. Broadcast is not yet integrated.
All rows are retained, including pre-match, halftime and post-match frames.
Raw frame ID minus one gives the submission frame. The source frame count and
pitch dimensions come from the published games.py constants, not private labels.

## Algorithm

1. Convert normalised pitch positions to metres. Differentiate adjacent observed
   positions at the supplied frame rate. Never differentiate across missing samples.
2. Downweight ball observations whose measured motion is unusually large. This is
   soft evidence, not automatic deletion: genuine kicks can create large changes.
3. Score player proximity using a Gaussian distance scale. Co-motion strengthens
   proximity evidence. Aggregate the strongest candidate for each team across a
   short centred window. Report nearest-team distances separately for inspection.
4. Carry earlier team-control evidence forward with exponential decay. Search
   forward for the next candidate control observation; matching teams provide a
   decaying bridge across a gap. Opposing endpoints do not create a same-team bridge.
5. Interpret an explicit allowlist of attacking phases as evidence for the tagged
   team and defensive phases as evidence for its opponent. Conflicting overlapping
   tags cancel; repeated supporting tags saturate. Unknown teams and ambiguous codes
   are ignored. Restart intervals supply soft local context, not dead-ball labels.
6. Build dead-ball evidence from observed off-pitch ball positions and restart
   context combined with outside/quiet-stationary evidence. Missing ball alone and
   stationary ball alone never create positive dead-ball evidence. Off-pitch evidence
   is imperfect: pitch-plane projection and tracker errors remain possible.
7. Resolve H/A/D using a three-state Viterbi path, with a cost for state switches.
   This searches the complete recording and can use future observations. A separate
   forward/backward pass records internal model marginals. These are **not calibrated
   probabilities of correctness**. Emission values are hand-designed log-potentials.

Flow/flight is represented by uncertain evidence between controls, not a fourth
submission state. Possession continuity is a team-level hypothesis, not an explicit
pass classification. No ball altitude is inferred from x/y. Observed and estimated
positions are not merged; this version computes motion and continuity evidence but
**does not yet interpolate missing ball coordinates**.

## Parameters and competition rules

Every exposed parameter in config.json is a fixed engineering assumption for this
initial implementation, **not an externally calibrated value**. For example, the
2 m radius is a proximity scale rather than a confirmed contact threshold; 0.4 s
is a support window rather than a learned duration. Motion scales, evidence weights,
restart context, decay times and transition cost likewise lack external calibration.
The internal candidate threshold is the Gaussian score at one radius multiplied by
its minimum co-motion factor, rather than a separately fitted contact classifier.

There is no training dataset in this version and no learned model. Before tuning
parameters, use an external dataset with appropriate licence and possession labels;
record its provenance and freeze parameters before running the challenge match.
Do not select parameters by challenge public scores, manual video checks, or manually
labelled/corrected stretches of this game. Synthetic tests establish code behaviour,
not football accuracy or compliance approval by the organisers.

## Outputs

- `evaluation/results/tactical_prototype.csv`: all video frames, one H/A/D label each.
- `data/processed/possession_v1/evidence.npz`: observed ball positions/visibility,
  derived velocity, reliability, nearest-team distances, control scores, boundary
  evidence, quiet-stationary evidence, tagging, continuity, emissions, decoded states,
  and internal marginals. State codes are 0=H, 1=A, 2=D.
- `data/processed/possession_v1/run.json`: exact parameters, source/output hashes,
  label counts and calibration status.

These outputs are ignored by Git and must not be published as tracking exports.

## Known limitations / next steps

Dead-ball inference is weak without ball visibility or useful restart context. The
algorithm cannot reliably identify referee decisions from trajectories. Long tracking
gaps remain weakly supported and can inherit earlier states. Forward search is built
from proximity candidates and can mistake incidental touches for control. Same-slot
velocity can be corrupted by player identity switches. A switch cost can erase short
possessions, and the model does not have learned duration distributions. Ball
projection near boundaries can be misleading, especially in flight. No human
half/full-time boundaries are hard-coded, so unscored recording regions can receive
arbitrary possession labels. The file is a prototype, not a claim of perfect labels.

Priorities: external calibration, stronger control evidence, calibrated motion/gap
uncertainty, then automatic broadcast alignment if it adds usable observations.

## Saved externally calibrated candidate

See CALIBRATION.md for external control results and STOPPAGE_DATA.md for the
separate challenge candidate and explicit live/dead source audit. To reproduce:

```bash
.venv/bin/python possession/pipeline.py --config possession/config_external_control_v1.json --out evaluation/results/tactical_external_control_v1.csv
```

This keeps candidate evidence in its own folder. Dead-ball and tagging parameters
remain uncalibrated; control calibration does not establish full submission quality.
