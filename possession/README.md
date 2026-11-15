# How our possession system works

This folder contains **GunnersFC's possession algorithm**: the code that turns
player and ball tracking into a label for every frame of the match.

| Label | Meaning |
| --- | --- |
| H | Home team possession |
| A | Away team possession |
| D | Dead play: the ball is not in play |

The submitted version combines **team-control rules calibrated on external data**
with **a live/dead model trained on external matches**. It generates all labels
automatically. The full submission write-up is in
[TECHNICAL_SUBMISSION.md](../TECHNICAL_SUBMISSION.md).

## The basic idea

We answer two questions separately:

1. **Which team appears to control the ball?** Look at nearby players, their
   movement, earlier and later observations, and generated phase tags.
2. **Is play live or dead?** Use a separate model that looks at ball movement,
   player movement and the available tracking.

When the second answer is dead, we output **D**. Otherwise, we output the team
selected by the first part, **H** or **A**.

## What goes into it

The system reads three automatic inputs from `data/tactical/`:

- **Raw tracking:** player and ball positions at each frame.
- **Tracking metadata:** team identities and the player slots in the tracking file.
- **Generated Smart Tagging:** attacking and defending phases that add evidence
  about the team in possession.

We convert the normalised coordinates into metres on a 105.3 × 68 metre pitch.
Speed and direction come from changes between consecutive observed positions;
we do not use the speed column supplied in the raw tracking.

Video, broadcast tracking, homography, corrected tracking and possession labels
are not inputs to the submitted algorithm. Raw tracking frame 1 becomes output
frame 0. Every recording frame is kept, including breaks and footage outside play.

## How we estimate team possession

**Start with proximity.** A player near the ball provides control evidence for
their team. Moving together strengthens that evidence. A sudden, unusually large
jump in the ball coordinates reduces trust in the observation.

**Look around the current frame.** Because this is an offline task, we can use
observations before and after it. If the home team controls the ball before a
short gap and again afterwards, that supports keeping home possession through
the gap. Earlier evidence gradually fades as the gap grows.

**Add phase information.** Generated attacking tags support the tagged team;
defending tags support its opponent. Their influence is limited so repeated or
overlapping tags do not keep increasing confidence.

**Choose a consistent sequence.** Viterbi decoding balances each frame's evidence
against a cost for changing teams. This reduces rapid changes caused by noisy
observations. Team decoding continues through predicted stoppages; this can help
continuity, but can also carry the previous team into a restart for too long.

## How we detect dead play

A small logistic model uses movement and tracking availability to estimate whether
play has stopped. Its inputs include:

- Ball speed, distance outside the pitch and distance to the nearest player.
- Average player movement and the proportion of players moving slowly.
- How much player and ball tracking is available, and whether ball motion looks reliable.

We summarise these measurements over 0.4, 2 and 6 second windows and include flags
for unavailable measurements. This produces 48 features. Another Viterbi decoder
reduces unsupported switches between live and dead.

Missing ball coordinates do **not** automatically mean dead play. Set-piece tags
do not directly assign D, and Smart Tagging is not an input to this learned model.

## Passes, gaps and uncertain situations

Possession can continue while a pass travels between players. We do not require
a nearby player in every frame, and we do not explicitly classify each pass.
During challenges or loose-ball situations, the sequence still chooses a team
for every frame predicted to be live; there is no separate neutral live-ball class.

Missing positions stay missing. We use surrounding evidence rather than filling
in a ball trajectory. Pitch x/y does not give ball height. The output contains
no X labels, and no manual half-time or final-whistle boundaries are supplied.

## Run the submitted version

From the **repository root**, with Python 3.11 and the authorised tactical assets
in place:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-submission.txt
.venv/bin/python -m possession.pipeline \
  --config possession/config_external_control_v1.json \
  --stoppage-model possession/model_external_stoppage_v1.json \
  --out evaluation/results/reproduced_v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/reproduced_v1.csv
```

**Use both the configuration and the stoppage-model arguments above.** Running
`pipeline.py` with its defaults runs the original prototype instead.

The model weights are already included. You do not need to download the external
training matches to generate the submission. For assets in another location,
add `--data-dir /path/to/data`, where that folder contains `tactical/`.
You can also set `MTAH_DATA_DIR` when no `--data-dir` is supplied.

The pipeline writes to the requested output paths, so choose a new output name
when keeping the results of multiple runs.

## What the run produces

| File | Contents |
| --- | --- |
| `evaluation/results/reproduced_v1.csv` | One H/A/D label per frame: 199,319 rows, indexed 0–199,318. |
| `data/processed/reproduced_v1/evidence.npz` | Per-frame movement, control evidence, separate team/dead decisions and final labels. |
| `data/processed/reproduced_v1/run.json` | Parameters, input/output hashes, label counts and model information. |

The evidence folder follows the CSV's filename. In the saved arrays, final state
codes are **0=H, 1=A, 2=D**. Internal scores describe model support; they are not
calibrated probabilities that the labels are correct.

The packaged submission is [GunnersFC.csv](../submission/GunnersFC.csv). We checked
that a separate minimal environment reproduces it byte for byte. Instructions
for that check are in [SETUP.md](../SETUP.md). The official validator checks file
format only; passing it does not establish possession accuracy.

## Calibration, results and limitations

We selected team-control parameters using external **SkillCorner** matches and
trained the live/dead model on external **IDSSE** matches. Challenge ground truth,
manual challenge labels, provided-video training/tuning and prediction/video
comparisons were not used. Some settings, including tag weights, remain fixed
engineering assumptions.

The main weaknesses are long tracking gaps, contested possession, airborne ball
projections and team selection immediately after restarts. External results may
not transfer to the challenge's tracking quality or possession definitions.
A switching cost can also remove real short possessions.

Later restart and spell experiments are kept in this folder for transparency.
They did not pass every replacement check, so the submitted version uses the
control configuration and stoppage model shown in the command above. External
validation is evidence about performance; it is not a challenge score.

## Where to find the details

| File | What it explains |
| --- | --- |
| `pipeline.py` | Reads tactical inputs, builds team evidence and writes predictions. |
| `stoppage.py` | Builds movement features and predicts live/dead play. |
| [CALIBRATION.md](CALIBRATION.md) | SkillCorner team-control calibration and results. |
| [STOPPAGE_CALIBRATION.md](STOPPAGE_CALIBRATION.md) | IDSSE live/dead training, data attribution and results. |
| [JOINT_VALIDATION.md](JOINT_VALIDATION.md) | Combined H/A/D and restart evaluation. |
| [RESTART_IMPROVEMENT.md](RESTART_IMPROVEMENT.md) | Later restart experiment and its tradeoffs. |
| [SPELL_CALIBRATION.md](SPELL_CALIBRATION.md) | Later spellF1 experiment and why it was not selected. |

To run the code tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests
```

These tests check implementation behaviour. Football performance is evaluated
separately on external reference data, as described in the linked reports.
