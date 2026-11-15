# GunnersFC — Football Possession

**Team:** GunnersFC

**Submission:** `GunnersFC.csv`

Our system assigns a possession label to every frame of the match:

| Label | Meaning |
| --- | --- |
| H | Home team possession |
| A | Away team possession |
| D | Dead play: the ball is not in play |

It combines player and ball movement, nearby observations and the supplied
Smart Tagging to decide which team has possession and when play has stopped.
All labels are generated automatically. Parameters that we fitted or tuned were
chosen using external matches only.

## What data we use

The final system uses three files from the **tactical recording**:

- **Automatic tracking:** player and ball positions for each frame.
- **Tracking metadata:** identifies the teams and the player slots in the tracking file.
- **Generated Smart Tagging:** attacking and defending phases that provide extra
  evidence about which team has possession.

The tracking coordinates describe positions on the pitch. We convert them to
metres using the documented pitch size of **105.3 × 68 metres**. We calculate
speed and direction from changes in position between consecutive observed frames.
The speed values already present in the tracking file are not used.

We explored the broadcast recording and homography data while understanding the
assets. The final system uses tactical pitch coordinates directly and does not
need either video, broadcast tracking or homography. Corrected tracking,
possession references and manually identified match boundaries are not inputs.

## How the system makes its decisions

We chose a combination of control rules and a small statistical model so that
we could inspect the evidence behind each decision and reproduce the results.

### 1. Estimate which team controls the ball

A player close to the ball provides evidence for their team. That evidence becomes
stronger when the player and ball move together. An unusually large jump in the
ball's reported position makes that observation less trustworthy.

We also look at nearby frames before and after the current one. For example, if
the home team has clear control before a short gap and again afterwards, the
system can carry home possession through the gap. Earlier control gradually
loses influence as the gap grows. This helps represent passes without requiring
the ball to stay beside a player in every frame.

Generated attacking and defending tags add a limited amount of team evidence.
They support the tracking observations rather than determining possession alone.

### 2. Estimate whether play is live or dead

A separate logistic model, trained on external matches, estimates whether play
has stopped. It considers ball movement, distance outside the pitch, distance to
the nearest player, player movement and how much tracking is available. It also
measures how many players are moving slowly and whether the ball observations
look reliable.

These measurements are summarised over **0.4, 2 and 6 second windows**, with flags
for unavailable measurements, giving 48 input features. Missing ball positions
are evidence of missing information; they do not automatically mean dead play.
Smart Tagging is not an input to this model.

### 3. Choose a consistent sequence of labels

We use Viterbi decoding: a sequence method that balances the evidence at each
frame against a cost for changing state. This reduces rapid, unsupported changes
between teams or between live and dead play.

Team possession and live/dead play are decoded separately. When the live/dead
model predicts dead play, the final label is **D**. Otherwise, the final label is
its predicted team, **H** or **A**.

## How we handle difficult situations

- **Passes and loose balls:** possession can continue between touches, using
  earlier and later control evidence. Every predicted live frame receives a team;
  the system has no separate neutral live-ball class.
- **Challenges:** proximity, relative movement and surrounding frames help choose
  a team. Being closest to the ball does not by itself prove a touch.
- **Missing tracking:** unavailable positions stay missing. We use the remaining
  player observations and surrounding context rather than inventing ball positions.
- **Restarts:** play resumes when the live/dead model changes back to live. Team
  evidence continues through stoppages, which can sometimes carry the previous
  team's possession into the restart.
- **High balls:** pitch coordinates do not give ball height. We do not interpret
  a large y-coordinate as an airborne ball.

Set-piece tags do not directly create dead intervals. We supply no manual
half-time or final-whistle boundaries. The output contains no **X** labels; the
official scorer excludes frames whose ground-truth label is X.

## External data used for calibration

We used provider-supplied labels from external matches. We created no new manual
labels. Challenge ground truth, provided-video training or tuning, public
challenge scores and comparisons between predictions and the provided video
were not used in our development. We follow the organisers' later rules clarification.

### SkillCorner Open Data

We used **three matches** from [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
to select the control radius, time windows, possession continuity and switching
cost. Only explicitly detected player and ball positions enter the features.
The provider's team-possession labels supply the reference where available;
unknown possession is excluded from scoring and is never treated as dead play.

| Purpose | Match IDs |
| --- | --- |
| Choose parameters from 17 predefined configurations | 2017461, 2016236 |
| Test the selected parameters on an unused match | 2015213 |

The source is pinned to revision
`4340d274572876239c154c90bc507a9b3250a656`. The upstream MIT licence and provenance
are retained locally. Credit: **SkillCorner and PySport**.

### IDSSE

We downloaded **five matches** from the IDSSE dataset. Its explicit ball-status
labels provide live/dead references; its team-possession fields provide H/A
references. Feature inputs contain observed x/y only. Provider speed, altitude,
status, possession and events are kept out of those inputs.

| Purpose | Match IDs |
| --- | --- |
| Fit the initial live/dead model | J03WPY |
| Choose settings, then refit using both development matches | J03WOH and J03WPY |
| Test live/dead and combined H/A/D decisions | J03WQQ |
| Test later restart and possession-sequence experiments | J03WR9, J03WMX |

The later experiments used the same two development matches and fresh validation
matches. Their validation results did not change fitted parameters. Those
experimental models are not used in this submission.

**Attribution:** Bassek, Rein, Weber & Memmert (2025), DFL / Sportec Solutions,
*An integrated dataset of spatiotemporal and event data in elite soccer*.
[Dataset, version 1](https://doi.org/10.6084/m9.figshare.28196177.v1) ·
[Publication](https://doi.org/10.1038/s41597-025-04505-y) ·
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

## How we checked the system

We tested on external matches excluded from fitting and parameter selection.
We also created damaged copies of their tracking: about 43% of ball observations
were hidden in blocks, 15% of player observations were dropped, and 1% of the
remaining ball positions were perturbed. These changes were generated without
looking at reference labels.

The table reports **macro F1**, which gives each evaluated class equal weight.
A score of 1 is perfect. The SkillCorner result covers only frames with known
H/A references: **43.5%** of that test match. IDSSE J03WQQ has **142,345** explicit
live/dead reference frames.

| External test | Original tracking | Damaged tracking |
| --- | ---: | ---: |
| SkillCorner: H/A, match 2015213 | 0.7942 | Not tested |
| IDSSE: live/dead, match J03WQQ | 0.9853 | 0.9523 |
| IDSSE: H/A/D, match J03WQQ | 0.9077 | 0.8507 |

The competition ranks **spellF1**, which measures how well complete possession
periods match, rather than just counting correct frames. On another unused
external match, J03WMX, this submission's spellF1 was **0.50188** with original
tracking and **0.39523** with damaged tracking.

A later experimental version improved these scores to 0.53748 and 0.43963, but
narrowly failed a restart check fixed before testing. We kept the earlier version
for this submission. The full comparison is in `possession/SPELL_CALIBRATION.md`.

These are external results, not scores on the challenge match. Original and
damaged tracking from the same match share the same reference and do not count
as two independent validation matches.

## Limitations and next improvements

The largest difficulties are long gaps in ball tracking, contested possession,
airborne balls and choosing the correct team immediately after a restart. The
team decoder can carry earlier possession across a stoppage for too long.

The external providers may define possession differently and have different
tracking noise. Our tests use few matches, and the external datasets have no
equivalent to the challenge's Smart Tagging. Its weights remain fixed assumptions.
Internal model scores should not be read as probabilities that a label is correct.

With more time, we would focus on missing-ball restarts and test improvements on
additional untouched external matches, using spellF1 as the main measure. We do
not claim perfect labels or a measured challenge score.

## How to reproduce the submission

Use **Python 3.11** and run these commands from the repository root. Put the
authorised match files under `data/tactical/`, following `data/README.md`.
The saved model is included, so external training downloads are unnecessary to
regenerate the submission. Keep match data out of the public repository.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-submission.txt
.venv/bin/python -m possession.pipeline \
  --config possession/config_external_control_v1.json \
  --stoppage-model possession/model_external_stoppage_v1.json \
  --out evaluation/results/reproduced_v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/reproduced_v1.csv
```

The output has **199,319 rows**, covering frames 0–199,318:
H=74,426, A=51,435, D=73,458 and X=0. The validator checks file structure; it does
not check whether the possession labels are correct.

We also regenerated the CSV from an isolated copy of the required source, using
a separate environment with Python 3.11.16, NumPy 2.4.6 and pandas 3.1.1. Its asset
directory contained only the three automatic input files. The regenerated CSV
matched the submission **byte for byte** and passed the official format check.

For the same check, run:

```bash
.venv/bin/python scripts/verify_reproduction.py \
  --python /path/to/clean/environment/bin/python
```

Install `requirements-submission.txt` in that environment first. The check saves
its report under `data/processed/reproduction_v1.json` and refuses to overwrite
an existing report.

Expected CSV SHA-256:

```text
791099fb5195ed0fc8a2c9a60de7fcc5a4cfae9d5899a196819b1c56b4518c3f
```
