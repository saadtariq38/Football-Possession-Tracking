# GunnersFC — Technical document

More than a Hack 2027, Phase 1: ball possession.

## 1. Summary

Our program reads the tracking data and gives every frame one label: `H` (Barcelona has the ball),
`A` (Athletic Club has the ball) or `D` (ball out of play). It uses fixed rules, not a trained
model. It has three steps:

1. **Measure.** For every frame, measure distances, speeds and team shapes.
2. **Find stoppages.** Mark the frames where the ball is out of play.
3. **Decide possession.** Between stoppages, follow which team controls the ball.

## 2. Why this approach

- The ranking score counts whole possessions. One wrong switch costs a lot. So our rules change possession only when the other team clearly controls the ball and is biased towards this.
- The challenge rules say a loose or contested ball stays with the last team. We follow this..
- No labels from this match could be used for training. Rules based on football knowledge were used for this reason.

## 3. Data used

| Data | Used? | How |
|---|---|---|
| Tactical tracking (positions and speeds) | Yes | Main input |
| Smart Tagging | Yes | Small hint for which team has the ball when the ball is not tracked |
| Graded half frames (published by the organisers) | Yes | To know which frames to label |
| Homography | No | Not needed for a method that uses tracking only |
| Videos (tactical and broadcast) | No | No time to build ball detection from video |
| Broadcast tracking | No | Not linked to the tactical timeline |

What we found in the tracking:

- The ball is missing on 41% of graded frames, often for 10 seconds or more. Many of these gaps
  are stoppages.
- There is no ball height, no "who has the ball" field and no "ball in play" field.
- Player tracks do not follow one person for long. We only use which team each track belongs to.

## 4. Training data

None. We trained no model and used no labelled data.

We used the public 5-minute answer key only to check the program for errors. We did not change any
setting after looking at it. All settings come from football knowledge or physics, for example
"walking pace is about 1.3 m/s".

## 5. Step 1: measurements

For every frame we compute:

- **The ball position, cleaned.** We remove positions far outside the pitch, one-frame jumps, and
  speeds above 40 m/s (faster than any real shot). Gaps of up to 0.2 s are filled with a straight
  line.
- **For each team:** how far its nearest player is from the ball, and the difference between that
  player's velocity and the ball's. Near zero means the player moves with the ball.
- **Team shape:** each team's width and average running speed.
- **Lines:** players standing still at each touchline, goal line or corner.

## 6. Step 2: stoppages (label `D`)

A frame is the middle of a stoppage if any of these is true:

| Sign | Rule |
|---|---|
| Players stop | Average player speed over 3 s is below walking pace (1.3 m/s), and the ball does not move faster than 4 m/s in the 2 s before or after |
| Ball out | The ball is seen more than 0.5 m over a line |
| Ball placed | The ball lies still for 1.5 s in a goal area or corner arc (goal kick, corner) |
| Throw-in | A player stands still at a touchline for 1.5 s, the ball was last seen near that touchline, and the ball is not being played |
| Kick-off | At least 16 players are tracked and at most 3 are past the halfway line, for 2 s |

Each stoppage is then extended:

- **Start.** Players slow down only after the ball goes out or the whistle blows. If the ball was
  last seen within 2 m of a line in the 6 s before, the stoppage starts right after that.
  Otherwise it starts just after the ball was last played, at most 3 s earlier.
- **End.** The first time the ball moves faster than 4 m/s again. For a kick-off, the moment the
  players leave the kick-off formation.
- **Start of each half.** Dead from the first graded frame until the first kick-off. This covers
  the 2nd half, which is graded about 2 minutes before its kick-off.
- **Clean-up.** Stoppages less than 4 s apart are joined. Stoppages shorter than 3 s are dropped.

## 7. Step 3: possession (labels `H` and `A`)

**Control.** A team controls the ball in a frame when one of its players is within 1.5 m of the
ball, moves with it (velocity difference at most 3 m/s), and no opponent is within 1.5 m. An
unbroken run of such frames is:

- a **control** if it lasts 0.4 s or more;
- a **touch** if it lasts 0.12 s to 0.4 s.

**Rules**, applied to each stretch of live play:

| Situation | What we do |
|---|---|
| Restart | The first team to touch or control the ball has it from the restart |
| The other team gets a control | Possession changes, from the start of that control |
| The other team gets one touch | Nothing changes (deflection, block) |
| The other team gets two touches in a row within 3 s, with no touch by the team in possession between them | Possession changes, from the first touch |
| Players of both teams within 1.5 m of the ball (duel, tackle) | Nothing is decided; we wait to see who controls the ball next |
| Ball loose or in the air | The last team keeps it |

**Missing ball.**

- If the same team has the ball before and after a gap, it keeps the whole gap.
- If the team changed and the ball was missing for 2 s or more before the change, we move the
  change to the point in the gap where the team shapes say it happened.
- If a stretch of live play has no control at all, the team shapes decide.

**Team shapes.** Where the ball is visible, we checked how often each shape signal agrees with our
own ball-based labels (no human labels involved):

| Signal | Agrees |
|---|---|
| The team with the ball is wider | 85% |
| The team without the ball runs more | 72% |
| The team named by Smart Tagging | 81% |
| The direction the teams move (not used) | 59% |

We add the first three, each scaled by its spread over the match. The result is used only where
the ball is missing.

## 8. What we can measure

We cannot see the full-match score. These are the checks we can do:

| Check | Result |
|---|---|
| Possessions in the full match | 224, average 15.8 s |
| Barcelona share of possession | 64% |
| Time with the ball out of play | 44.5% (47 of 106 minutes) |
| Smart Tagging set pieces with a detected stoppage | 48 of 52 |
| Public 5 minutes, macroF1 | 0.816 (official baselines: 0.58 to 0.62) |
| Public 5 minutes, spellF1 | 0.400 (only 17 possessions, so very uncertain) |

## 9. Limitations

- **Stoppages start late.** After a whistle, players take a few seconds to slow down.
- **Short possessions** (under about 2 s) are often missed.
- **No ball height**, so a high ball can look like it is at a player's feet.
- **When the ball is missing**, the team is decided by team shape, which agrees with the ball-based
  labels about 85% of the time.
- **Settings are untested.** They come from football knowledge and were not tested on labelled
  matches.
- **Reproduction.** Our runs give the identical file, also from a fresh copy with a fresh install.
  A very different processor could, in rare cases, round a number differently.

With more time we would:

- tune the settings on labelled matches from other games (for example the IDSSE Bundesliga open
  data, which the organisers approved);
- find stoppage starts from the referee's whistle in the video's sound;
- find the ball in the video when the tracking loses it.

## 10. How to reproduce

Full steps are in [README.md](README.md). In short:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-possession.txt
.venv/bin/python -m possession.run --out GunnersFC.csv
```

The file's SHA-256 must be
`6999f9bc50a4ccebefd98fe84b453858f204c0047549c6aac1e9c92d20833dbe`.

| File | What it does |
|---|---|
| `possession/atd.py` | Reads the tracking file |
| `possession/tagging.py` | Reads Smart Tagging |
| `possession/features.py` | Step 1, measurements |
| `possession/deadball.py` | Step 2, stoppages |
| `possession/tracker.py` | Step 3, possession |
| `possession/run.py` | Runs all steps and writes the CSV |
