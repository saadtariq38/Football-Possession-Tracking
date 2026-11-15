# Technical Specification (`Tracking/technical_details.md`)

This document explains the engineering design, thresholds, algorithms, and concrete examples behind our Football Possession Detection pipeline on the **FC Barcelona vs. Athletic Club (`BAR-ATH`)** match (`199,319` frames at `25 fps`, pitch size `105.3m × 68.0m`).

---

## 1. System Inputs, Outputs & Evaluation Metrics

### Input Data & Output Labels
For every frame `0 .. 199,318`, the system outputs one of four labels:
- **`H` (Home — FC Barcelona):** Barcelona is in controlled possession during live play.
- **`A` (Away — Athletic Club):** Athletic Club is in controlled possession during live play.
- **`D` (Dead Ball):** Play is stopped inside a regulation half (throw-in, corner, goal kick, foul, kickoff setup).
- **`X` (Excluded):** Outside regulation halves (`half == 0`: pre-match, half-time, post-match).

Regulation halves are strictly bounded by the official match clock:
- **1st Half (`half = 1`):** frames `11,108` to `82,972` (`71,865` frames / `47m 54.6s`)
- **2nd Half (`half = 2`):** frames `108,297` to `195,289` (`86,993` frames / `57m 59.7s`)

### How the Competition Scores Predictions
1. **`spellF1` (Primary Ranking Score):**
   - Groups consecutive `H` or `A` frames into whole "possession spells" and matches each predicted spell to the closest ground-truth spell of the same team based on overlap (`IoU` across `0.3, 0.4, 0.5, 0.6, 0.7`).
   - **Why this drives our design:** A single false `0.4s` opponent flip in the middle of a `40s` possession cuts **1 real spell** into **2 half-length spells** plus **1 fake opponent spell**—hurting `spellF1` three times at once.
2. **`macroF1` (Tie-Breaker Score):**
   - Average per-frame F1 accuracy across `H`, `A`, and `D` equally.

---

## 2. Shared Feature & Cleaning Layer ([`possession/features.py`](../possession/features.py))

Before any possession logic runs, `features.load(GAME)` builds (and caches to `.npz`) a clean per-frame table (`199,319` rows) in real meters and `m/s`:
- **Ball Glitch Filtering (`clean_ball`):**
  - Removes readings more than `8.0m` outside the pitch.
  - Removes 1-frame teleport spikes (`> 1.5m` jump away and immediately back).
  - Removes physically impossible speeds (`> 40.0 m/s`).
  - Fills tiny missing gaps (`<= 5 frames` / `0.2s`) with a straight line.
- **Velocity & Speed (`centred_rate`):**
  - Computed over a `±2 frame` (`0.16s`) window around each frame for the ball and all 22 players.
- **Player-to-Ball Kinematics (`nearest_to_ball`):**
  - `near_home_dist`, `near_away_dist`: Distance (in meters) from the ball to the closest Home and Away player.
  - `near_home_rel_speed`, `near_away_rel_speed`: Difference between the ball's velocity vector and that nearest player's velocity vector.
    - *Why it matters:* If a pass flies past a jogging defender at `14 m/s`, the distance briefly drops to `0.9m`, **but** the relative speed is `~13 m/s`—proving the defender is not controlling the ball.
- **Pitch Tempo (`all_mean_speed`):**
  - Average speed (`m/s`) of all visible players on the pitch. During live play this is typically `1.8–3.8 m/s`; during stoppages it drops below `1.35 m/s`.

---

## 3. Stage 1: ATD Physics & Enhanced Possession Engine ([`stage1_atd_tracking.py`](stage1_atd_tracking.py))

Stage 1 contains two parts: **Stage 1A** (ATD geometry & probabilities for Stage 3 Viterbi) and **Stage 1B** (our #1 Enhanced Hybrid Possession Engine called by [`main.py`](main.py)).

### 3.1 Short-Gap Linear Ball Interpolation (`interpolate_short_ball_gaps`, Stage 1A)
Raw `ball_x` is missing (`NaN`) in **`43.1%` (`85,906` / `199,319`)** of frames because optical trackers lose high aerial balls or balls hidden behind crowds.
- **Rule:** Whenever the ball disappears for **`<= 75 frames` (`3.0s`)** AND both the last seen position before the gap and first seen position after the gap are inside the pitch (`[0, 105.3] × [0, 68.0]`), Stage 1A fills in the missing path with a straight line.

> **Example (Frames `12,410` to `12,448`):**
> Barcelona hits a lofted switch from `(42.1m, 18.4m)` to `(61.8m, 52.0m)`. The camera loses the ball for `38 frames` (`1.52s`) in mid-air. Because `38 <= 75` and both ends are in-bounds, linear interpolation reconstructs the flight path instead of falsely calling a Dead Ball (`D`).

---

### 3.2 Enhanced Stoppage Detection (`detect_enhanced_stoppages`, Stage 1B)

Dead-ball (`D`) stoppages are triggered when any of these **6 physical stoppage conditions** holds for its minimum duration:

| Stoppage Signal | Plain-English Trigger Condition | Minimum Duration |
|---|---|---|
| 1. `out_of_play` | Ball is outside the `105.3m × 68.0m` pitch | `0.5s` (`12 frames`) |
| 2. `ball_lost` | Ball is missing (`NaN`) | `5.0s` (`125 frames`) |
| 3. `everyone_slow` | 2-sec avg player speed `< 1.35 m/s` AND nearest player `> 2.0m` away | `3.0s` (`75 frames`) |
| 4. `placed` | Ball speed `< 0.8 m/s` at corner arc (`< 3m`), 6-yard box, or center spot (`< 2m`) | `1.5s` (`37 frames`) |
| 5. `kickoff` | Teams lined up in own halves (`<= 1` player across halfway) and `all_mean_speed < 1.8 m/s` | `2.0s` (`50 frames`) |
| 6. **`walking_retrieval` (New)** | **2-sec avg player speed `< 1.12 m/s` AND ball within `8.5m` of the touchline** | **`2.5s` (`62 frames`)** |

#### Sideline Ball-Retrieval Toss Filter (`is_sideline_toss`)
Normally, `ball_speed > 4.0 m/s` signals that play has restarted. However, during a throw-in stoppage, players often toss or roll the dead ball along the touchline at `5–10 m/s` to the thrower while everyone else is still walking.
- **Rule:** Ignore ball movement up to `11.0 m/s` as a "restart" whenever `ball_touchline_m < 8.5m` AND the 1-second smoothed `all_mean_speed < 1.00 m/s`.

> **Example (Frames `19,357` to `19,990` in Public Ground Truth):**
> Ground truth has a `25.3s` Dead Ball (`D`) stoppage for a throw-in. At frame `19,705`, the ball is tossed along the sideline at `6.8 m/s` while all 22 players are walking at `0.89 m/s`. Without our rule, the detector prematurely ends the stoppage at `19,708` (`11.3s` too early). With `walking_retrieval` + `is_sideline_toss`, the stoppage holds until the actual throw at frame `19,960` (matching GT `19,990` within `1.2s`).

---

### 3.3 One-Touch Kick & Impulse Contact Promotion (`enhanced_control_events`, Stage 1B)

Baseline control requires a player to stay within `1.5m` of the ball with `rel_speed <= 3.0 m/s` for at least `10 frames` (`0.40s`). That misses two common football actions, which we fix:

1. **Impulse Kicks (`home_impulse`, `away_impulse`):**
   - When a player strikes a slow ball hard, the centered finite-difference `rel_speed` spikes above `3.0 m/s` on the exact kick frame.
   - **Fix:** Recover contact frames where a player is within `0.85m` of the ball, at least `0.35m` closer than the nearest opponent, and `ball_speed <= 5.5 m/s`.
2. **One-Touch Clearances & Passes (`5 to 9 frames` / `0.20s–0.36s`):**
   - A one-touch clearance only stays within `1.5m` of the kicker's foot for `5–9 frames` before flying away.
   - **Fix:** Promote a `5–9 frame` touch to full control (`is_control = True`) if:
     - The ball accelerates sharply right after contact (`post_speed - pre_speed >= 3.5 m/s` OR `post_speed >= 8.5 m/s`), **AND**
     - That player's team stays closer to the ball on at least **`60%` of the next `30 frames` (`1.2s`)**.

> **Example (Frames `16,803` to `16,833` in Public Ground Truth):**
> Ground truth has a `1.2s` (`30-frame`) Barcelona (`H`) spell during an interception/clearance between two Athletic (`A`) spells. Because the touch only lasted `< 10 frames`, the baseline ignored it completely (`0` overlap). One-Touch Promotion recovers it at `[16780, 16832]` (`IoU = 0.566`).

---

### 3.4 Sandwiched Pass-By & Deflection Filter (`clean_segment_changes`, Stage 1B)

During a long buildup, a ground pass sometimes rolls within `1.5m` of a jogging defender for `10–15 frames` (`0.40–0.60s`), creating a fake 1-touch opponent turnover before the original team picks up the pass.

- **Rule:** Remove a brief opponent turnover and immediately merge the surrounding possession if **all 4 checks** pass:
  1. **Sandwiched:** Possession immediately returns to the same team right after.
  2. **Single Event:** The opponent only got `<= 1` true control event during that spell.
  3. **Brief Contact:** Total controlled contact frames `<= 15 frames` (`0.60s`).
  4. **Short Spell:** Total spell duration `<= 150 frames` (`6.0s`).

> **Example (Frames `14,672` to `15,747` in Public Ground Truth):**
> Ground truth has a single `43.0s` (`1,075-frame`) Barcelona (`H`) possession. At frame `14,897`, a pass brushes past an Athletic player for `12 frames` (`[14897, 14909]`) before returning to Barcelona at `14,940`. Filtering out this `43-frame` sandwiched blip restores the full `[14672, 15747]` possession (`IoU = 1.000`).

---

## 4. Stage 2: Smart Tagging Tactical Priors ([`stage2_smart_tagging.py`](stage2_smart_tagging.py))

Parses all `638` tactical events from `BAR-ATH_tactical_smart_tagging.xml`:
1. **Defensive Phase Inversion (`possessing_team_of_event`):**
   - Attacking tags (`BUILD UP`, `PROGRESSION`, `BALL IN THE BOX`, etc.) credit the tagged team.
   - Defensive tags (`DEFENDING IN ... THIRD`, `DEFENSIVE TRANSITION`) **flip** to credit the opposing team (e.g., *Barcelona defending* $\implies$ *Athletic has the ball*).
2. **Set-Piece Dead/Live Split (`find_set_piece_restart_frame`):**
   - Scans inside each of the `59` `SET PIECES` windows for the first in-bounds frame where `ball_speed >= 3.5 m/s`. Everything before that kick frame gets a strong Dead Ball prior (`tag_p_D = 0.92`); everything after gets a strong Live Restart prior (`0.85`).
3. **26 `Team = N/A` Throw-In Resolver:**
   - All `26` Throw-ins in the XML have `Team = N/A`. Stage 2 assigns the throw-in to whichever team has the smaller median distance to the ball over the `25 frames` (`1.0s`) starting at the restart frame.

---

## 5. Stage 3: Viterbi Decoder ([`stage3_viterbi_and_eval.py`](stage3_viterbi_and_eval.py)) & Optional Visual Tools

1. **3-State Viterbi Decoder (`viterbi_decode_half` in `stage3_viterbi_and_eval.py`):**
   - Fuses Stage 1A + Stage 2 log-probabilities and decodes each half globally with `self_stay_prob = 0.996` and `min_spell_frames = 45` (`1.8s`).
---

## 6. Ablation Results (Public 5-Minute Ground Truth: `7,500` Frames, `17` Possessions)

| Configuration | `spellF1` (Primary) | `macroF1` (Tie-Breaker) | `H` F1 | `A` F1 | `D` F1 | `transF1@25` | `shareE` | Predicted Possessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_raw` (nearest player) | 0.0378 | 0.5972 | 0.659 | 0.453 | 0.679 | 0.1410 | 0.046 | 131 |
| `baseline_smooth125` (5s mode) | 0.4242 | 0.5812 | 0.670 | 0.449 | 0.625 | 0.1667 | 0.052 | 16 |
| `ours` (`possession/` baseline) | 0.4000 | 0.8155 | 0.869 | 0.779 | 0.799 | 0.2632 | 0.041 | 15 |
| `pipeline_stage3` (Stage 1A+2+Viterbi) | 0.4556 | 0.6803 | 0.742 | 0.621 | 0.678 | 0.3256 | 0.046 | 19 |
| **+ Sideline Walking Retrieval** | 0.4250 | 0.8420 | 0.875 | 0.798 | 0.853 | 0.3158 | 0.035 | 15 |
| **+ One-Touch Kick Promotion** | 0.4848 | 0.8482 | 0.881 | 0.806 | 0.857 | 0.3590 | 0.024 | 16 |
| **+ Sandwiched Pass-By Filter (`tracking_final`)** | **0.5419** | **0.8636** | **0.897** | **0.835** | **0.857** | **0.3784** | **0.017** | **14** |
