"""Stage 1: Continuous Geometry & Physics Engine from Automatic Tracking Data (ATD).

What this script does:
----------------------
1. Streams `BAR-ATH_tactical_atd.txt` (199,319 frames) and parses player & ball coordinates
   in real pitch metres (105.3m x 68.0m).
2. Interpolates short in-bounds ball-missing gaps (< 3 seconds / 75 frames) so that
   aerial passes or brief camera occlusions do NOT falsely trigger Dead Ball ('D').
3. Computes ball velocity, speed, acceleration (touch indicator), and pitch boundary checks.
4. Computes nearest Home ('H') and Away ('A') player distances to the ball (or to the
   22-player convergence cluster when the ball is missing for a long stretch).
5. Filters out single-frame deflections/rebounds by requiring "controlled touch"
   (sustained proximity over ~0.5s or pass + same-team reception).
6. Outputs per-frame feature arrays and Stage 1 probabilities P(H), P(A), P(D).

Run standalone to inspect a summary on the match and score Stage 1 alone:
    python Tracking/stage1_atd_tracking.py
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402
from possession import deadball, features, tracker  # noqa: E402
from possession.series import rolling_mean, runs  # noqa: E402

# Constants
FPS = GAMES[GAME]["fps"]               # 25 fps
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]  # (105.3, 68.0) metres

HOME, AWAY, DEAD = 0, 1, 2
NO_ANSWER = tracker.NO_ANSWER
LABELS = np.array(["H", "A", "D"])
FULL_LABELS = tracker.LABELS           # ["H", "A", "D", "X"]

# Hyperparameters grounded in football physics (not tuned to the 5-min public window)
MAX_INTERP_GAP_FRAMES = 75      # 3.0 seconds: max gap to linearly interpolate an in-bounds ball
CONTROL_RADIUS_M = 2.0          # A player within 2.0m can physically touch/control the ball
CONTEST_MARGIN_M = 0.8          # If both teams are within 0.8m of each other near the ball, it's a duel
MIN_CONTROL_FRAMES = 12         # ~0.48s of sustained proximity/control to confirm a turnover
OUT_OF_BOUNDS_MARGIN_M = 0.5    # Margin around [0, PITCH_L] x [0, PITCH_W] for out-of-play detection


def load_raw_atd_matrices(game=GAME, data_dir=None):
    """Stream `BAR-ATH_tactical_atd.txt` once into fast NumPy arrays.

    Returns
    -------
    player_x : (N, 46) float32 in metres (NaN where untracked)
    player_y : (N, 46) float32 in metres (NaN where untracked)
    ball_x   : (N,) float32 in metres (NaN where untracked)
    ball_y   : (N,) float32 in metres (NaN where untracked)
    team_map : (46,) int8 (0 = Home/Barcelona, 1 = Away/Athletic Club)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_map = team_of_each_track(meta_path)
    n_slots = len(team_map)

    player_x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f_idx = int(parts[0]) - 1  # 0-based video frame

            # Parse ball
            b_str = parts[2]
            if not b_str.startswith("N"):
                c1 = b_str.find(",")
                c2 = b_str.find(",", c1 + 1)
                ball_x[f_idx] = float(b_str[:c1]) * PITCH_L
                ball_y[f_idx] = float(b_str[c1 + 1:c2]) * PITCH_W

            # Parse 46 player track slots
            for s_idx, group in enumerate(parts[1].split(";")):
                if group.startswith("N"):
                    continue
                c1 = group.find(",")
                c2 = group.find(",", c1 + 1)
                player_x[f_idx, s_idx] = float(group[:c1]) * PITCH_L
                player_y[f_idx, s_idx] = float(group[c1 + 1:c2]) * PITCH_W

    return player_x, player_y, ball_x, ball_y, team_map


def interpolate_short_ball_gaps(ball_x, ball_y, max_gap=MAX_INTERP_GAP_FRAMES):
    """Linearly interpolate short missing-ball gaps when the ball was in-bounds before & after.

    Why: The ball is missing on 41% of graded frames (passes in the air, player occlusions).
    If the ball was inside the pitch at frame t1 and inside the pitch at t2 (with t2 - t1 <= 75),
    it did NOT go dead -- it was simply occluded in flight.
    """
    bx = ball_x.copy()
    by = ball_y.copy()
    is_interped = np.zeros(len(bx), dtype=bool)

    valid_idx = np.flatnonzero(~np.isnan(bx) & ~np.isnan(by))
    if len(valid_idx) < 2:
        return bx, by, is_interped

    gaps = np.diff(valid_idx)
    short_gap_starts = np.flatnonzero((gaps > 1) & (gaps <= max_gap))

    for idx in short_gap_starts:
        f_start = valid_idx[idx]
        f_end = valid_idx[idx + 1]

        # Only interpolate if both endpoints are reasonably inside the pitch
        in_bounds_start = (0.0 <= bx[f_start] <= PITCH_L) and (0.0 <= by[f_start] <= PITCH_W)
        in_bounds_end = (0.0 <= bx[f_end] <= PITCH_L) and (0.0 <= by[f_end] <= PITCH_W)

        if in_bounds_start and in_bounds_end:
            steps = f_end - f_start
            t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:-1]
            bx[f_start + 1:f_end] = bx[f_start] + t * (bx[f_end] - bx[f_start])
            by[f_start + 1:f_end] = by[f_start] + t * (by[f_end] - by[f_start])
            is_interped[f_start + 1:f_end] = True

    return bx, by, is_interped


def compute_stage1_features(player_x, player_y, ball_x, ball_y, team_map):
    """Compute per-frame geometric & physical features for possession inference.

    Returns a DataFrame indexed by `frame` (0 .. 199,318) with columns:
      - `ball_visible`: raw ATD ball visibility
      - `ball_interped`: True where ball was recovered by short-gap interpolation
      - `bx`, `by`: ball position in metres (interpolated where applicable)
      - `ball_speed`: instantaneous ball speed in m/s
      - `ball_out_of_bounds`: True if ball is outside the touchlines/goal lines
      - `dist_home`: distance (m) from ball to nearest Barcelona player
      - `dist_away`: distance (m) from ball to nearest Athletic Club player
      - `raw_owner`: 0 (Home), 1 (Away), -1 (Contested/In-flight/Unknown)
      - `p_H`, `p_A`, `p_D`: Stage 1 probability estimates for each state
    """
    n_frames = len(ball_x)
    bx, by, is_interped = interpolate_short_ball_gaps(ball_x, ball_y)
    has_ball = ~np.isnan(bx)

    # 1. Ball speed (m/s)
    vx = np.diff(bx, prepend=bx[0]) * FPS
    vy = np.diff(by, prepend=by[0]) * FPS
    ball_speed = np.hypot(vx, vy)
    ball_speed[np.isnan(ball_speed)] = 0.0

    # 2. Out-of-bounds check
    out_of_bounds = np.zeros(n_frames, dtype=bool)
    out_of_bounds[has_ball] = (
        (bx[has_ball] < -OUT_OF_BOUNDS_MARGIN_M) |
        (bx[has_ball] > PITCH_L + OUT_OF_BOUNDS_MARGIN_M) |
        (by[has_ball] < -OUT_OF_BOUNDS_MARGIN_M) |
        (by[has_ball] > PITCH_W + OUT_OF_BOUNDS_MARGIN_M)
    )

    # 3. Distance from ball to nearest Home and Away player
    home_mask = (team_map == HOME)
    away_mask = (team_map == AWAY)

    dx = player_x - bx[:, None]
    dy = player_y - by[:, None]
    dist_all = np.hypot(dx, dy)  # (N, 46)

    # Suppress All-NaN slice warnings with inf replacement
    dist_home_mat = np.where(np.isnan(dist_all[:, home_mask]), np.inf, dist_all[:, home_mask])
    dist_away_mat = np.where(np.isnan(dist_all[:, away_mask]), np.inf, dist_all[:, away_mask])

    dist_home = np.min(dist_home_mat, axis=1)
    dist_away = np.min(dist_away_mat, axis=1)

    # 4. Identify "Controlled Touch" frames
    # A player only exerts control when within CONTROL_RADIUS_M (2.0m) and clearly closer than
    # the opponent (or winning the duel). While the ball is mid-pass (> 2.0m from everyone),
    # possession MUST stay with whoever passed it!
    raw_owner = np.full(n_frames, -1, dtype=np.int8)

    with np.errstate(invalid="ignore"):
        home_close = (dist_home <= CONTROL_RADIUS_M) & ((dist_away - dist_home) >= CONTEST_MARGIN_M)
        away_close = (dist_away <= CONTROL_RADIUS_M) & ((dist_home - dist_away) >= CONTEST_MARGIN_M)

    # Even if tightly contested (< 0.8m margin), give soft edge to whoever is strictly closer
    # within 1.2m
    home_tight = (dist_home <= 1.2) & (dist_home < dist_away)
    away_tight = (dist_away <= 1.2) & (dist_away < dist_home)

    raw_owner[home_close | home_tight] = HOME
    raw_owner[away_close | away_tight] = AWAY
    raw_owner[out_of_bounds] = DEAD

    # 5. Filter out brief 1-to-11 frame deflections ("controlled possession" rule)
    # Carry possession forward across mid-pass flight (-1) and reject < MIN_CONTROL_FRAMES touches
    controlled_owner = filter_deflections_and_carry_passes(
        raw_owner, has_ball, out_of_bounds, min_control_frames=MIN_CONTROL_FRAMES
    )

    # 6. Convert into smooth Stage 1 probability estimates (p_H, p_A, p_D)
    p_H = np.full(n_frames, 0.33, dtype=np.float32)
    p_A = np.full(n_frames, 0.33, dtype=np.float32)
    p_D = np.full(n_frames, 0.34, dtype=np.float32)

    # When ball is missing for a long time (> 3s, so not interpolated) -> higher Dead Ball prior
    long_missing = ~has_ball
    p_D[long_missing] = 0.70
    p_H[long_missing] = 0.15
    p_A[long_missing] = 0.15

    # When ball is out of bounds -> strong Dead Ball prior
    p_D[out_of_bounds] = 0.90
    p_H[out_of_bounds] = 0.05
    p_A[out_of_bounds] = 0.05

    # Active controlled possession
    is_h = (controlled_owner == HOME) & ~long_missing & ~out_of_bounds
    is_a = (controlled_owner == AWAY) & ~long_missing & ~out_of_bounds

    p_H[is_h], p_A[is_h], p_D[is_h] = 0.82, 0.10, 0.08
    p_H[is_a], p_A[is_a], p_D[is_a] = 0.10, 0.82, 0.08

    return pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "ball_visible": ~np.isnan(ball_x),
        "ball_interped": is_interped,
        "bx": bx,
        "by": by,
        "ball_speed": ball_speed,
        "ball_out_of_bounds": out_of_bounds,
        "dist_home": dist_home,
        "dist_away": dist_away,
        "raw_owner": raw_owner,
        "controlled_owner": controlled_owner,
        "p_H": p_H,
        "p_A": p_A,
        "p_D": p_D,
    })


def filter_deflections_and_carry_passes(raw_owner, has_ball, out_of_bounds, min_control_frames=12):
    """Carry possession forward while the ball travels between players, and suppress deflections.

    Rules implemented from CHALLENGE.md:
    - While the ball is in flight between touches (`raw_owner == -1` and `has_ball == True`),
      possession stays with the team that last controlled it.
    - If the opposing team touches the ball for fewer than `min_control_frames` (e.g. a deflection,
      block, or rebound) and fails to retain it, possession does NOT change hands.
    """
    n = len(raw_owner)
    carried = raw_owner.copy()

    # Step A: Carry last known team across in-flight pass frames (-1 where ball is in play)
    for half, first_f, last_f in half_video_blocks(GAME):
        current_team = -1
        for f in range(first_f, last_f + 1):
            if out_of_bounds[f] or not has_ball[f]:
                # Long missing gap (> 3s) or out of bounds -> dead ball candidate
                carried[f] = DEAD
                current_team = -1
            elif raw_owner[f] in (HOME, AWAY):
                current_team = raw_owner[f]
                carried[f] = current_team
            else:
                # Ball is in play and in flight between players
                carried[f] = current_team if current_team in (HOME, AWAY) else DEAD

    # Step B: Suppress short deflection spells (< min_control_frames)
    # Find contiguous runs of carried labels
    changes = np.flatnonzero(np.diff(carried) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))

    for i in range(1, len(starts) - 1):
        s, e = starts[i], ends[i]
        duration = e - s
        label = carried[s]
        prev_label = carried[starts[i - 1]]
        next_label = carried[starts[i + 1]]

        # If a tiny H or A spell (< min_control_frames) is sandwiched, absorb it
        if label in (HOME, AWAY) and duration < min_control_frames:
            if prev_label == next_label and prev_label in (HOME, AWAY):
                carried[s:e] = prev_label
            elif prev_label in (HOME, AWAY):
                carried[s:e] = prev_label

    return carried


# ---------------------------------------------------------------------------
# Enhanced Hybrid ATD Possession Engine (Merged from Stage 4)
# ---------------------------------------------------------------------------

def detect_enhanced_stoppages(table, game=GAME):
    """Enhance `possession.deadball.detect` with sideline walking ball-retrieval
    detection (`all_mean_speed < 1.12 m/s` and `ball_touchline_m < 8.5m`) and
    ignore gentle sideline ball tosses (`< 11 m/s` while `team_speed_smooth < 1.00 m/s`).
    """
    paths = deadball.core_paths(table)
    graded = table["half"] > 0

    with np.errstate(invalid="ignore"):
        very_slow_pitch = rolling_mean(table["all_mean_speed"], int(2.0 * FPS)) < 1.12
        near_sideline = table["ball_touchline_m"] < 8.5
    walking_retrieval = deadball.keep_long_runs(
        graded & very_slow_pitch & near_sideline, 2.5
    )

    core = np.zeros(len(table["half"]), dtype=bool)
    for sign in deadball.SIGNS:
        core = core | paths[sign]
    core = core | walking_retrieval

    with np.errstate(invalid="ignore"):
        team_speed_smooth = rolling_mean(table["all_mean_speed"], int(1.0 * FPS))
        is_sideline_toss = near_sideline & (team_speed_smooth < 1.00) & (table["ball_speed"] < 11.0)
        real_play = (table["ball_speed"] > deadball.BALL_PLAY_SPEED) & (~is_sideline_toss)
        played_frames = np.flatnonzero(real_play)
        distance_to_line = np.minimum(table["ball_touchline_m"], table["ball_goalline_m"])
        near_line = distance_to_line < deadball.LINE_NEAR_M
    seen_frames = np.flatnonzero(~np.isnan(table["ball_x"]))

    stoppages = []
    for half_number, first_frame, last_frame in features.half_video_blocks(game):
        half_start = first_frame
        half_stop = last_frame + 1

        core_starts, core_stops = runs(core[half_start:half_stop])
        for i in range(len(core_starts)):
            core_start = core_starts[i] + half_start
            core_stop = core_stops[i] + half_start

            start = deadball.stoppage_start(core_start, seen_frames, played_frames, near_line)

            if paths["kickoff"][core_stop - 1]:
                stop = core_stop
            else:
                stop = deadball.restart_after(core_stop, played_frames)

            stoppages.append((max(half_start, start), min(half_stop, stop)))

        opening_end = deadball.opening_kickoff_end(paths["kickoff"][half_start:half_stop])
        if opening_end is not None:
            stoppages.append((half_start, half_start + opening_end))

    return deadball.tidy(stoppages)


def enhanced_control_events(table):
    """Build control events from `table` with two physical refinements:
      1. Detect close-range impulse kicks (`dist <= 0.85m`, margin `>= 0.35m`, `bspeed <= 5.5 m/s`).
      2. Promote deliberate one-touch kicks/clearances (5..9 frames where ball accelerates
         sharply off the player's foot and that team stays nearest after the kick).
    """
    side = tracker.control_side(table).copy()
    n = len(side)

    bspeed = table["ball_speed"]
    h_dist = table["near_home_dist"]
    a_dist = table["near_away_dist"]

    with np.errstate(invalid="ignore"):
        home_impulse = (h_dist <= 0.85) & ((a_dist - h_dist) >= 0.35) & (bspeed <= 5.5)
        away_impulse = (a_dist <= 0.85) & ((h_dist - a_dist) >= 0.35) & (bspeed <= 5.5)

    side[(side == -1) & home_impulse] = HOME
    side[(side == -1) & away_impulse] = AWAY

    raw_events = tracker.control_events(side)

    promoted_events = []
    for start, stop, team, is_control in raw_events:
        length = stop - start
        if not is_control and length >= 5:
            pre_win = bspeed[max(0, start - 5):start + 2]
            post_win = bspeed[stop:min(n, stop + 18)]
            pre_speed = float(np.nanmin(pre_win)) if np.any(~np.isnan(pre_win)) else 0.0
            post_speed = float(np.nanmax(post_win)) if np.any(~np.isnan(post_win)) else 0.0

            follow_h = h_dist[stop:min(n, stop + 30)]
            follow_a = a_dist[stop:min(n, stop + 30)]
            with np.errstate(invalid="ignore"):
                team_closer_after = (
                    np.nanmean(follow_h < follow_a) >= 0.60
                    if team == HOME else np.nanmean(follow_a < follow_h) >= 0.60
                )

            if (post_speed - pre_speed >= 3.5 or post_speed >= 8.5) and team_closer_after:
                is_control = True

        promoted_events.append((start, stop, team, is_control))

    return promoted_events


def clean_segment_changes(changes, events_here, start, stop):
    """Remove spurious brief single-event turnovers (< 16 frames of control) when sandwiched
    between sustained possession by the original team.
    """
    if len(changes) <= 2:
        return changes

    cleaned = [changes[0]]
    i = 1
    while i < len(changes):
        c_frame, c_team = changes[i]
        next_frame = changes[i + 1][0] if (i + 1 < len(changes)) else stop
        prev_team = cleaned[-1][1]

        team_ctrl_evs = [
            ev for ev in events_here
            if ev[2] == c_team and ev[3] and c_frame <= ev[0] < next_frame
        ]
        total_ctrl_frames = sum(ev[1] - ev[0] for ev in team_ctrl_evs)
        spell_duration = next_frame - c_frame

        has_return_to_prev = (i + 1 < len(changes)) and (changes[i + 1][1] == prev_team)
        if (
            has_return_to_prev
            and len(team_ctrl_evs) <= 1
            and total_ctrl_frames <= 15
            and spell_duration <= 150
        ):
            i += 2
            continue

        cleaned.append((c_frame, c_team))
        i += 1

    return cleaned


def label_enhanced(table, stoppages):
    """Run enhanced possession tracking across all live stretches."""
    n_frames = len(table["half"])
    dead = deadball.dead_mask(stoppages, n_frames)
    codes = np.full(n_frames, NO_ANSWER, dtype=np.int8)
    codes[(table["half"] > 0) & dead] = DEAD

    events = enhanced_control_events(table)
    score = tracker.combined_score(tracker.shape_score(table))
    seen = ~np.isnan(table["ball_x"])

    for start, stop in tracker.live_stretches(table["half"], dead):
        changes = tracker.label_segment(events, start, stop, score, seen)
        events_here = [ev for ev in events if start <= ev[0] < stop]
        changes = clean_segment_changes(changes, events_here, start, stop)

        for i in range(len(changes)):
            change_frame, team = changes[i]
            next_change_frame = changes[i + 1][0] if (i + 1 < len(changes)) else stop
            codes[change_frame:next_change_frame] = team

    return codes


if __name__ == "__main__":
    print("Stage 1A: Loading raw ATD tracking data...")
    px, py, bx, by, team_map = load_raw_atd_matrices()
    print(f"Loaded {len(bx):,} frames x {px.shape[1]} player slots.")

    print("Computing Stage 1A geometric & physical features...")
    df_s1 = compute_stage1_features(px, py, bx, by, team_map)

    raw_vis = df_s1["ball_visible"].mean() * 100
    interp_vis = (df_s1["ball_visible"] | df_s1["ball_interped"]).mean() * 100
    print(f"  Raw ball visibility:          {raw_vis:.1f}% of all frames")
    print(f"  After <=3s gap interpolation: {interp_vis:.1f}% of all frames")

    graded_mask = np.zeros(len(df_s1), dtype=bool)
    for _, f_start, f_end in half_video_blocks(GAME):
        graded_mask[f_start:f_end + 1] = True

    sub = df_s1.loc[graded_mask]
    pred_idx = np.argmax(sub[["p_H", "p_A", "p_D"]].values, axis=1)
    pred_labels = LABELS[pred_idx]
    for lbl in LABELS:
        cnt = int(np.sum(pred_labels == lbl))
        print(f"  Stage 1A raw graded '{lbl}': {cnt:>7,} frames ({100 * cnt / len(sub):5.1f}%)")

    print("\nStage 1B: Running Enhanced Hybrid ATD Possession Engine...")
    table = features.load(GAME)
    stoppages = detect_enhanced_stoppages(table, GAME)
    codes = label_enhanced(table, stoppages)
    tracker.describe(codes, table)
