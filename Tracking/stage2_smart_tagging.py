"""Stage 2: Tactical Phase & Set-Piece Priors from Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`).

What this script does:
----------------------
1. Loads all 638 automatic tactical annotations (`SET PIECES`, `BUILD UP`, `PROGRESSION`,
   `DEFENDING IN ...`, `GOALS`, etc.) aligned to the 0-based tactical video frames.
2. Translates club names (`Barcelona` -> Home 'H', `Atletic Club` -> Away 'A') and inverts
   defensive phase tags (`DEFENDING IN ...` by Team X means the OPPOSING team has possession!).
3. Pinpoints exact Dead-Ball ('D') -> Live Restart ('H' or 'A') transition frames inside
   each of the 59 `SET PIECES` intervals by combining the tag window with Stage 1 ball kinematics
   (finding the exact frame where the stationary/out-of-play ball is kicked or thrown back into play).
4. Resolves the 26 `Throw-in` instances (where Smart Tagging always writes `Team = N/A`) by
   identifying which team's player is standing at the touchline at the moment of the throw.
5. Outputs per-frame Stage 2 probability modifiers / priors (`tag_p_H`, `tag_p_A`, `tag_p_D`)
   and can fuse them with Stage 1 to produce a combined prediction.

Run standalone to inspect Smart Tagging priors and see how they refine Stage 1:
    python Tracking/stage2_smart_tagging.py
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from games import GAME, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tags  # noqa: E402

HOME, AWAY, DEAD = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Phase code classification
ATTACKING_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "CREATING CHANCES",
    "LONG BALL",
    "ATTACKING TRANSITION",
    "GOALS",
}

DEFENSIVE_CODES = {
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}


def map_tag_team(team_str):
    """Map Smart Tagging 'Team' string to HOME (0), AWAY (1), or None ('N/A')."""
    if not team_str:
        return None
    t = team_str.strip().lower()
    if "barcelona" in t:
        return HOME
    if "atletic" in t or "athletic" in t:
        return AWAY
    return None


def possessing_team_of_event(event):
    """Determine which team is in possession during a Smart Tagging event.

    Crucial rule:
    - For attacking codes (`BUILD UP`, `PROGRESSION`, etc.), the tagged `Team` has the ball.
    - For defensive codes (`DEFENDING IN MIDDLE THIRD`, etc.), the tagged `Team` is defending,
      which means the OPPOSING team has the ball!
    """
    code = (event.get("code") or "").strip().upper()
    tagged_team = map_tag_team(event.get("tags", {}).get("Team"))

    if tagged_team is None:
        return None

    if code in ATTACKING_CODES or code == "SET PIECES":
        return tagged_team
    if code in DEFENSIVE_CODES:
        return AWAY if tagged_team == HOME else HOME
    return tagged_team


def find_set_piece_restart_frame(event, df_stage1=None):
    """Split a `SET PIECES` interval `[start_frame, end_frame]` into:
      1. Dead-ball setup (`[dead_start, restart_frame)`) -> Label 'D'
      2. Live restart (`[restart_frame, end_frame]`)    -> Label of restarting team

    Why: A `SET PIECES` tag in Smart Tagging spans ~5 to 10 seconds around a Corner, Free Kick,
    Goal Kick, Throw-in, or Kick Off. The first half of that window is Dead Ball ('D') while
    players set up; the exact moment the ball is struck (`ball_speed > 3.5 m/s` inside the pitch)
    is when live possession begins.
    """
    s_frame = max(0, event["start_frame"])
    e_frame = event["end_frame"]

    if df_stage1 is None:
        # Fallback if Stage 1 kinematics aren't passed: midpoint of the tagged window
        restart_frame = (s_frame + e_frame) // 2
        return s_frame, restart_frame, possessing_team_of_event(event)

    n_frames = len(df_stage1)
    e_frame = min(n_frames - 1, e_frame)
    window = df_stage1.iloc[s_frame:e_frame + 1]

    # Look for the frame where the ball is visible, in-bounds, and accelerates (> 3.5 m/s)
    in_play = window["ball_visible"] & (~window["ball_out_of_bounds"])
    fast_ball = in_play & (window["ball_speed"] >= 3.5)

    if fast_ball.any():
        restart_frame = int(window.loc[fast_ball, "frame"].iloc[0])
    elif in_play.any():
        restart_frame = int(window.loc[in_play, "frame"].iloc[0])
    else:
        restart_frame = (s_frame + e_frame) // 2

    # Look backwards up to 10 seconds (250 frames) before `restart_frame` to find when the ball
    # actually went dead (out of bounds or missing before the set piece)
    lookback_start = max(0, s_frame - 200)
    pre_window = df_stage1.iloc[lookback_start:restart_frame]
    dead_candidates = pre_window["ball_out_of_bounds"] | (~pre_window["ball_visible"])
    if dead_candidates.any():
        first_dead = int(pre_window.loc[dead_candidates, "frame"].iloc[0])
        dead_start = min(s_frame, first_dead)
    else:
        dead_start = s_frame

    # Determine restarting team (resolve Throw-in 'N/A' using nearest player at restart_frame)
    team = possessing_team_of_event(event)
    if team is None:
        # Throw-ins have Team = N/A in Smart Tagging; check which team is closest at restart
        win_post = df_stage1.iloc[restart_frame:min(n_frames, restart_frame + 25)]
        h_dist = np.median(win_post["dist_home"])
        a_dist = np.median(win_post["dist_away"])
        team = HOME if h_dist <= a_dist else AWAY

    return dead_start, restart_frame, team


def build_stage2_priors(df_stage1=None, game=GAME, data_dir=None):
    """Build per-frame tactical priors from Smart Tagging (`tag_p_H`, `tag_p_A`, `tag_p_D`).

    Parameters
    ----------
    df_stage1 : pd.DataFrame, optional
        Output of `stage1_atd_tracking.compute_stage1_features()`. Used to pinpoint exact
        set-piece restart frames and resolve `Team = N/A` throw-ins.

    Returns
    -------
    df_s2 : pd.DataFrame with columns:
        `frame`, `active_code`, `set_piece_type`, `tag_team`, `tag_p_H`, `tag_p_A`, `tag_p_D`
    """
    n_frames = len(df_stage1) if df_stage1 is not None else source_frames("tactical", game)
    events = load_smart_tags(game, data_dir)

    # Neutral uniform prior (1/3 each) where no tag applies
    tag_p_H = np.full(n_frames, 1.0 / 3.0, dtype=np.float32)
    tag_p_A = np.full(n_frames, 1.0 / 3.0, dtype=np.float32)
    tag_p_D = np.full(n_frames, 1.0 / 3.0, dtype=np.float32)

    active_code = np.full(n_frames, "", dtype=object)
    set_piece_type = np.full(n_frames, "", dtype=object)
    tag_team = np.full(n_frames, "", dtype=object)

    # 1. Apply continuous Phase-of-Play priors (BUILD UP, PROGRESSION, DEFENDING, etc.)
    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        if code == "SET PIECES":
            continue

        s = max(0, ev["start_frame"])
        e = min(n_frames, ev["end_frame"])
        if s >= e:
            continue

        team = possessing_team_of_event(ev)
        active_code[s:e] = code
        if team == HOME:
            tag_team[s:e] = "H"
            # Soft phase prior: boosts possessing team, suppresses Dead Ball during active phase
            tag_p_H[s:e] = 0.65
            tag_p_A[s:e] = 0.25
            tag_p_D[s:e] = 0.10
        elif team == AWAY:
            tag_team[s:e] = "A"
            tag_p_H[s:e] = 0.25
            tag_p_A[s:e] = 0.65
            tag_p_D[s:e] = 0.10

    # 2. Apply Set-Piece Dead-Ball & Restart anchors (overrides phase priors at stoppages)
    set_piece_summary = []
    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        if code != "SET PIECES":
            continue

        sp_type = ev.get("tags", {}).get("Type", "Unknown")
        dead_start, restart_f, team = find_set_piece_restart_frame(ev, df_stage1)
        end_f = min(n_frames, max(restart_f + 25, ev["end_frame"]))

        # Dead-ball stretch leading up to the restart kick/throw
        if dead_start < restart_f:
            active_code[dead_start:restart_f] = "SET_PIECE_DEAD"
            set_piece_type[dead_start:restart_f] = sp_type
            tag_team[dead_start:restart_f] = "D"
            tag_p_D[dead_start:restart_f] = 0.92
            tag_p_H[dead_start:restart_f] = 0.04
            tag_p_A[dead_start:restart_f] = 0.04

        # Post-restart live possession stretch (at least ~1-2 seconds after restart)
        if restart_f < end_f and team in (HOME, AWAY):
            active_code[restart_f:end_f] = "SET_PIECE_LIVE"
            set_piece_type[restart_f:end_f] = sp_type
            if team == HOME:
                tag_team[restart_f:end_f] = "H"
                tag_p_H[restart_f:end_f] = 0.85
                tag_p_A[restart_f:end_f] = 0.10
                tag_p_D[restart_f:end_f] = 0.05
            else:
                tag_team[restart_f:end_f] = "A"
                tag_p_H[restart_f:end_f] = 0.10
                tag_p_A[restart_f:end_f] = 0.85
                tag_p_D[restart_f:end_f] = 0.05

        set_piece_summary.append({
            "type": sp_type,
            "xml_start": ev["start_frame"],
            "xml_end": ev["end_frame"],
            "dead_start": dead_start,
            "restart_frame": restart_f,
            "resolved_team": "H" if team == HOME else ("A" if team == AWAY else "Unknown"),
        })

    df_s2 = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "active_code": active_code,
        "set_piece_type": set_piece_type,
        "tag_team": tag_team,
        "tag_p_H": tag_p_H,
        "tag_p_A": tag_p_A,
        "tag_p_D": tag_p_D,
    })
    return df_s2, pd.DataFrame(set_piece_summary)


def combine_stage1_and_stage2(df_s1, df_s2):
    """Multiply Stage 1 (ATD physics) and Stage 2 (Smart Tagging) probabilities in log-space."""
    p1 = df_s1[["p_H", "p_A", "p_D"]].values
    p2 = df_s2[["tag_p_H", "tag_p_A", "tag_p_D"]].values

    # Bayesian log-linear pooling: P(state | ATD, Tags) proportional to P(state | ATD) * P(state | Tags)
    joint = p1 * p2
    joint /= np.sum(joint, axis=1, keepdims=True)

    pred_codes = np.argmax(joint, axis=1)
    return LABELS[pred_codes], joint


if __name__ == "__main__":
    from stage1_atd_tracking import load_raw_atd_matrices, compute_stage1_features

    print("1. Running Stage 1 (ATD Tracking)...")
    px, py, bx, by, team_map = load_raw_atd_matrices()
    df_s1 = compute_stage1_features(px, py, bx, by, team_map)

    print("\n2. Running Stage 2 (Smart Tagging Priors & Set-Piece Alignment)...")
    df_s2, df_sp = build_stage2_priors(df_s1)
    print(f"   Processed {len(df_sp)} Set Pieces (including resolving all 26 N/A Throw-ins):")
    print(df_sp.head(10).to_string(index=False))

    labels, joint_probs = combine_stage1_and_stage2(df_s1, df_s2)
    print("\n3. Combined Stage 1 + Stage 2 Label Distribution (all 199,319 frames):")
    for lbl in LABELS:
        cnt = int(np.sum(labels == lbl))
        print(f"   Label '{lbl}': {cnt:>7,} frames ({100 * cnt / len(labels):5.1f}%)")
