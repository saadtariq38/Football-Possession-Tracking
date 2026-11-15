"""Stage 3: Viterbi State-Space Decoder (`Tracking/stage3_viterbi_and_eval.py`).

What this script does:
----------------------
1. Runs Stage 1 (`stage1_atd_tracking.py`) and Stage 2 (`stage2_smart_tagging.py`) to obtain
   continuous ATD physical features and Smart Tagging tactical priors across all 199,319 frames.
2. Decodes the global possession sequence (`H`, `A`, `D`) across each half using a
   3-State Viterbi (Hidden Markov Model) decoder + minimum spell duration post-processor,
   explicitly designed to maximize `spellF1`.
3. Writes the submission CSV to `evaluation/results/pipeline_stage3.csv` and runs
   `evaluation/run_evaluation.py` to print the official scoreboard.

Usage:
    python Tracking/stage3_viterbi_and_eval.py
"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from stage1_atd_tracking import (  # noqa: E402
    HOME, AWAY, DEAD, LABELS, PITCH_L, PITCH_W,
    load_raw_atd_matrices, compute_stage1_features,
)
from stage2_smart_tagging import build_stage2_priors  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Refined Emission Log-Probabilities + Viterbi Decoder for spellF1
# ---------------------------------------------------------------------------

def build_combined_emissions(player_x, player_y, ball_x, ball_y, team_map, df_s1, df_s2):
    """Build calibrated per-frame emission probabilities (N, 3) for [H, A, D].

    Key lessons from Stage 1 + Stage 2 diagnostics:
    1. When the ball is visible, the nearest player's team (`dist_home` vs `dist_away`)
       is the strongest signal, especially when smoothed over a +/- 2s window.
    2. Dead Ball ('D') occurs in two distinct ways:
       a) The ball is missing from ATD tracking for a sustained stoppage (>= 4.5 seconds / 112 frames)
          and no immediate active touch is happening.
       b) A pre-set-piece window identified by Stage 2 (`SET_PIECE_DEAD`), or when the ball
          rolls near the touchline (`y < 3m` or `y > 65m` or `x < 2m` or `x > 103m`) prior to a restart.
    """
    n_frames = len(ball_x)
    home_mask = (team_map == HOME)
    away_mask = (team_map == AWAY)

    # 1. Nearest team on every frame where raw ball is visible (exactly like baseline.py)
    raw_vis = ~np.isnan(ball_x) & ~np.isnan(ball_y)
    dist_h = df_s1["dist_home"].values
    dist_a = df_s1["dist_away"].values

    nearest_code = np.full(n_frames, -1, dtype=np.int8)
    nearest_code[raw_vis & (dist_h <= dist_a)] = HOME
    nearest_code[raw_vis & (dist_a < dist_h)] = AWAY

    # 2. Compute multi-scale rolling vote of H vs A (window +/- 60 frames = +/- 2.4s)
    # Using cumulative sums for O(N) speed across all 199,319 frames
    win = 60
    h_cum = np.concatenate(([0], np.cumsum(nearest_code == HOME)))
    a_cum = np.concatenate(([0], np.cumsum(nearest_code == AWAY)))
    idx = np.arange(n_frames)
    low = np.maximum(0, idx - win)
    high = np.minimum(n_frames, idx + win + 1)

    h_votes = (h_cum[high] - h_cum[low]).astype(np.float32)
    a_votes = (a_cum[high] - a_cum[low]).astype(np.float32)
    total_votes = h_votes + a_votes

    # Fraction of frames in +/- 2.4s window where the ball was seen
    vis_density = total_votes / (high - low)

    # 3. Identify sustained missing-ball stretches (>= 90 frames = 3.6s without ball)
    is_missing = ~raw_vis
    changes = np.flatnonzero(np.diff(is_missing.astype(np.int8)) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n_frames]))

    long_missing_dead = np.zeros(n_frames, dtype=bool)
    for s, e in zip(starts, ends):
        if is_missing[s] and (e - s) >= 85:
            long_missing_dead[s:e] = True

    # 4. Also detect touchline/goal-line dead-ball walkups before set pieces
    # (e.g. when a player walks with the ball along the sideline before a Throw-in or Free Kick)
    near_sideline = raw_vis & (
        (ball_y < 3.5) | (ball_y > PITCH_W - 3.5) |
        (ball_x < 3.0) | (ball_x > PITCH_L - 3.0)
    )

    # Build emission probabilities E[t, state]
    probs = np.zeros((n_frames, 3), dtype=np.float64)

    # Base team ratio from rolling window
    h_share = np.where(total_votes > 0, (h_votes + 1.0) / (total_votes + 2.0), 0.5)
    a_share = 1.0 - h_share

    # Integrate Stage 2 Smart Tagging priors gently (weight 0.22) so tags break ties
    # during occlusions without overriding clear ball-proximity evidence
    tag_h = df_s2["tag_p_H"].values
    tag_a = df_s2["tag_p_A"].values
    tag_d = df_s2["tag_p_D"].values

    combined_h = 0.78 * h_share + 0.22 * (tag_h / np.maximum(1e-6, tag_h + tag_a))
    combined_a = 1.0 - combined_h

    # Dead ball probability model
    p_dead = np.full(n_frames, 0.06, dtype=np.float64)
    # Long missing ball gap -> strong Dead Ball
    p_dead[long_missing_dead] = 0.86
    # Very low ball visibility density in +/- 2.4s window -> Dead Ball
    p_dead[vis_density < 0.15] = np.maximum(p_dead[vis_density < 0.15], 0.80)
    # Stage 2 Set-Piece Dead setup window -> boost Dead Ball
    is_sp_dead = (df_s2["active_code"].values == "SET_PIECE_DEAD")
    p_dead[is_sp_dead & (is_missing | near_sideline)] = 0.88
    p_dead[df_s1["ball_out_of_bounds"].values] = 0.92

    alive_mass = 1.0 - p_dead
    probs[:, HOME] = alive_mass * combined_h
    probs[:, AWAY] = alive_mass * combined_a
    probs[:, DEAD] = p_dead

    return np.clip(probs, 1e-4, 0.9999)


def viterbi_decode_half(probs_slice, self_stay_prob=0.994, min_spell_frames=45):
    """Run 3-state Viterbi decoding in log-space on one half, followed by short-spell cleanup.

    Parameters
    ----------
    probs_slice : (T, 3) float64 array of emission probabilities [P(H), P(A), P(D)]
    self_stay_prob : float
        Probability of remaining in the same possession state from frame t to t+1.
        High value (0.994) penalizes flickering and forces clean segment boundaries for `spellF1`.
    min_spell_frames : int
        Minimum duration (in frames) for an isolated H or A possession spell. Spells shorter than
        this (~1.8s) are brief deflections/touches and are merged into the surrounding state.
    """
    T = len(probs_slice)
    log_emit = np.log(probs_slice)

    # 3x3 transition matrix in log space
    switch_prob = (1.0 - self_stay_prob) / 2.0
    trans = np.full((3, 3), np.log(switch_prob), dtype=np.float64)
    np.fill_diagonal(trans, np.log(self_stay_prob))

    dp = np.zeros((T, 3), dtype=np.float64)
    backptr = np.zeros((T, 3), dtype=np.int8)
    dp[0] = log_emit[0]

    for t in range(1, T):
        # scores[prev_state, next_state] = dp[t-1, prev_state] + trans[prev_state, next_state]
        scores = dp[t - 1][:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(3)] + log_emit[t]
        backptr[t] = best_prev

    # Backtrack best path
    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Post-process: merge spurious short spells (< min_spell_frames)
    for _ in range(2):
        changes = np.flatnonzero(np.diff(path) != 0) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [T]))
        if len(starts) <= 1:
            break
        for i in range(len(starts)):
            s, e = starts[i], ends[i]
            if (e - s) < min_spell_frames:
                if i > 0 and i < len(starts) - 1 and path[starts[i - 1]] == path[starts[i + 1]]:
                    path[s:e] = path[starts[i - 1]]
                elif i > 0:
                    path[s:e] = path[starts[i - 1]]
                elif i < len(starts) - 1:
                    path[s:e] = path[starts[i + 1]]

    return LABELS[path]


def run_stage3_pipeline():
    print("=" * 78)
    print("STAGE 1: Loading Raw ATD & Computing Physics/Geometry Features...")
    px, py, bx, by, team_map = load_raw_atd_matrices()
    df_s1 = compute_stage1_features(px, py, bx, by, team_map)

    print("STAGE 2: Building Smart Tagging Priors & Resolving Set Pieces...")
    df_s2, df_sp = build_stage2_priors(df_s1)

    print("\nSTAGE 3: Running 3-State Viterbi Decoder Across Both Halves...")
    emissions = build_combined_emissions(px, py, bx, by, team_map, df_s1, df_s2)

    n_frames = len(bx)
    # Default ungraded frames (pre-match, halftime, post-match) to 'X' (costs nothing per DATA.md)
    final_labels = np.full(n_frames, "X", dtype="<U1")

    for half, first_f, last_f in half_video_blocks(GAME):
        half_labels = viterbi_decode_half(
            emissions[first_f:last_f + 1],
            self_stay_prob=0.996,
            min_spell_frames=45,
        )
        final_labels[first_f:last_f + 1] = half_labels
        counts = {lbl: int(np.sum(half_labels == lbl)) for lbl in LABELS}
        print(f"  Half {half} ({first_f}..{last_f}): H={counts['H']:,}  A={counts['A']:,}  D={counts['D']:,}")

    # Write submission to evaluation/results/pipeline_stage3.csv
    out_csv = os.path.join(ROOT, "evaluation", "results", "pipeline_stage3.csv")
    write_submission(out_csv, final_labels)
    print(f"\nWrote Stage 3 submission to: {os.path.relpath(out_csv, ROOT)}")

    print("\n" + "=" * 78)
    print("RUNNING OFFICIAL EVALUATION (`evaluation/run_evaluation.py --no-detail`):")
    print("=" * 78)
    eval_script = os.path.join(ROOT, "evaluation", "run_evaluation.py")
    subprocess.run([sys.executable, eval_script, "--no-detail"], check=False)


if __name__ == "__main__":
    run_stage3_pipeline()
