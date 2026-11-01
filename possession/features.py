"""Per-frame measurements: the table every later step reads instead of the raw tracking.

One row per video frame (199319), one column per measurement, held as a dict of 1-D numpy arrays
and cached in data/cache/. Nothing here decides possession or dead ball -- it only measures, so a
decision rule can be changed without re-reading anything.

Units are metres, seconds and m/s. x runs along the pitch (0..105.3), y across it (0..68).
"Height" is how far a point is from a team's OWN goal line, so it already accounts for which way
that team attacks in that half: a centre-back has a small height, a striker a large one.

    .venv/bin/python -m possession.features            # build, cache, print a summary
    .venv/bin/python -m possession.features --refresh  # rebuild even if the cache is fresh

Column groups (prefix ``home_`` / ``away_`` where it is per team):

    ball_*        cleaned ball position, velocity, acceleration, distance to the lines
    near_*        nearest player of each team to the ball, and whether they move with it
    *_shape       team centroid, spread, defensive line, goal-side count, forward velocity
    *_closing     how fast a team's players close on their nearest opponent (pressing)
    line_*        players standing at a touchline, a goal line or a corner flag
    tag_*         Smart Tagging hints
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession import atd, tagging  # noqa: E402

# Params -- fixed in advance from football and physics, never fitted to this match's labels.
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
# Bump when a column changes meaning, so an old cache is rebuilt rather than misread.
FEATURE_VERSION = 3

# Centred differences over +/-2 frames (0.16 s): short enough to see a kick, long enough that a
# few centimetres of position noise does not become metres per second.
VELOCITY_HALF_WINDOW = 2
# A ball that jumps more than this in one frame and straight back is a glitch: 1.5 m in 0.04 s is
# 37 m/s, about the hardest shot in football, and real shots do not return the next frame.
SPIKE_JUMP_M = 1.5
# A ball further outside the pitch than this is not the match ball -- a spare ball, the crowd.
MAX_BALL_OFF_PITCH_M = 8.0
# Missing-ball gaps up to 0.2 s are bridged by a straight line; anything longer is left missing.
FILL_GAP_FRAMES = 5
# Faster than the hardest shots ever measured: a velocity above this is the tracker jumping between
# balls (warm-ups, a spare ball), so it is discarded rather than read as a kick.
MAX_BALL_SPEED = 40.0
# A player counts as in the opponent's half once this far past the halfway line, which leaves room
# for the kick-off taker standing at the ball and for tracking error.
HALFWAY_MARGIN_M = 1.0
# Players of both teams this close to the ball = a contested ball (duel).
DUEL_RADIUS_M = 3.0
# "At the line" means within this distance of it, or beyond it.
LINE_MARGIN_M = 1.0
CORNER_RADIUS_M = 3.0
# Below this a player counts as standing still.
STILL_SPEED = 0.5
# Pressing is a change of distance over ~1 s, so it gets a wider window than velocity.
CLOSING_HALF_WINDOW = 12
# Frames per block in the all-pairs player distance step; bounds memory at a few tens of MB.
CHUNK = 10000


def cache_path(game=GAME, data_dir=None):
    return os.path.join(atd.cache_dir(data_dir), "{}_features.npz".format(game))


def half_of_frames(n_frames, game=GAME):
    """Per-frame half: 1, 2, or 0 outside the graded blocks."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def home_forward_sign(n_frames, direction, game=GAME):
    """+1 where the home side attacks towards larger x, -1 where towards smaller x.

    Ends swap at halftime. Every frame from the second-half block onwards is the second half;
    everything before it, including the pre-match frames, counts as the first.
    """
    second_half_start = [first for number, first, _ in half_video_blocks(game) if number == 2][0]
    sign = np.full(n_frames, direction, dtype=np.int8)
    sign[second_half_start:] = -direction
    return sign


def centred_rate(values, half_window):
    """Rate of change per second along axis 0, (v[i+h] - v[i-h]) / (2h / FPS). NaN at the ends
    and wherever either endpoint is missing, so a gap never produces a fake velocity."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    out[half_window:-half_window] = ((values[2 * half_window:] - values[:-2 * half_window])
                                     * (FPS / (2.0 * half_window)))
    return out


def fill_short_gaps(x, y, max_frames):
    """Straight-line bridge over NaN runs of at most max_frames with a known point either side.

    Returns (x, y, bridged) with bridged True on the frames that were filled in.
    """
    missing = np.isnan(x).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    bridged = np.zeros(len(x), dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start > max_frames or start == 0 or stop == len(x):
            continue
        weight = np.arange(1, stop - start + 1) / float(stop - start + 1)
        x[start:stop] = x[start - 1] + weight * (x[stop] - x[start - 1])
        y[start:stop] = y[start - 1] + weight * (y[stop] - y[start - 1])
        bridged[start:stop] = True
    return x, y, bridged


def clean_ball(ball):
    """(x, y, seen, removed, bridged) in metres from the raw normalised ball track.

    seen is the raw "the tracker reported a ball" flag; removed marks reports thrown away as too far
    off the pitch or as one-frame spikes; bridged marks short gaps filled by a straight line.
    """
    x = ball[:, 0].astype(np.float64) * PITCH_LENGTH
    y = ball[:, 1].astype(np.float64) * PITCH_WIDTH
    seen = ~np.isnan(x)

    off = ((x < -MAX_BALL_OFF_PITCH_M) | (x > PITCH_LENGTH + MAX_BALL_OFF_PITCH_M)
           | (y < -MAX_BALL_OFF_PITCH_M) | (y > PITCH_WIDTH + MAX_BALL_OFF_PITCH_M))
    x[off] = y[off] = np.nan

    # A spike is far from both neighbours while the neighbours are close to each other. NaN
    # compares False, so a frame at the edge of a gap is never called a spike.
    step = np.full(len(x), np.nan)
    step[1:] = np.hypot(np.diff(x), np.diff(y))
    across = np.full(len(x), np.nan)
    across[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])
    after = np.append(step[1:], np.nan)
    with np.errstate(invalid="ignore"):
        spike = (step > SPIKE_JUMP_M) & (after > SPIKE_JUMP_M) & (across < SPIKE_JUMP_M)
    x[spike] = y[spike] = np.nan

    x, y, bridged = fill_short_gaps(x, y, FILL_GAP_FRAMES)
    return x, y, seen, off | spike, bridged


def nearest_to_ball(px, py, pvx, pvy, team, side, bx, by, bvx, bvy):
    """(distance, slot, relative_speed, n_within_duel_radius) for one side's nearest player.

    relative_speed is |player velocity - ball velocity|: near zero when the player runs with the
    ball at their feet, large when a ball flies past them. That is the "moving the same way" test,
    and it is also what tells a ball overhead from a ball at someone's feet, since the tracking
    has no ball height.
    """
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where((team == side)[None, :] & ~np.isnan(dist), dist, np.inf)
    slot = np.argmin(dist, axis=1)
    rows = np.arange(len(bx))
    nearest = dist[rows, slot].astype(np.float32)
    nearest[np.isinf(nearest)] = np.nan
    relative = np.hypot(pvx[rows, slot] - bvx, pvy[rows, slot] - bvy).astype(np.float32)
    relative[np.isnan(nearest)] = np.nan
    within = (dist <= DUEL_RADIUS_M).sum(axis=1).astype(np.int8)
    return nearest, slot.astype(np.int8), relative, within


def nearest_opponent_distance(px, py, team):
    """(n, slots) distance from each player to the nearest player of the other side."""
    out = np.full(px.shape, np.nan, dtype=np.float32)
    home, away = np.flatnonzero(team == atd.HOME), np.flatnonzero(team == atd.AWAY)
    for start in range(0, len(px), CHUNK):
        block = slice(start, start + CHUNK)
        pair = np.hypot(px[block][:, home, None] - px[block][:, None, away],
                        py[block][:, home, None] - py[block][:, None, away])
        out[block, home] = np.nanmin(pair, axis=2)
        out[block, away] = np.nanmin(pair, axis=1)
    return out


def team_shape(prefix, columns, px, py, pvx, speed, closing, forward, ball_x):
    """Shape of one side, measured without the ball except for goal_side.

    forward is the per-frame sign of this side's attacking direction along x.
    """
    tx, ty = px[:, columns], py[:, columns]
    height = np.where(forward[:, None] > 0, tx, PITCH_LENGTH - tx)
    ball_height = np.where(forward > 0, ball_x, PITCH_LENGTH - ball_x)

    # Second-lowest height is the defensive line: the lowest is usually the goalkeeper.
    ordered = np.sort(np.where(np.isnan(height), np.inf, height), axis=1)
    line = ordered[:, 1].astype(np.float32)
    line[np.isinf(line)] = np.nan

    with np.errstate(invalid="ignore"):
        goal_side = np.sum(height < ball_height[:, None], axis=1).astype(np.float32)
    goal_side[np.isnan(ball_x)] = np.nan

    with np.errstate(invalid="ignore"):
        past_halfway = np.sum(height > PITCH_LENGTH / 2 + HALFWAY_MARGIN_M, axis=1)

    return {
        prefix + "n_players": np.sum(~np.isnan(tx), axis=1).astype(np.int8),
        prefix + "n_past_halfway": past_halfway.astype(np.int8),
        prefix + "centroid_height": np.nanmean(height, axis=1),
        prefix + "centroid_y": np.nanmean(ty, axis=1),
        prefix + "spread_length": np.nanstd(tx, axis=1),
        prefix + "spread_width": np.nanstd(ty, axis=1),
        prefix + "line_height": line,
        prefix + "goal_side": goal_side,
        prefix + "forward_velocity": np.nanmean(pvx[:, columns], axis=1) * forward,
        prefix + "mean_speed": np.nanmean(speed[:, columns], axis=1),
        prefix + "closing": np.nanmean(closing[:, columns], axis=1),
    }


def line_presence(px, py, speed):
    """Counts of players at each touchline, at the goal lines and at the corner flags.

    The touchlines are kept apart (low = y near 0, high = y near 68) because a throw-in happens on
    one specific side, and that side has to match where the ball was last seen.
    """
    still = speed < STILL_SPEED
    low = py < LINE_MARGIN_M
    high = py > PITCH_WIDTH - LINE_MARGIN_M
    goal = (px < LINE_MARGIN_M) | (px > PITCH_LENGTH - LINE_MARGIN_M)
    corner = np.zeros(px.shape, dtype=bool)
    for cx in (0.0, PITCH_LENGTH):
        for cy in (0.0, PITCH_WIDTH):
            corner |= np.hypot(px - cx, py - cy) < CORNER_RADIUS_M
    return {
        "line_touch_low": low.sum(axis=1).astype(np.int8),
        "line_touch_high": high.sum(axis=1).astype(np.int8),
        "line_touch_low_still": (low & still).sum(axis=1).astype(np.int8),
        "line_touch_high_still": (high & still).sum(axis=1).astype(np.int8),
        "line_goal": goal.sum(axis=1).astype(np.int8),
        "line_corner": corner.sum(axis=1).astype(np.int8),
    }


def build(game=GAME, data_dir=None, refresh=False):
    """The full measurements table for one game, as a dict of equal-length 1-D arrays."""
    tracking = atd.load(game, data_dir, refresh)
    players, team = tracking["players"], tracking["team"]
    n_frames = len(players)

    px = players[:, :, 0] * np.float32(PITCH_LENGTH)
    py = players[:, :, 1] * np.float32(PITCH_WIDTH)
    pvx = centred_rate(px, VELOCITY_HALF_WINDOW)
    pvy = centred_rate(py, VELOCITY_HALF_WINDOW)
    speed = np.hypot(pvx, pvy)

    bx, by, seen, removed, bridged = clean_ball(tracking["ball"])
    bvx = centred_rate(bx, VELOCITY_HALF_WINDOW)
    bvy = centred_rate(by, VELOCITY_HALF_WINDOW)
    with np.errstate(invalid="ignore"):
        too_fast = np.hypot(bvx, bvy) > MAX_BALL_SPEED
    bvx[too_fast] = bvy[too_fast] = np.nan
    bax = centred_rate(bvx, VELOCITY_HALF_WINDOW)
    bay = centred_rate(bvy, VELOCITY_HALF_WINDOW)

    home_sign = home_forward_sign(n_frames, tracking["direction"], game)

    table = {
        "half": half_of_frames(n_frames, game),
        "home_forward": home_sign,
        "ball_seen": seen,
        "ball_removed": removed,
        "ball_bridged": bridged,
        "ball_too_fast": too_fast,
        "ball_x": bx.astype(np.float32),
        "ball_y": by.astype(np.float32),
        "ball_vx": bvx,
        "ball_vy": bvy,
        "ball_speed": np.hypot(bvx, bvy),
        "ball_accel": np.hypot(bax, bay),
        # Signed: negative once the ball is over the line.
        "ball_touchline_m": np.minimum(by, PITCH_WIDTH - by).astype(np.float32),
        "ball_goalline_m": np.minimum(bx, PITCH_LENGTH - bx).astype(np.float32),
        "all_mean_speed": None,
    }

    with warnings.catch_warnings():
        # nanmean / nanmin over a frame with nobody visible warns and returns NaN, which is the
        # right answer for an empty frame; the warning is noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        closing = centred_rate(nearest_opponent_distance(px, py, team), CLOSING_HALF_WINDOW)
        table["all_mean_speed"] = np.nanmean(speed, axis=1)
        for side, name in ((atd.HOME, "home"), (atd.AWAY, "away")):
            nearest, slot, relative, within = nearest_to_ball(
                px, py, pvx, pvy, team, side, bx, by, bvx, bvy)
            table["near_{}_dist".format(name)] = nearest
            table["near_{}_slot".format(name)] = slot
            table["near_{}_rel_speed".format(name)] = relative
            table["near_{}_duel_count".format(name)] = within
            forward = home_sign if side == atd.HOME else -home_sign
            table.update(team_shape("{}_".format(name), team == side, px, py, pvx, speed,
                                    closing, forward, bx))

    table.update(line_presence(px, py, speed))
    hints = tagging.frame_hints(tracking["names"], n_frames, game, data_dir)
    table.update({"tag_" + key: value for key, value in hints.items()})
    return table


def load(game=GAME, data_dir=None, refresh=False):
    """The table from cache if it was built from the current tracking with this FEATURE_VERSION."""
    path = cache_path(game, data_dir)
    raw_path = atd.require(["atd_raw"], game, data_dir)[0]
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path)),
                      FEATURE_VERSION], dtype=np.int64)
    if not refresh and os.path.isfile(path):
        with np.load(path) as stored:
            if "_stamp" in stored and np.array_equal(stored["_stamp"], stamp):
                return {key: stored[key] for key in stored.files if key != "_stamp"}

    table = build(game, data_dir, refresh)
    np.savez_compressed(path, _stamp=stamp, **table)
    return table


def summary(table, game=GAME):
    """What the table says, with no ground truth involved. A sanity check, not an evaluation."""
    graded = table["half"] > 0
    n_graded = int(graded.sum())
    print("{} frames, {} graded, {} columns".format(len(graded), n_graded, len(table)))

    print("\nball")
    print("  reported by the tracker   {:5.1f}% of graded frames".format(
        100.0 * table["ball_seen"][graded].mean()))
    print("  thrown away as glitches   {:>6} frames".format(int(table["ball_removed"].sum())))
    print("  short gaps bridged        {:>6} frames".format(int(table["ball_bridged"].sum())))
    print("  impossible speed dropped  {:>6} frames".format(int(table["ball_too_fast"].sum())))
    usable = graded & ~np.isnan(table["ball_x"])
    print("  usable after cleaning     {:5.1f}% of graded frames".format(
        100.0 * usable.sum() / n_graded))
    print("  speed m/s  p50 {:.1f}  p90 {:.1f}  p99 {:.1f}".format(
        *np.nanpercentile(table["ball_speed"][usable], [50, 90, 99])))

    print("\nnearest player to the ball (ball usable)")
    for name in ("home", "away"):
        dist = table["near_{}_dist".format(name)][usable]
        rel = table["near_{}_rel_speed".format(name)][usable]
        print("  {}  distance p50 {:.1f} m   moving with the ball (rel < 2 m/s, < 1.5 m) {:.1f}%"
              .format(name, np.nanmedian(dist), 100.0 * np.mean((dist < 1.5) & (rel < 2.0))))

    # Direction check: at each kickoff every player stands in their own half, so both sides must
    # have a centroid height under half the pitch. If this fails the forward sign is backwards
    # and every team-shape column is inverted.
    print("\nattack direction check at the kickoffs")
    for frame in (11185, 113742):
        heights = (table["home_centroid_height"][frame], table["away_centroid_height"][frame])
        ok = all(h < PITCH_LENGTH / 2 for h in heights)
        print("  frame {:>6}  home {:5.1f} m  away {:5.1f} m from own goal   {}".format(
            frame, heights[0], heights[1], "OK" if ok else "BACKWARDS?"))

    print("\nteam shape over graded frames (median)")
    for name in ("home", "away"):
        print("  {}  players {:.0f}  centroid {:.1f} m from own goal  line {:.1f} m  "
              "spread {:.1f} x {:.1f} m".format(
                  name, np.median(table[name + "_n_players"][graded]),
                  np.nanmedian(table[name + "_centroid_height"][graded]),
                  np.nanmedian(table[name + "_line_height"][graded]),
                  np.nanmedian(table[name + "_spread_length"][graded]),
                  np.nanmedian(table[name + "_spread_width"][graded])))

    print("\nsmart tagging coverage of graded frames")
    print("  some side named as in possession  {:5.1f}%".format(
        100.0 * np.mean((table["tag_home_votes"] + table["tag_away_votes"])[graded] > 0)))
    print("  in the first 10 s of a set piece  {:5.1f}%".format(
        100.0 * np.mean(table["tag_set_piece_age_s"][graded] < 10)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()
    summary(load(refresh=args.refresh))
