"""Step 1: measure things in every frame.

The result is a table with one row per video frame (199,319 rows). Each column is one
measurement, stored as a numpy array. Later steps read this table, not the tracking file.
Nothing here decides possession; it only measures.

Units: metres, seconds, metres per second.
    x runs along the pitch (0 to 105.3), y runs across it (0 to 68).
    "Height" = how far a player is from their OWN goal line.

Columns:
    half                      1 or 2 inside the graded halves, 0 outside
    ball_seen                 the tracker reported a ball in this frame
    ball_removed              a reported ball position we threw away as a glitch
    ball_bridged              a short gap we filled in
    ball_too_fast             a ball speed above 40 m/s, which we threw away
    ball_x, ball_y            cleaned ball position (NaN if unknown)
    ball_speed                ball speed
    ball_touchline_m          distance from the ball to the nearest touchline (negative = out)
    ball_goalline_m           distance from the ball to the nearest goal line (negative = out)
    all_mean_speed            average speed of all visible players
    near_home_dist            distance from the ball to the nearest home player
    near_home_rel_speed       difference between that player's velocity and the ball's
    near_away_dist, near_away_rel_speed       the same for the away team
    home_n_players            visible home players
    home_n_past_halfway       home players in the other team's half
    home_centroid_height      average height of the home players
    home_spread_width         how spread out the home team is across the pitch
    home_mean_speed           average speed of the home players
    away_...                  the same five for the away team
    line_touch_low_still      players standing still at the touchline y = 0
    line_touch_high_still     players standing still at the touchline y = 68
    tag_home_votes, tag_away_votes, tag_set_piece_age_s    Smart Tagging hints (see tagging.py)

    .venv/bin/python -m possession.features            # build, save, print a summary
    .venv/bin/python -m possession.features --refresh  # rebuild even if a saved copy exists
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
from possession.series import runs  # noqa: E402

# Params -- chosen from football and physics, not from this match's answers.
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
# Change this number whenever the columns change, so an old saved table is rebuilt.
FEATURE_VERSION = 4

# Speed is measured over 2 frames before and 2 after (0.16 s).
VELOCITY_HALF_WINDOW = 2
# A ball that jumps more than 1.5 m in one frame (37 m/s) and straight back is a glitch.
SPIKE_JUMP_M = 1.5
# A ball more than 8 m outside the pitch is not the match ball.
MAX_BALL_OFF_PITCH_M = 8.0
# Ball gaps of up to 5 frames (0.2 s) are filled with a straight line.
FILL_GAP_FRAMES = 5
# No real shot is faster than 40 m/s, so faster readings are glitches.
MAX_BALL_SPEED = 40.0
# A player counts as in the other half only when more than 1 m past the halfway line.
HALFWAY_MARGIN_M = 1.0
# "At the line" means within 1 m of it, or beyond it.
LINE_MARGIN_M = 1.0
# Below 0.5 m/s a player is standing still.
STILL_SPEED = 0.5


def cache_path(game=GAME, data_dir=None):
    return os.path.join(atd.cache_dir(data_dir), "{}_features.npz".format(game))


def half_of_frames(n_frames, game=GAME):
    """For every frame: 1 or 2 inside a graded half, 0 outside."""
    half = np.zeros(n_frames, dtype=np.int8)
    for half_number, first_frame, last_frame in half_video_blocks(game):
        half[first_frame:last_frame + 1] = half_number
    return half


def home_forward_sign(n_frames, direction, game=GAME):
    """For every frame: +1 if the home team attacks towards x = 105.3, -1 if towards x = 0.

    direction is the first-half value. The teams swap ends from the start of the second half.
    """
    second_half_start = None
    for half_number, first_frame, last_frame in half_video_blocks(game):
        if half_number == 2:
            second_half_start = first_frame

    sign = np.full(n_frames, direction, dtype=np.int8)
    sign[second_half_start:] = -direction
    return sign


def centred_rate(values, half_window):
    """How fast values change, per second, for every frame.

    For frame i: (value at i + h  -  value at i - h) / (2h frames, in seconds).
    The result is NaN at the very start and end, and wherever either value is unknown.
    """
    rate = np.full(values.shape, np.nan, dtype=np.float32)
    later = values[2 * half_window:]
    earlier = values[:-2 * half_window]
    frames_to_seconds = FPS / (2.0 * half_window)
    rate[half_window:-half_window] = (later - earlier) * frames_to_seconds
    return rate


def fill_short_gaps(x, y, max_frames):
    """Fill short gaps in the ball track with a straight line.

    A gap is filled only if it is at most max_frames long and the ball is known on both sides.
    Returns x, y and a mask of the frames that were filled.
    """
    bridged = np.zeros(len(x), dtype=bool)
    gap_starts, gap_stops = runs(np.isnan(x))

    for i in range(len(gap_starts)):
        start = gap_starts[i]
        stop = gap_stops[i]
        gap_length = stop - start
        if gap_length > max_frames:
            continue
        if start == 0 or stop == len(x):
            continue  # no known point on one side

        # weight goes 1/(n+1), 2/(n+1), ... across the gap
        weight = np.arange(1, gap_length + 1) / float(gap_length + 1)
        x[start:stop] = x[start - 1] + weight * (x[stop] - x[start - 1])
        y[start:stop] = y[start - 1] + weight * (y[stop] - y[start - 1])
        bridged[start:stop] = True

    return x, y, bridged


def clean_ball(ball):
    """Turn the raw ball track into metres and remove obvious glitches.

    Returns:
        x, y      cleaned ball position in metres (NaN where unknown)
        seen      the tracker reported a ball
        removed   positions we threw away
        bridged   short gaps we filled in
    """
    x = ball[:, 0].astype(np.float64) * PITCH_LENGTH
    y = ball[:, 1].astype(np.float64) * PITCH_WIDTH
    seen = ~np.isnan(x)
    n = len(x)

    # 1. Far outside the pitch: not the match ball.
    too_far_left = x < -MAX_BALL_OFF_PITCH_M
    too_far_right = x > PITCH_LENGTH + MAX_BALL_OFF_PITCH_M
    too_far_down = y < -MAX_BALL_OFF_PITCH_M
    too_far_up = y > PITCH_WIDTH + MAX_BALL_OFF_PITCH_M
    off_pitch = too_far_left | too_far_right | too_far_down | too_far_up
    x[off_pitch] = np.nan
    y[off_pitch] = np.nan

    # 2. One-frame jumps: far from the frame before AND the frame after, while those two are
    #    close to each other. (Comparisons with NaN are False, so gap edges are never removed.)
    jump_from_previous = np.full(n, np.nan)
    jump_from_previous[1:] = np.hypot(np.diff(x), np.diff(y))
    jump_to_next = np.append(jump_from_previous[1:], np.nan)
    previous_to_next = np.full(n, np.nan)
    previous_to_next[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])
    with np.errstate(invalid="ignore"):
        spike = ((jump_from_previous > SPIKE_JUMP_M)
                 & (jump_to_next > SPIKE_JUMP_M)
                 & (previous_to_next < SPIKE_JUMP_M))
    x[spike] = np.nan
    y[spike] = np.nan

    # 3. Fill very short gaps.
    x, y, bridged = fill_short_gaps(x, y, FILL_GAP_FRAMES)

    removed = off_pitch | spike
    return x, y, seen, removed, bridged


def nearest_to_ball(px, py, pvx, pvy, is_this_team, bx, by, bvx, bvy):
    """For one team, in every frame: its player nearest the ball.

    Returns two arrays:
        distance         how far that player is from the ball
        relative_speed   size of (player velocity - ball velocity). Near zero means the player
                         moves with the ball, like a ball at their feet. Large means the ball
                         is flying past them.
    """
    # Distance from every player to the ball, in every frame: array (frames, 46).
    distance_to_ball = np.hypot(px - bx[:, None], py - by[:, None])

    # Ignore the other team's players and unseen players by giving them an endless distance.
    usable = is_this_team[None, :] & ~np.isnan(distance_to_ball)
    distance_to_ball = np.where(usable, distance_to_ball, np.inf)

    # The slot number of the nearest player, in every frame.
    nearest_slot = np.argmin(distance_to_ball, axis=1)
    rows = np.arange(len(bx))

    distance = distance_to_ball[rows, nearest_slot].astype(np.float32)
    distance[np.isinf(distance)] = np.nan  # nobody from this team was visible

    difference_x = pvx[rows, nearest_slot] - bvx
    difference_y = pvy[rows, nearest_slot] - bvy
    relative_speed = np.hypot(difference_x, difference_y).astype(np.float32)
    relative_speed[np.isnan(distance)] = np.nan

    return distance, relative_speed


def team_shape(name, is_this_team, px, py, speed, forward):
    """Five measurements of one team's shape, for every frame.

    name is "home" or "away". forward is +1 / -1 per frame: which way this team attacks.
    """
    team_x = px[:, is_this_team]
    team_y = py[:, is_this_team]

    # Height = distance from the team's own goal line.
    height = np.where(forward[:, None] > 0, team_x, PITCH_LENGTH - team_x)

    with np.errstate(invalid="ignore"):
        past_halfway = np.sum(height > PITCH_LENGTH / 2 + HALFWAY_MARGIN_M, axis=1)

    columns = {}
    columns[name + "_n_players"] = np.sum(~np.isnan(team_x), axis=1).astype(np.int8)
    columns[name + "_n_past_halfway"] = past_halfway.astype(np.int8)
    columns[name + "_centroid_height"] = np.nanmean(height, axis=1)
    columns[name + "_spread_width"] = np.nanstd(team_y, axis=1)
    columns[name + "_mean_speed"] = np.nanmean(speed[:, is_this_team], axis=1)
    return columns


def players_still_at_touchlines(py, speed):
    """For every frame: how many players stand still at each touchline.

    The two touchlines are counted apart, because a throw-in is on one side only.
    """
    still = speed < STILL_SPEED
    at_low_touchline = py < LINE_MARGIN_M                   # touchline at y = 0
    at_high_touchline = py > PITCH_WIDTH - LINE_MARGIN_M    # touchline at y = 68

    low_count = (at_low_touchline & still).sum(axis=1).astype(np.int8)
    high_count = (at_high_touchline & still).sum(axis=1).astype(np.int8)
    return {"line_touch_low_still": low_count, "line_touch_high_still": high_count}


def build(game=GAME, data_dir=None, refresh=False):
    """Make the whole measurements table."""
    tracking = atd.load(game, data_dir, refresh)
    players = tracking["players"]
    team = tracking["team"]
    n_frames = len(players)

    # Players: position in metres, velocity and speed.
    px = players[:, :, 0] * np.float32(PITCH_LENGTH)
    py = players[:, :, 1] * np.float32(PITCH_WIDTH)
    pvx = centred_rate(px, VELOCITY_HALF_WINDOW)
    pvy = centred_rate(py, VELOCITY_HALF_WINDOW)
    speed = np.hypot(pvx, pvy)

    # Ball: cleaned position, velocity and speed.
    bx, by, seen, removed, bridged = clean_ball(tracking["ball"])
    bvx = centred_rate(bx, VELOCITY_HALF_WINDOW)
    bvy = centred_rate(by, VELOCITY_HALF_WINDOW)
    with np.errstate(invalid="ignore"):
        too_fast = np.hypot(bvx, bvy) > MAX_BALL_SPEED
    bvx[too_fast] = np.nan
    bvy[too_fast] = np.nan

    home_sign = home_forward_sign(n_frames, tracking["direction"], game)
    away_sign = -home_sign

    table = {}
    table["half"] = half_of_frames(n_frames, game)
    table["ball_seen"] = seen
    table["ball_removed"] = removed
    table["ball_bridged"] = bridged
    table["ball_too_fast"] = too_fast
    table["ball_x"] = bx.astype(np.float32)
    table["ball_y"] = by.astype(np.float32)
    table["ball_speed"] = np.hypot(bvx, bvy)
    table["ball_touchline_m"] = np.minimum(by, PITCH_WIDTH - by).astype(np.float32)
    table["ball_goalline_m"] = np.minimum(bx, PITCH_LENGTH - bx).astype(np.float32)

    is_home = team == atd.HOME
    is_away = team == atd.AWAY

    # Averages over frames where nobody is visible give NaN and a warning. NaN is the right
    # answer there, so the warning is hidden.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        table["all_mean_speed"] = np.nanmean(speed, axis=1)

        distance, relative_speed = nearest_to_ball(px, py, pvx, pvy, is_home, bx, by, bvx, bvy)
        table["near_home_dist"] = distance
        table["near_home_rel_speed"] = relative_speed
        table.update(team_shape("home", is_home, px, py, speed, home_sign))

        distance, relative_speed = nearest_to_ball(px, py, pvx, pvy, is_away, bx, by, bvx, bvy)
        table["near_away_dist"] = distance
        table["near_away_rel_speed"] = relative_speed
        table.update(team_shape("away", is_away, px, py, speed, away_sign))

    table.update(players_still_at_touchlines(py, speed))

    hints = tagging.frame_hints(tracking["names"], n_frames, game, data_dir)
    table["tag_home_votes"] = hints["home_votes"]
    table["tag_away_votes"] = hints["away_votes"]
    table["tag_set_piece_age_s"] = hints["set_piece_age_s"]
    return table


def load(game=GAME, data_dir=None, refresh=False):
    """The measurements table: from the saved copy if it is up to date, otherwise built again."""
    path = cache_path(game, data_dir)
    raw_path = atd.require(["atd_raw"], game, data_dir)[0]
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path)),
                      FEATURE_VERSION], dtype=np.int64)

    if not refresh and os.path.isfile(path):
        with np.load(path) as stored:
            if "_stamp" in stored and np.array_equal(stored["_stamp"], stamp):
                table = {}
                for key in stored.files:
                    if key != "_stamp":
                        table[key] = stored[key]
                return table

    table = build(game, data_dir, refresh)
    np.savez_compressed(path, _stamp=stamp, **table)
    return table


def summary(table):
    """Print a few checks of the table. No answers are used."""
    graded = table["half"] > 0
    n_graded = int(graded.sum())
    usable = graded & ~np.isnan(table["ball_x"])

    print("{} frames, {} graded, {} columns".format(len(graded), n_graded, len(table)))
    print("ball reported by the tracker: {:.1f}% of graded frames".format(
        100.0 * table["ball_seen"][graded].mean()))
    print("ball usable after cleaning:   {:.1f}% of graded frames".format(
        100.0 * usable.sum() / n_graded))
    print("frames thrown away as glitches: {}, short gaps filled: {}, too fast: {}".format(
        int(table["ball_removed"].sum()), int(table["ball_bridged"].sum()),
        int(table["ball_too_fast"].sum())))

    # At each kick-off every player is in their own half, so both teams' average height must be
    # under half the pitch. If not, the attacking direction is the wrong way round.
    print("attack direction check at the two kick-offs:")
    for frame in (11185, 113742):
        home_height = table["home_centroid_height"][frame]
        away_height = table["away_centroid_height"][frame]
        if home_height < PITCH_LENGTH / 2 and away_height < PITCH_LENGTH / 2:
            result = "OK"
        else:
            result = "WRONG WAY ROUND?"
        print("  frame {}: home {:.1f} m, away {:.1f} m from own goal  {}".format(
            frame, home_height, away_height, result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="rebuild even if saved")
    args = parser.parse_args()
    summary(load(refresh=args.refresh))
