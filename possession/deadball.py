"""Step 2: find the stoppages (the times the ball is out of play).

The tracking has no "ball out of play" field, so we look for signs of a stoppage.

First we mark frames that can only be a stoppage. There are five signs:

    whistle   players walk (average speed below 1.3 m/s) AND nobody plays the ball
    out       the ball is seen over a line
    placed    the ball lies still on a goal area or corner arc (goal kick, corner)
    thrower   a player stands still at the touchline where the ball was last seen (throw-in)
    kickoff   (almost) every player is in their own half

Then each group of marked frames becomes a full stoppage:

    start   Players slow down a few seconds after the ball goes out. So if the ball was last seen
            near a line shortly before, the stoppage starts right after that. Otherwise it starts
            just after the ball was last played, but at most 3 seconds earlier.
    end     The first time the ball is played again (the restart). For a kick-off: the moment
            the players leave the kick-off formation.

Each half is also dead from its first graded frame until its first kick-off.

    .venv/bin/python -m possession.deadball       # find the stoppages and print a summary
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from possession import features  # noqa: E402
from possession.series import rolling_any, rolling_mean, runs  # noqa: E402

# Params -- chosen from football rules, football practice and physics.
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]

# whistle: walking pace, averaged over 1.5 s either side.
QUIET_SPEED = 1.3
QUIET_HALF_WINDOW = int(1.5 * FPS)
# The ball is "being played" when it moves faster than 4 m/s.
BALL_PLAY_SPEED = 4.0
# "Nobody plays the ball" = no fast ball within 2 s either side.
IDLE_HALF_WINDOW = 2 * FPS

# out: the whole ball must be over the line, so at least 0.5 m past it.
OUT_MARGIN_M = 0.5

# placed: still ball (below 0.5 m/s) for 1.5 s. Sizes from the laws of the game, plus 1 m.
PLACED_SPEED = 0.5
PLACED_S = 1.5
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
CORNER_ARC_M = 1.0
SPOT_MARGIN_M = 1.0

# thrower: standing at the line for 1.5 s.
THROWER_S = 1.5

# kickoff: at least 16 players visible, at most 3 in the other half, for 2 s.
KICKOFF_MIN_PLAYERS = 16
KICKOFF_PAST_HALFWAY = 3
KICKOFF_FORMATION_S = 2.0
# The first kick-off of a half must appear within 4 minutes of the half's first frame.
KICKOFF_SEARCH_S = 240.0
# Before a kick-off the formation can break and form again. Breaks under 30 s are joined.
OPENING_CHAIN_GAP_S = 30.0

# Start and end of a stoppage.
LINE_NEAR_M = 2.0        # "near a line"
OUT_LOOKBACK_S = 6.0     # how far back to look for the ball near a line
LOOKBACK_S = 3.0         # otherwise, at most this far back
RESTART_WAIT_S = 3.0     # how long to wait for the restart kick

# Clean-up: stoppages under 3 s are dropped, stoppages under 4 s apart are joined.
MIN_DEAD_S = 3.0
MERGE_GAP_S = 4.0

SIGNS = ("whistle", "out", "placed", "thrower", "kickoff")


def keep_long_runs(mask, min_seconds):
    """Copy of mask that keeps only the stretches of True lasting at least min_seconds."""
    result = np.zeros(len(mask), dtype=bool)
    starts, stops = runs(mask)
    for i in range(len(starts)):
        if stops[i] - starts[i] >= min_seconds * FPS:
            result[starts[i]:stops[i]] = True
    return result


def last_seen(values):
    """For every frame: the most recent known value, and how many frames ago it was seen.

    Before the first known value: NaN, and a very large number of frames.
    """
    n = len(values)
    last_value = np.full(n, np.nan, dtype=values.dtype)
    frames_ago = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    known = ~np.isnan(values)

    latest_frame = -1
    for frame in range(n):
        if known[frame]:
            latest_frame = frame
        if latest_frame >= 0:
            last_value[frame] = values[latest_frame]
            frames_ago[frame] = frame - latest_frame
    return last_value, frames_ago


def whistle_sign(table):
    with np.errstate(invalid="ignore"):
        team_speed = rolling_mean(table["all_mean_speed"], QUIET_HALF_WINDOW)
        players_walk = team_speed < QUIET_SPEED
        ball_played = table["ball_speed"] > BALL_PLAY_SPEED
    nobody_plays_ball = ~rolling_any(ball_played, IDLE_HALF_WINDOW)
    return players_walk & nobody_plays_ball, nobody_plays_ball


def out_sign(table):
    with np.errstate(invalid="ignore"):
        over_touchline = table["ball_touchline_m"] < -OUT_MARGIN_M
        over_goal_line = table["ball_goalline_m"] < -OUT_MARGIN_M
    return over_touchline | over_goal_line


def placed_sign(table):
    x = table["ball_x"]
    y = table["ball_y"]
    with np.errstate(invalid="ignore"):
        distance_to_goal_line = np.minimum(x, PITCH_LENGTH - x)
        distance_to_touchline = np.minimum(y, PITCH_WIDTH - y)

        in_goal_area = ((distance_to_goal_line <= GOAL_AREA_DEPTH_M + SPOT_MARGIN_M)
                        & (np.abs(y - PITCH_WIDTH / 2) <= GOAL_AREA_HALF_WIDTH_M + SPOT_MARGIN_M))
        distance_to_corner = np.hypot(distance_to_goal_line, distance_to_touchline)
        in_corner_arc = distance_to_corner <= CORNER_ARC_M + SPOT_MARGIN_M

        ball_still = table["ball_speed"] < PLACED_SPEED
    return keep_long_runs(ball_still & (in_goal_area | in_corner_arc), PLACED_S)


def thrower_sign(table, nobody_plays_ball):
    last_y, frames_ago = last_seen(table["ball_y"])
    with np.errstate(invalid="ignore"):
        seen_recently = frames_ago <= OUT_LOOKBACK_S * FPS
        # y = 0 is one touchline ("low"), y = 68 the other ("high").
        thrower_low = (table["line_touch_low_still"] > 0) & (last_y < LINE_NEAR_M)
        thrower_high = (table["line_touch_high_still"] > 0) & (last_y > PITCH_WIDTH - LINE_NEAR_M)
    candidate = nobody_plays_ball & seen_recently & (thrower_low | thrower_high)
    return keep_long_runs(candidate, THROWER_S)


def kickoff_sign(table):
    players_visible = table["home_n_players"] + table["away_n_players"]
    players_past_halfway = table["home_n_past_halfway"] + table["away_n_past_halfway"]
    formation = ((players_visible >= KICKOFF_MIN_PLAYERS)
                 & (players_past_halfway <= KICKOFF_PAST_HALFWAY))
    return keep_long_runs(formation, KICKOFF_FORMATION_S)


def core_paths(table):
    """For each of the five signs, a mask of the frames that show it (graded frames only)."""
    graded = table["half"] > 0
    whistle, nobody_plays_ball = whistle_sign(table)
    signs = {}
    signs["whistle"] = whistle & graded
    signs["out"] = out_sign(table) & graded
    signs["placed"] = placed_sign(table) & graded
    signs["thrower"] = thrower_sign(table, nobody_plays_ball) & graded
    signs["kickoff"] = kickoff_sign(table) & graded
    return signs


def last_before(frames, frame):
    """The largest number in the sorted list frames that is below frame, or None."""
    position = np.searchsorted(frames, frame) - 1
    if position < 0:
        return None
    return int(frames[position])


def first_at_or_after(frames, frame):
    """The smallest number in the sorted list frames that is frame or above, or None."""
    position = np.searchsorted(frames, frame)
    if position >= len(frames):
        return None
    return int(frames[position])


def detect(table, game=GAME, paths=None):
    """All stoppages, as a list of (first frame, frame just after the end), in time order."""
    if paths is None:
        paths = core_paths(table)

    # A frame is a "core" stoppage frame if any sign marks it.
    core = np.zeros(len(table["half"]), dtype=bool)
    for sign in SIGNS:
        core = core | paths[sign]

    # Frames where the ball is being played, and frames where the ball is seen at all.
    with np.errstate(invalid="ignore"):
        played_frames = np.flatnonzero(table["ball_speed"] > BALL_PLAY_SPEED)
        distance_to_line = np.minimum(table["ball_touchline_m"], table["ball_goalline_m"])
        near_line = distance_to_line < LINE_NEAR_M
    seen_frames = np.flatnonzero(~np.isnan(table["ball_x"]))

    stoppages = []
    for half_number, first_frame, last_frame in features.half_video_blocks(game):
        half_start = first_frame
        half_stop = last_frame + 1

        core_starts, core_stops = runs(core[half_start:half_stop])
        for i in range(len(core_starts)):
            core_start = core_starts[i] + half_start
            core_stop = core_stops[i] + half_start

            start = stoppage_start(core_start, seen_frames, played_frames, near_line)

            if paths["kickoff"][core_stop - 1]:
                # A kick-off restarts when the formation breaks.
                stop = core_stop
            else:
                stop = restart_after(core_stop, played_frames)

            stoppages.append((max(half_start, start), min(half_stop, stop)))

        opening_end = opening_kickoff_end(paths["kickoff"][half_start:half_stop])
        if opening_end is not None:
            stoppages.append((half_start, half_start + opening_end))

    return tidy(stoppages)


def stoppage_start(core_start, seen_frames, played_frames, near_line):
    """Go back from the first core frame to when the stoppage really began."""
    last_sighting = last_before(seen_frames, core_start)
    if last_sighting is not None:
        seen_near_line = near_line[last_sighting]
        seen_recently = core_start - last_sighting <= OUT_LOOKBACK_S * FPS
        if seen_near_line and seen_recently:
            # The ball went out where it was last seen.
            return last_sighting + 1

    # Otherwise: just after the ball was last played, but at most LOOKBACK_S back.
    start = core_start - int(LOOKBACK_S * FPS)
    last_play = last_before(played_frames, core_start)
    if last_play is not None:
        start = max(start, last_play + 1)
    return start


def restart_after(frame, played_frames):
    """The first frame the ball is played at or after frame, if it comes soon enough."""
    restart = first_at_or_after(played_frames, frame)
    latest_allowed = frame + IDLE_HALF_WINDOW + int(RESTART_WAIT_S * FPS)
    if restart is not None and restart <= latest_allowed:
        return restart
    return frame


def opening_kickoff_end(kickoff_in_half):
    """Frame (counted from the half's start) where the half's first kick-off happens, or None.

    Before a kick-off the formation can appear in several pieces. Pieces less than
    OPENING_CHAIN_GAP_S apart belong together; the kick-off is the end of the last piece.
    """
    opening_end = None
    starts, stops = runs(kickoff_in_half)
    for i in range(len(starts)):
        if opening_end is None and starts[i] > KICKOFF_SEARCH_S * FPS:
            break  # the first formation comes too late to be the opening kick-off
        if opening_end is not None and starts[i] - opening_end > OPENING_CHAIN_GAP_S * FPS:
            break  # too long after the previous piece: a different kick-off
        opening_end = int(stops[i])
    return opening_end


def tidy(stoppages):
    """Join stoppages that overlap or are less than MERGE_GAP_S apart, then drop short ones."""
    joined = []
    for start, stop in sorted(stoppages):
        if len(joined) > 0 and start - joined[-1][1] < MERGE_GAP_S * FPS:
            previous_start, previous_stop = joined[-1]
            joined[-1] = (previous_start, max(previous_stop, stop))
        else:
            joined.append((start, stop))

    long_enough = []
    for start, stop in joined:
        if stop - start >= MIN_DEAD_S * FPS:
            long_enough.append((start, stop))
    return long_enough


def dead_mask(stoppages, n_frames):
    """A mask that is True on every stoppage frame."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in stoppages:
        mask[start:stop] = True
    return mask


def report(table, stoppages, paths):
    """Print a summary of the stoppages. No answers are used."""
    graded = table["half"] > 0
    dead = dead_mask(stoppages, len(graded))
    dead_minutes = dead[graded].sum() / FPS / 60.0
    graded_minutes = graded.sum() / FPS / 60.0
    print("{} stoppages, {:.1f}% of graded time dead ({:.1f} of {:.1f} minutes)".format(
        len(stoppages), 100.0 * dead[graded].mean(), dead_minutes, graded_minutes))

    for sign in SIGNS:
        count = 0
        for start, stop in stoppages:
            if paths[sign][start:stop].any():
                count += 1
        print("  {:<8} seen in {} stoppages".format(sign, count))

    # Check against Smart Tagging: each tagged set piece should have a stoppage near its start.
    tag_starts = np.flatnonzero((table["tag_set_piece_age_s"] == 0) & graded)
    matched = 0
    for frame in tag_starts:
        for start, stop in stoppages:
            if start <= frame + 10 * FPS and stop >= frame:
                matched += 1
                break
    print("tagged set pieces with a detected stoppage in their first 10 s: {} of {}".format(
        matched, len(tag_starts)))


if __name__ == '__main__':
    table = features.load()
    paths = core_paths(table)
    report(table, detect(table, paths=paths), paths)
