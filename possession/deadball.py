"""Find the stoppages: every stretch of a graded half where the ball is out of play.

The tracking has no "ball dead" flag, so a stoppage is recognised by what it does to the play.
First the CORE -- frames that can only be a stoppage -- found five independent ways:

  A. whistle   players drop to walking pace AND nobody plays the ball (unseen, held or placed).
               Either alone is not enough: slow build-up has walking players and a moving ball,
               and a goalkeeper holding the ball has a still ball and players moving into shape.
  B. out       the ball is seen clearly over a line.
  C. placed    the ball sits still on a goal area or a corner arc: a goal kick or a corner waiting
               to be taken. Players often jog back meanwhile, so A misses these.
  D. thrower   a player stands still at a touchline, the ball was last seen near that same
               touchline, and nobody plays the ball: a throw-in waiting to be taken.
  E. kick-off  every player in their own half -- the formation alone, because a spare ball being
               knocked about before a half starts would otherwise hide it. The kick-off is the
               moment the formation breaks.

Then each core is widened to the whole stoppage:

  start   players only react a second or two after the ball goes out or the whistle goes. If the
          ball was last seen near a line just before the core, it went out there and the stoppage
          starts right after that sighting; otherwise just after the ball was last played, but
          never more than LOOKBACK_S before the core.
  stop    the restart: the first time the ball is played after the core.

A half is dead from its first graded frame until its first kick-off, which is what the second half
needs: it is graded from 110808 but kicks off minutes later.

Every number below comes from the laws of the game, football practice or physics, and was fixed
without looking at any label of this match.

    .venv/bin/python -m possession.deadball       # detect, then print what was found
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

# Params
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]

# A -- average human walking pace. Below it, averaged over the visible players, the team walks.
QUIET_SPEED = 1.3
# Team speed is averaged over +/-1.5 s so one sprinting player cannot break a stoppage.
QUIET_HALF_WINDOW = int(1.5 * FPS)
# The ball counts as "being played" above a slow dribble. A ball carried or rolled back to the
# taker moves slower than this; a pass or a dribble at pace moves faster.
BALL_PLAY_SPEED = 4.0
# In open play the ball is played at least every few seconds, so +/-2 s with no play is a pause.
IDLE_HALF_WINDOW = 2 * FPS

# B -- the ball is out only when all of it is over the line. Half a metre clears the ball's own
# size and typical tracking error.
OUT_MARGIN_M = 0.5

# C -- a ball this slow is lying on the grass. Goal area and corner arc from the laws of the game,
# plus a metre for tracking error and for a ball placed on the edge.
PLACED_SPEED = 0.5
PLACED_S = 1.5
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
CORNER_ARC_M = 1.0
SPOT_MARGIN_M = 1.0

# D -- the thrower has to stand at the line for a moment before the throw.
THROWER_S = 1.5

# E -- kick-off. At least this many players tracked (fewer means the teams are still walking out),
# and at most this many past the halfway line in total: the takers stand at the ball on the line
# and tracking puts a player or two per side a metre over it. In open play several attackers are
# always in the other half, so this stays far from anything but a kick-off.
KICKOFF_MIN_PLAYERS = 16
KICKOFF_PAST_HALFWAY = 3
KICKOFF_FORMATION_S = 2.0
# The opening kick-off of a half is looked for this long after the half's first graded frame.
KICKOFF_SEARCH_S = 240.0
# Teams line up over a minute or two and the formation forms and breaks as players take position,
# so formations this close together belong to the same opening. A real kick-off is followed by far
# more open play than this before any goal could bring the next one.
OPENING_CHAIN_GAP_S = 30.0

# Widening a core into a stoppage.
LINE_NEAR_M = 2.0
OUT_LOOKBACK_S = 6.0
LOOKBACK_S = 3.0
RESTART_WAIT_S = 3.0

# The quickest restarts (a quick throw-in) take about three seconds.
MIN_DEAD_S = 3.0
# A restart followed by a new stoppage within a few seconds is rare; a stoppage briefly "broken"
# by the ball being kicked back to the taker is common. Closer than this, they are one stoppage.
MERGE_GAP_S = 4.0

PATHS = ("whistle", "out", "placed", "thrower", "kickoff")


def sustained(mask, min_seconds):
    """mask with every True run shorter than min_seconds removed."""
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in zip(*runs(mask)):
        if stop - start >= min_seconds * FPS:
            out[start:stop] = True
    return out


def last_seen(values):
    """(last known value, frames since it was seen) at every frame, by forward fill."""
    known = ~np.isnan(values)
    index = np.where(known, np.arange(len(values)), -1)
    np.maximum.accumulate(index, out=index)
    filled = np.where(index >= 0, values[np.maximum(index, 0)], np.nan)
    age = np.where(index >= 0, np.arange(len(values)) - index, np.iinfo(np.int32).max)
    return filled, age


def core_paths(table):
    """{path: per-frame bool} for the five ways a frame can be recognised as a stoppage."""
    x, y, speed = table["ball_x"], table["ball_y"], table["ball_speed"]
    with np.errstate(invalid="ignore"):
        played = speed > BALL_PLAY_SPEED
        quiet = rolling_mean(table["all_mean_speed"], QUIET_HALF_WINDOW) < QUIET_SPEED
        idle = ~rolling_any(played, IDLE_HALF_WINDOW)

        out = (table["ball_touchline_m"] < -OUT_MARGIN_M) | (table["ball_goalline_m"] < -OUT_MARGIN_M)

        goal_area = ((np.minimum(x, PITCH_LENGTH - x) <= GOAL_AREA_DEPTH_M + SPOT_MARGIN_M)
                     & (np.abs(y - PITCH_WIDTH / 2) <= GOAL_AREA_HALF_WIDTH_M + SPOT_MARGIN_M))
        corner = (np.hypot(np.minimum(x, PITCH_LENGTH - x), np.minimum(y, PITCH_WIDTH - y))
                  <= CORNER_ARC_M + SPOT_MARGIN_M)
        placed = sustained((speed < PLACED_SPEED) & (goal_area | corner), PLACED_S)

        last_y, age = last_seen(y)
        recent = age <= OUT_LOOKBACK_S * FPS
        thrower = sustained(idle & recent & (
            ((table["line_touch_low_still"] > 0) & (last_y < LINE_NEAR_M))
            | ((table["line_touch_high_still"] > 0) & (last_y > PITCH_WIDTH - LINE_NEAR_M))),
            THROWER_S)

        kickoff = sustained(
            (table["home_n_players"] + table["away_n_players"] >= KICKOFF_MIN_PLAYERS)
            & (table["home_n_past_halfway"] + table["away_n_past_halfway"]
               <= KICKOFF_PAST_HALFWAY),
            KICKOFF_FORMATION_S)

    graded = table["half"] > 0
    return {"whistle": quiet & idle & graded, "out": out & graded, "placed": placed & graded,
            "thrower": thrower & graded, "kickoff": kickoff & graded}


def last_before(indices, frame):
    """Largest value in sorted indices that is < frame, or None."""
    position = np.searchsorted(indices, frame) - 1
    return int(indices[position]) if position >= 0 else None


def first_at_or_after(indices, frame):
    """Smallest value in sorted indices that is >= frame, or None."""
    position = np.searchsorted(indices, frame)
    return int(indices[position]) if position < len(indices) else None


def detect(table, game=GAME, paths=None):
    """[(start, stop)] half-open stoppages, in frame order, never crossing a half boundary.

    paths is core_paths(table), passed in when the caller already has it.
    """
    paths = paths if paths is not None else core_paths(table)
    core = np.zeros(len(table["half"]), dtype=bool)
    for mask in paths.values():
        core |= mask

    with np.errstate(invalid="ignore"):
        played = np.flatnonzero(table["ball_speed"] > BALL_PLAY_SPEED)
        near_line = np.minimum(table["ball_touchline_m"], table["ball_goalline_m"]) < LINE_NEAR_M
    seen = np.flatnonzero(~np.isnan(table["ball_x"]))

    def restart_after(frame):
        restart = first_at_or_after(played, frame)
        limit = frame + IDLE_HALF_WINDOW + int(RESTART_WAIT_S * FPS)
        return restart if restart is not None and restart <= limit else frame

    stoppages = []
    for _, first, last in features.half_video_blocks(game):
        half_start, half_stop = first, last + 1
        starts, stops = runs(core[half_start:half_stop])
        for core_start, core_stop in zip(starts + half_start, stops + half_start):
            sighting = last_before(seen, core_start)
            if (sighting is not None and near_line[sighting]
                    and core_start - sighting <= OUT_LOOKBACK_S * FPS):
                start = sighting + 1
            else:
                last_play = last_before(played, core_start)
                start = core_start - int(LOOKBACK_S * FPS)
                if last_play is not None:
                    start = max(start, last_play + 1)
            # A kick-off restarts when its formation breaks, not at the next fast ball, which
            # before a half can be a spare ball knocked about elsewhere on the pitch.
            kicked_off = paths["kickoff"][core_stop - 1]
            stop = core_stop if kicked_off else restart_after(core_stop)
            stoppages.append((max(half_start, start), min(half_stop, stop)))

        # Dead from the first graded frame until the opening kick-off, if one is found in time:
        # the end of the last formation in the opening chain.
        opening = None
        for kick_start, kick_stop in zip(*runs(paths["kickoff"][half_start:half_stop])):
            if opening is None and kick_start > KICKOFF_SEARCH_S * FPS:
                break
            if opening is not None and kick_start - opening > OPENING_CHAIN_GAP_S * FPS:
                break
            opening = int(kick_stop)
        if opening is not None:
            stoppages.append((half_start, half_start + opening))

    return tidy(stoppages)


def tidy(stoppages):
    """Merge overlapping or nearly touching stoppages, then drop the ones too short to be real."""
    merged = []
    for start, stop in sorted(stoppages):
        if merged and start - merged[-1][1] < MERGE_GAP_S * FPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return [(start, stop) for start, stop in merged if stop - start >= MIN_DEAD_S * FPS]


def dead_mask(stoppages, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in stoppages:
        mask[start:stop] = True
    return mask


def report(table, stoppages, paths):
    """What was found, judged without any ground truth: counts, share and agreement with tags."""
    graded = table["half"] > 0
    dead = dead_mask(stoppages, len(graded))
    seconds = np.array([stop - start for start, stop in stoppages]) / float(FPS)
    print("{} stoppages, {:.1f}% of graded time dead ({:.1f} of {:.1f} minutes)".format(
        len(stoppages), 100.0 * dead[graded].mean(), dead[graded].sum() / FPS / 60.0,
        graded.sum() / FPS / 60.0))
    print("duration s  p10 {:.1f}  p50 {:.1f}  p90 {:.1f}  max {:.1f}".format(
        *np.percentile(seconds, [10, 50, 90, 100])))
    print("stoppages each path contributed to:  {}".format("  ".join(
        "{} {}".format(name, sum(paths[name][a:b].any() for a, b in stoppages))
        for name in PATHS)))
    for _, first, _ in features.half_video_blocks(GAME):
        opening = [s for s in stoppages if s[0] == first]
        print("  half starting {}: {}".format(
            first, "dead until {}".format(opening[0][1]) if opening else "no opening stoppage"))

    # Smart Tagging is independent evidence: every tagged set piece should sit on a detected
    # stoppage, with the restart a few seconds after the tag window opens.
    age = table["tag_set_piece_age_s"]
    opens = np.flatnonzero((age == 0) & graded)
    hit, offsets = 0, []
    for frame in opens:
        overlapping = [s for s in stoppages if s[0] <= frame + 10 * FPS and s[1] >= frame]
        if overlapping:
            hit += 1
            offsets.append((overlapping[0][1] - frame) / float(FPS))
    print("tagged set pieces with a detected stoppage in their first 10 s: {} of {}".format(
        hit, len(opens)))
    if offsets:
        print("  restart this long after the tag opens, s: p10 {:.1f}  p50 {:.1f}  p90 {:.1f}"
              .format(*np.percentile(offsets, [10, 50, 90])))


if __name__ == '__main__':
    table = features.load()
    paths = core_paths(table)
    report(table, detect(table, paths=paths), paths)
