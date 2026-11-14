"""Step 3: decide which team has the ball between stoppages.

The rules:
    - After a stoppage, the team that restarts has the ball.
    - Possession changes only when the OTHER team gets CONTROL of the ball.
    - While the ball is loose, contested or not seen, the last team keeps it.

Control: a player is within 1.5 m of the ball, moves with it (velocity difference at most
3 m/s), and no opponent is within 1.5 m. An unbroken run of such frames is an "event":
    control   0.4 s or longer: enough to take the ball
    touch     0.12 s to 0.4 s: one touch by the other team changes nothing (a deflection),
              two touches in a row by the other team change possession

When both teams are within 1.5 m (a duel), there is no event, so nothing is decided until
someone comes out with the ball.

When the ball is not seen, there are no events. If the same team has the ball before and after
the gap, it keeps the whole gap. If not, the change is placed where the team shapes change.

    .venv/bin/python -m possession.tracker      # label the match and print a summary
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from possession import deadball, features  # noqa: E402
from possession.series import rolling_mean, runs  # noqa: E402

# Params -- chosen from football and the tracking's accuracy.
FPS = GAMES[GAME]["fps"]
CONTROL_RADIUS_M = 1.5      # ball at a player's feet is within about 1.5 m of the player
CONTROL_REL_SPEED = 3.0     # a dribbled ball runs a little ahead and behind the player
TOUCH_S = 0.12              # 3 frames: the shortest contact we trust
CONTROL_S = 0.4             # 10 frames: real control
TWO_TOUCH_GAP_S = 3.0       # two touches further apart than this are not one move
GAP_PLACE_S = 2.0           # place a change by team shape if the ball was unseen this long
SHAPE_HALF_WINDOW = FPS     # team shapes are averaged over 1 s either side

HOME, AWAY, DEAD, NO_ANSWER = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])  # the letter for each of the numbers above


def control_side(table):
    """For every frame: HOME or AWAY if that team controls the ball, otherwise -1."""
    with np.errstate(invalid="ignore"):
        home_close = table["near_home_dist"] <= CONTROL_RADIUS_M
        away_close = table["near_away_dist"] <= CONTROL_RADIUS_M
        home_moves_with_ball = table["near_home_rel_speed"] <= CONTROL_REL_SPEED
        away_moves_with_ball = table["near_away_rel_speed"] <= CONTROL_REL_SPEED

    home_controls = home_close & home_moves_with_ball & ~away_close
    away_controls = away_close & away_moves_with_ball & ~home_close

    side = np.full(len(home_close), -1, dtype=np.int8)
    side[home_controls] = HOME
    side[away_controls] = AWAY
    return side


def control_events(side):
    """Turn the per-frame control into a list of events, sorted by time.

    Each event is (start frame, stop frame, team, is_control). is_control is True for a
    control (0.4 s or more) and False for a touch.
    """
    events = []
    for team in (HOME, AWAY):
        starts, stops = runs(side == team)
        for i in range(len(starts)):
            length = stops[i] - starts[i]
            if length >= TOUCH_S * FPS:
                is_control = length >= CONTROL_S * FPS
                events.append((int(starts[i]), int(stops[i]), team, is_control))
    return sorted(events)


def standardise(values):
    """Divide values by how much they vary over the match, so different signals can be added.

    Unknown values (NaN, or infinite from the log of zero) become 0.
    """
    values = np.where(np.isfinite(values), values, np.nan)
    scale = np.nanstd(values)
    if not scale:
        return np.zeros(len(values), dtype=np.float32)
    scaled = np.where(np.isnan(values), 0.0, values / scale)
    return scaled.astype(np.float32)


def shape_score(table):
    """Three signals for every frame. Positive means "looks like the home team has the ball".

        width    the team with the ball spreads wider across the pitch
        effort   the team without the ball runs more (it is chasing)
        tags     Smart Tagging votes

    We checked these against our own ball-based answers where the ball is seen. Width and effort
    agreed much more often than the direction the teams move, which back-passes confuse.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        width = np.log(table["home_spread_width"]) - np.log(table["away_spread_width"])
        effort = table["away_mean_speed"] - table["home_mean_speed"]
    # A moment of play usually gets two tags (one per team), so the votes are halved.
    votes = (table["tag_home_votes"].astype(np.float32) - table["tag_away_votes"]) / 2.0
    return {"width": standardise(width), "effort": standardise(effort), "tags": votes}


def combined_score(parts):
    """The three signals added together, averaged over 1 s either side."""
    total = parts["width"] + parts["effort"] + parts["tags"]
    return rolling_mean(total, SHAPE_HALF_WINDOW)


def place_change(score, start, stop, new_team):
    """Find the best frame in [start, stop) for possession to change to new_team.

    The best frame splits the window so that, before it, the score points at the old team and,
    after it, at the new team. With a running total of the score, that is where the running
    total is lowest (change to home) or highest (change to away).
    """
    window = np.nan_to_num(score[start:stop])
    if len(window) == 0:
        return start
    running_total = np.concatenate(([0.0], np.cumsum(window)))
    if new_team == HOME:
        best = np.argmin(running_total)
    else:
        best = np.argmax(running_total)
    return start + int(best)


def label_segment(events, start, stop, score, seen):
    """Decide possession for one stretch of live play, from start to stop.

    Returns the possession changes as a list of (frame, team). The first is at start.
    """
    events_here = []
    for event in events:
        if start <= event[0] < stop:
            events_here.append(event)

    if len(events_here) == 0:
        # Nobody controlled the ball at all: use the team shapes.
        if np.nansum(score[start:stop]) >= 0:
            return [(start, HOME)]
        return [(start, AWAY)]

    # The first team with an event restarted the play.
    team_with_ball = events_here[0][2]
    changes = [(start, team_with_ball)]
    waiting_touch = None              # one touch by the other team, waiting for a second
    last_own_event_end = events_here[0][1]

    for event_start, event_end, team, is_control in events_here[1:]:
        if team == team_with_ball:
            # The team with the ball touched it again: forget the other team's touch.
            waiting_touch = None
            last_own_event_end = event_end
            continue

        # The event is by the other team.
        if is_control:
            if waiting_touch is not None:
                change_frame = waiting_touch[0]
            else:
                change_frame = event_start
        elif waiting_touch is not None and event_start - waiting_touch[1] <= TWO_TOUCH_GAP_S * FPS:
            change_frame = waiting_touch[0]   # second touch in a row
        else:
            waiting_touch = (event_start, event_end)   # first touch: wait
            continue

        # If the ball was unseen for a while before the change, place it by team shape.
        unseen_frames = np.sum(~seen[last_own_event_end:change_frame])
        if unseen_frames >= GAP_PLACE_S * FPS:
            change_frame = place_change(score, last_own_event_end, change_frame, team)

        changes.append((change_frame, team))
        team_with_ball = team
        waiting_touch = None
        last_own_event_end = event_end

    return changes


def live_stretches(half, dead):
    """List of (start, stop) for each stretch of live play inside the graded halves."""
    starts, stops = runs((half > 0) & ~dead)
    stretches = []
    for i in range(len(starts)):
        stretches.append((int(starts[i]), int(stops[i])))
    return stretches


def label(table, stoppages):
    """The answer for every frame of the video, as numbers (see LABELS).

    D on stoppage frames, H or A on live frames, X outside the graded halves.
    """
    n_frames = len(table["half"])
    dead = deadball.dead_mask(stoppages, n_frames)
    codes = np.full(n_frames, NO_ANSWER, dtype=np.int8)
    codes[(table["half"] > 0) & dead] = DEAD

    events = control_events(control_side(table))
    score = combined_score(shape_score(table))
    seen = ~np.isnan(table["ball_x"])

    for start, stop in live_stretches(table["half"], dead):
        changes = label_segment(events, start, stop, score, seen)
        for i in range(len(changes)):
            change_frame, team = changes[i]
            if i + 1 < len(changes):
                next_change_frame = changes[i + 1][0]
            else:
                next_change_frame = stop
            codes[change_frame:next_change_frame] = team
    return codes


def shape_agreement(table, codes):
    """Print how often each team-shape signal agrees with our ball-based answer.

    Only frames where the ball is seen are used. No real answers are involved.
    """
    alive = (codes == HOME) | (codes == AWAY)
    ball_seen = alive & ~np.isnan(table["ball_x"])
    parts = shape_score(table)
    parts["combined"] = combined_score(parts)
    for name in parts:
        smoothed = rolling_mean(parts[name], SHAPE_HALF_WINDOW)
        has_vote = ball_seen & (smoothed != 0) & ~np.isnan(smoothed)
        agree = np.mean((smoothed[has_vote] > 0) == (codes[has_vote] == HOME))
        print("  {:<9} agrees on {:.1f}% of {:.0f} minutes".format(
            name, 100.0 * agree, has_vote.sum() / FPS / 60.0))


def describe(codes, table):
    """Print what our answer says about the match. No real answers are involved."""
    graded = table["half"] > 0
    possession_seconds = []
    for half_number, first_frame, last_frame in features.half_video_blocks(GAME):
        half_codes = codes[first_frame:last_frame + 1]
        for team in (HOME, AWAY):
            starts, stops = runs(half_codes == team)
            for i in range(len(starts)):
                possession_seconds.append((stops[i] - starts[i]) / float(FPS))

    alive = (codes == HOME) | (codes == AWAY)
    home_share = 100.0 * np.sum(codes == HOME) / alive.sum()
    dead_share = 100.0 * np.mean(codes[graded] == DEAD)
    print("{} possessions, mean {:.1f} s, median {:.1f} s; home share {:.1f}%; dead {:.1f}%".format(
        len(possession_seconds), np.mean(possession_seconds), np.median(possession_seconds),
        home_share, dead_share))


if __name__ == '__main__':
    table = features.load()
    codes = label(table, deadball.detect(table))
    describe(codes, table)
    print("\nteam-shape signals vs our ball-based answer, where the ball is seen:")
    shape_agreement(table, codes)
