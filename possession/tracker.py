"""Who has the ball between stoppages.

The rule the challenge states, applied as literally as the tracking allows:

  * after every stoppage, the team that restarts has the ball;
  * possession changes only when the OTHER team gains CONTROL of it;
  * while the ball is loose, contested or out of sight, the last team keeps it.

Control is the one thing the tracking can almost see directly: a player within CONTROL_RADIUS_M of
the ball, moving with it (their velocity within CONTROL_REL_SPEED of the ball's), and no opponent
that close. A run of such frames is a CONTROL EVENT for that team. With no ball height in the data,
"moving with it" is also what separates a ball at someone's feet from one flying over their head.

  strong event   a run of at least CONTROL_S. Enough on its own to take the ball.
  weak event     a shorter run, at least TOUCH_S. A touch. One touch by the other team is a
                 deflection or a block and changes nothing; two in a row by the same team, with no
                 touch by the team in possession between them, are control.

A duel -- both teams within CONTROL_RADIUS_M -- produces no event at all, so it decides nothing by
itself: the tracker waits and sees who comes out of it with the ball, and dates the change to
their first touch.

The ball is missing on long stretches of open play, where no events exist. If the same team has
it either side of a gap, the gap is theirs. If the team changed, the change happened somewhere in
the gap, and it is placed where the team shapes say it did (see shape_score).

    .venv/bin/python -m possession.tracker      # label the match, print what it claims
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from possession import atd, deadball, features  # noqa: E402
from possession.series import rolling_mean, runs  # noqa: E402

# Params -- from football and the tracking's known precision, not fitted to this match's labels.
FPS = GAMES[GAME]["fps"]
# Ball at a player's feet sits up to about a metre from where the tracking puts the player.
CONTROL_RADIUS_M = 1.5
# A dribbled ball runs ahead of the player and is caught up, so its speed swings around theirs.
CONTROL_REL_SPEED = 3.0
# Three frames: the shortest contact the tracking can show as more than noise.
TOUCH_S = 0.12
CONTROL_S = 0.4
# Two touches further apart than this are not one move: the first was a deflection.
TWO_TOUCH_GAP_S = 3.0
# A change with at least this much unseen ball before it is placed by team shape, not by the
# moment the ball reappeared.
GAP_PLACE_S = 2.0
SHAPE_HALF_WINDOW = FPS

HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])


def control_side(table):
    """Per-frame HOME / AWAY where that team controls the ball, -1 otherwise."""
    with np.errstate(invalid="ignore"):
        home_close = table["near_home_dist"] <= CONTROL_RADIUS_M
        away_close = table["near_away_dist"] <= CONTROL_RADIUS_M
        home = home_close & (table["near_home_rel_speed"] <= CONTROL_REL_SPEED)
        away = away_close & (table["near_away_rel_speed"] <= CONTROL_REL_SPEED)
    side = np.full(len(home), -1, dtype=np.int8)
    side[home & ~away_close] = HOME
    side[away & ~home_close] = AWAY
    return side


def control_events(side):
    """[(start, stop, team, strong)] frame-sorted, from the per-frame control side."""
    events = []
    for team in (HOME, AWAY):
        for start, stop in zip(*runs(side == team)):
            if stop - start >= TOUCH_S * FPS:
                events.append((int(start), int(stop), team, stop - start >= CONTROL_S * FPS))
    return sorted(events)


def shape_score(table):
    """Per-frame evidence from the team shapes alone: positive = home has the ball.

    Two signals that do not depend on which way anyone is running, so a back-pass -- everyone
    moving towards the passer's own goal -- does not flip them:

      width    the team with the ball stretches the pitch across; the team without it squeezes
      effort   the team without the ball does the running, chasing it

    plus the Smart Tagging votes. Each signal is put on a common scale by its own spread over the
    match (no labels involved), then they are added and smoothed over +/-1 s.

    Chosen by shape_agreement(), which needs no ground truth: where the ball is in view, how often
    a signal points at the team the ball-based tracker gave possession to. Width and effort agree
    far more often than the alternatives tried (spread along the pitch, pressing, team speed
    towards goal, team height); speed towards goal is barely better than a coin, because back-
    passes and pressing send both teams the same way.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        width = np.log(table["home_spread_width"]) - np.log(table["away_spread_width"])
        effort = table["away_mean_speed"] - table["home_mean_speed"]
    # Mirrored tag pairs mean one phase usually brings two votes; halving puts it near unit scale.
    votes = (table["tag_home_votes"].astype(np.float32) - table["tag_away_votes"]) / 2.0
    return {"width": standardise(width), "effort": standardise(effort), "tags": votes}


def standardise(values):
    """values / their spread over the match; 0 where unknown. Infinite counts as unknown: a log
    of a zero spread, when a team has one player in view, is no measurement at all."""
    values = np.where(np.isfinite(values), values, np.nan)
    scale = np.nanstd(values)
    return np.where(np.isnan(values), 0.0, values / scale).astype(np.float32) if scale else \
        np.zeros(len(values), dtype=np.float32)


def combined_score(parts):
    return rolling_mean(sum(parts.values()), SHAPE_HALF_WINDOW)


def place_change(score, start, stop, new_team):
    """Best single change point in [start, stop) for a change to new_team, by team shape.

    Picks the frame t that best splits the window into "old team" before and "new team" after:
    the old team is the opposite sign, so for a change to home the best t minimises the running
    sum of the score up to t, and for a change to away it maximises it.
    """
    window = np.nan_to_num(score[start:stop])
    if not len(window):
        return start
    prefix = np.concatenate(([0.0], np.cumsum(window)))
    best = np.argmin(prefix) if new_team == HOME else np.argmax(prefix)
    return start + int(best)


def label_segment(events, start, stop, score, seen):
    """[(frame, team)] possession changes within one live stretch, the first at its start."""
    inside = [event for event in events if start <= event[0] < stop]
    if not inside:
        # No control anywhere: the shapes are all there is.
        return [(start, HOME if np.nansum(score[start:stop]) >= 0 else AWAY)]

    state = inside[0][2]
    changes = [(start, state)]
    pending = None
    last_own = inside[0][1]
    for event_start, event_stop, team, strong in inside[1:]:
        if team == state:
            pending, last_own = None, event_stop
            continue
        if strong:
            at = pending[0] if pending is not None else event_start
        elif pending is not None and event_start - pending[1] <= TWO_TOUCH_GAP_S * FPS:
            at = pending[0]
        else:
            pending = (event_start, event_stop)
            continue
        if np.sum(~seen[last_own:at]) >= GAP_PLACE_S * FPS:
            at = place_change(score, last_own, at, team)
        changes.append((at, team))
        state, pending, last_own = team, None, event_stop
    return changes


def live_stretches(half, dead):
    """[(start, stop)] of each run of live graded frames, never crossing a half."""
    starts, stops = runs((half > 0) & ~dead)
    return list(zip(starts.tolist(), stops.tolist()))


def label(table, stoppages):
    """Per-frame label codes for the whole video: H/A/D on graded frames, X outside them."""
    n_frames = len(table["half"])
    dead = deadball.dead_mask(stoppages, n_frames)
    codes = np.full(n_frames, ABSTAIN, dtype=np.int8)
    codes[(table["half"] > 0) & dead] = DEAD

    events = control_events(control_side(table))
    score = combined_score(shape_score(table))
    seen = ~np.isnan(table["ball_x"])
    for start, stop in live_stretches(table["half"], dead):
        changes = label_segment(events, start, stop, score, seen)
        for (frame, team), (next_frame, _) in zip(changes, changes[1:] + [(stop, None)]):
            codes[frame:next_frame] = team
    return codes


def shape_agreement(table, codes):
    """How often each shape signal agrees with the ball-based label, where the ball is in view.

    A check that needs no ground truth: the shape signals exist to stand in for the ball when it
    is missing, so where the ball IS seen they should mostly point the same way as the label the
    ball produced. A signal near 50% carries nothing and should not be trusted in the gaps.
    """
    alive = (codes == HOME) | (codes == AWAY)
    seen = alive & ~np.isnan(table["ball_x"])
    parts = shape_score(table)
    parts["combined"] = combined_score(parts)
    for name, values in parts.items():
        smoothed = rolling_mean(values, SHAPE_HALF_WINDOW)
        voted = seen & (smoothed != 0) & ~np.isnan(smoothed)
        agree = np.mean((smoothed[voted] > 0) == (codes[voted] == HOME))
        print("  {:<9} agrees with the ball-based team on {:5.1f}% of {:.0f} min".format(
            name, 100.0 * agree, voted.sum() / FPS / 60.0))


def describe(codes, table):
    """What the labels claim, in the terms of the scorer, with no ground truth involved."""
    graded = table["half"] > 0
    spells = []
    for _, first, last in features.half_video_blocks(GAME):
        window = codes[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        spells += [(bounds[i + 1] - bounds[i]) / float(FPS) for i in range(len(bounds) - 1)
                   if window[bounds[i]] in (HOME, AWAY)]
    alive = (codes == HOME) | (codes == AWAY)
    print("{} possessions, mean {:.1f} s, median {:.1f} s; home share {:.1f}%; dead {:.1f}%".format(
        len(spells), np.mean(spells), np.median(spells),
        100.0 * np.sum(codes == HOME) / alive.sum(), 100.0 * np.mean(codes[graded] == DEAD)))
    print("possessions shorter than 2 s: {}   longer than 60 s: {}".format(
        sum(s < 2 for s in spells), sum(s > 60 for s in spells)))


if __name__ == '__main__':
    table = features.load()
    codes = label(table, deadball.detect(table))
    describe(codes, table)
    print("\nshape signals vs the ball, where the ball is seen")
    shape_agreement(table, codes)
