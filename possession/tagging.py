"""Read the Smart Tagging file and turn it into two hints per frame.

Smart Tagging is a list of time windows made by a computer, for example
"BUILD UP, Barcelona, 312 s to 330 s". It has mistakes, so we only use it as a hint.

1. Votes: which team the tags say has the ball.
   - Some tags name the team WITH the ball, e.g. "BUILD UP, Barcelona".
   - Some name the team WITHOUT the ball, e.g. "DEFENDING IN MIDDLE THIRD, Barcelona".
   Both count as a vote for the team with the ball.

2. Set-piece age: seconds since a set-piece tag (corner, throw-in, ...) started. The tag window
   opens a few seconds before the restart, so this tells us roughly when a restart happens.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from load_smart_tagging import load as load_events  # noqa: E402

# Params
# Tags that name the team WITH the ball (lower case: the file uses upper case).
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box"}
# Tags that name the team WITHOUT the ball.
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}
SET_PIECE_CODE = "set pieces"
NO_TEAM = -1


def team_number(tag_team, names):
    """0 for the home team, 1 for away, NO_TEAM for "N/A" or anything else.

    names is (home club name, away club name), taken from the tracking metadata.
    """
    value = (tag_team or "").strip().lower()
    home_name = (names[0] or "").strip().lower()
    away_name = (names[1] or "").strip().lower()
    if value == home_name:
        return 0
    if value == away_name:
        return 1
    return NO_TEAM


def frame_hints(names, n_frames, game=GAME, data_dir=None):
    """Hints for every frame:

        home_votes        how many tags covering this frame say the home team has the ball
        away_votes        the same for the away team
        set_piece_age_s   seconds since a set-piece tag window opened (NaN outside one)
    """
    fps = float(GAMES[game]["fps"])
    home_votes = np.zeros(n_frames, dtype=np.int8)
    away_votes = np.zeros(n_frames, dtype=np.int8)
    set_piece_age = np.full(n_frames, np.nan, dtype=np.float32)

    for event in load_events(game, data_dir):
        start = max(0, event["start_frame"])
        stop = min(n_frames, event["end_frame"])
        if stop <= start:
            continue

        code = (event["code"] or "").strip().lower()
        team = team_number(event["tags"].get("Team"), names)

        if code == SET_PIECE_CODE:
            frames = np.arange(start, stop)
            set_piece_age[start:stop] = (frames - event["start_frame"]) / fps
            continue
        if team == NO_TEAM:
            continue

        # Work out which team has the ball according to this tag.
        if code in IN_POSSESSION:
            team_with_ball = team
        elif code in OUT_OF_POSSESSION:
            team_with_ball = 1 - team
        else:
            continue

        if team_with_ball == 0:
            home_votes[start:stop] += 1
        else:
            away_votes[start:stop] += 1

    return {"home_votes": home_votes, "away_votes": away_votes, "set_piece_age_s": set_piece_age}
