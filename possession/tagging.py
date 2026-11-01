"""Smart Tagging as per-frame hints: which side the tagger thinks has the ball, and set pieces.

Hints, never labels. The tagging is automatic and has false positives and negatives, its windows
are a few seconds loose, and a phase is not a possession. So this module only counts how many tags
cover each frame and what they claim; deciding what to believe is the job of later steps.

A tag says who has the ball in one of two ways. "BUILD UP, Barcelona" names the side in
possession; "DEFENDING IN MIDDLE THIRD, Barcelona" names the side *without* it. Both become a vote
for the side that has the ball. Several codes come in mirrored pairs with identical windows
(PROGRESSION / DEFENDING IN MIDDLE THIRD, BUILD UP / DEFENDING IN ATTACKING THIRD, ...), so one
moment of play usually carries two votes for the same side, not two independent opinions.

Set pieces are fixed-length windows -- 35 s for corners, throw-ins, goal kicks and free kicks, 5 s
for kickoffs -- that open a few seconds BEFORE the restart: averaged over the 35 s windows the ball
is mostly unseen and the players slow for the first ~4-6 s, then play resumes. So what a set-piece
tag says is "dead ball, restart shortly after this window opens", and set_piece_age_s, the time
since the window opened, is the useful form of it. Being somewhere inside the window says little.
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
# Codes whose Team tag is the side in possession, and codes whose Team tag is the side without it.
# Matched lower case: the file has upper-case codes and title-case values.
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box"}
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"
# Set piece Type -> small integer code, 0 meaning no set piece. Unknown types still mark a set
# piece, as OTHER_SET_PIECE, rather than silently vanishing.
SET_PIECE_TYPES = {"throw-in": 1, "corner kick": 2, "goal kick": 3, "free kick": 4,
                   "kick off (after goal)": 5, "kick off (start)": 6}
OTHER_SET_PIECE = 7
NO_SIDE = -1


def side_of(tag_team, names):
    """0 for home, 1 for away, NO_SIDE for N/A or anything unrecognised. names = (home, away)."""
    value = (tag_team or "").strip().lower()
    for side, name in enumerate(names):
        if value == (name or "").strip().lower():
            return side
    return NO_SIDE


def frame_hints(names, n_frames, game=GAME, data_dir=None):
    """Per-frame arrays built from the tagging. names = (home club, away club) from the ATD.

    home_votes / away_votes   how many tags covering the frame say that side has the ball
    set_piece                 SET_PIECE_TYPES code of a set piece covering the frame, 0 if none
    set_piece_side            side the set piece is tagged to, NO_SIDE if N/A (every throw-in)
    set_piece_age_s           seconds since that set piece window opened, NaN outside one
    goal_side                 side tagged as scoring a goal covering the frame, NO_SIDE if none
    """
    fps = float(GAMES[game]["fps"])
    votes = np.zeros((2, n_frames), dtype=np.int8)
    set_piece = np.zeros(n_frames, dtype=np.int8)
    set_piece_side = np.full(n_frames, NO_SIDE, dtype=np.int8)
    set_piece_age = np.full(n_frames, np.nan, dtype=np.float32)
    goal_side = np.full(n_frames, NO_SIDE, dtype=np.int8)

    for event in load_events(game, data_dir):
        start = max(0, event["start_frame"])
        stop = min(n_frames, event["end_frame"])
        if stop <= start:
            continue
        code = (event["code"] or "").strip().lower()
        side = side_of(event["tags"].get("Team"), names)

        if code == SET_PIECE_CODE:
            kind = (event["tags"].get("Type") or "").strip().lower()
            set_piece[start:stop] = SET_PIECE_TYPES.get(kind, OTHER_SET_PIECE)
            set_piece_side[start:stop] = side
            set_piece_age[start:stop] = (np.arange(start, stop) - event["start_frame"]) / fps
        elif code == GOAL_CODE:
            goal_side[start:stop] = side
        elif side != NO_SIDE and code in IN_POSSESSION:
            votes[side, start:stop] += 1
        elif side != NO_SIDE and code in OUT_OF_POSSESSION:
            votes[1 - side, start:stop] += 1

    return {"home_votes": votes[0], "away_votes": votes[1], "set_piece": set_piece,
            "set_piece_side": set_piece_side, "set_piece_age_s": set_piece_age,
            "goal_side": goal_side}
