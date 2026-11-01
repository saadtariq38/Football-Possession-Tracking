"""Load the tactical tracking into numpy arrays once, and cache them next to the data.

Parsing 199319 lines of text takes a few seconds and every later step wants arrays, not text. The
cache lives in data/cache/, which is gitignored with the rest of data/ -- a frame-level export of
the tracking is exactly the kind of derived file the terms say must not be published.

Everything here stays in the ATD's own units: positions normalised 0-1, speed in m/s. Conversion
to metres happens in features.py, in one place.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, require, source_frames  # noqa: E402
from baseline import HOME, AWAY, team_names, team_of_each_track  # noqa: E402

# Params
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
DIRECTION_PARAMETER = "attack_direction_first_half"
DIRECTIONS = {"left_to_right": 1, "right_to_left": -1}


def cache_dir(data_dir=None):
    return os.path.join(data_dir or DATA_DIR, "cache")


def attack_direction(meta_path):
    """+1 if the home side attacks towards x = 1 in the first half, -1 if towards x = 0.

    Read from the ATD metadata rather than assumed: every "forward" measurement flips sign on it,
    so getting it backwards would invert every team-shape feature without any error.
    """
    home = next(ET.parse(meta_path).getroot().iter("Team"))
    for parameter in home.iter("ProviderParameter"):
        if parameter.findtext("Name") == DIRECTION_PARAMETER:
            value = (parameter.findtext("Value") or "").strip().lower()
            if value in DIRECTIONS:
                return DIRECTIONS[value]
    raise ValueError("{}: no usable {} for the home team".format(meta_path, DIRECTION_PARAMETER))


def parse(meta_path, raw_path, n_frames):
    """(players, ball, team): players (n, slots, 3) and ball (n, 3) as x, y, speed; NaN = unseen."""
    team = team_of_each_track(meta_path)
    players = np.full((n_frames, len(team), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, slots, ball_text = line.rstrip("\n").split(FIELD_SEP)
            row = int(frame_id) - 1  # frame_id counts from 1, submissions from 0
            players[row] = np.array(slots.replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP),
                                    dtype=np.float32).reshape(len(team), 3)
            ball[row] = np.array(ball_text.split(VALUE_SEP), dtype=np.float32)
    return players, ball, team


def load(game=GAME, data_dir=None, refresh=False):
    """{players, ball, team, direction, names} for the tactical recording, from cache when fresh.

    The cache is keyed on the raw file's size and modification time, so a re-download is picked up
    without anyone remembering to delete it.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))], dtype=np.int64)
    path = os.path.join(cache_dir(data_dir), "{}_atd.npz".format(game))

    if not refresh and os.path.isfile(path):
        with np.load(path) as stored:
            if np.array_equal(stored["stamp"], stamp):
                return {"players": stored["players"], "ball": stored["ball"],
                        "team": stored["team"], "direction": int(stored["direction"]),
                        "names": team_names(meta_path)}

    players, ball, team = parse(meta_path, raw_path, source_frames("tactical", game))
    direction = attack_direction(meta_path)
    if not os.path.isdir(cache_dir(data_dir)):
        os.makedirs(cache_dir(data_dir))
    np.savez(path, players=players, ball=ball, team=team, direction=direction, stamp=stamp)
    return {"players": players, "ball": ball, "team": team, "direction": direction,
            "names": team_names(meta_path)}


__all__ = ["HOME", "AWAY", "attack_direction", "parse", "load", "cache_dir"]
