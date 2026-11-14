"""Read the tracking file into number arrays, and keep a copy (cache) so it is read only once.

Each line of the tracking file is one frame:

    frame_id : player 1 ; player 2 ; ... ; player 46 : ball

and each player or ball is "x,y,speed" (NaN when not seen). Positions are between 0 and 1.

The cache is saved in data/cache/. Like everything in data/, it is never committed to git.
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
FRAME_SEPARATOR = ":"
PLAYER_SEPARATOR = ";"
VALUE_SEPARATOR = ","
DIRECTION_PARAMETER = "attack_direction_first_half"


def cache_dir(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    return os.path.join(data_dir, "cache")


def attack_direction(meta_path):
    """+1 if the home team attacks towards x = 1 in the first half, -1 if towards x = 0.

    Read from the tracking metadata file, so it is never guessed.
    """
    root = ET.parse(meta_path).getroot()
    home_team = next(root.iter("Team"))  # the first team listed is the home team
    for parameter in home_team.iter("ProviderParameter"):
        if parameter.findtext("Name") != DIRECTION_PARAMETER:
            continue
        value = (parameter.findtext("Value") or "").strip().lower()
        if value == "left_to_right":
            return 1
        if value == "right_to_left":
            return -1
    raise ValueError("{}: no usable {} for the home team".format(meta_path, DIRECTION_PARAMETER))


def parse(meta_path, raw_path, n_frames):
    """Read the tracking file.

    Returns:
        players   array (frames, 46 slots, 3): x, y, speed of each player slot
        ball      array (frames, 3): x, y, speed of the ball
        team      array (46 slots): 0 if the slot is a home player, 1 if away
    """
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            frame_id, players_text, ball_text = line.rstrip("\n").split(FRAME_SEPARATOR)
            row = int(frame_id) - 1  # the file counts frames from 1, the submission from 0

            # "x,y,s;x,y,s;..." -> one flat list of numbers -> 46 rows of 3
            player_numbers = players_text.replace(PLAYER_SEPARATOR, VALUE_SEPARATOR)
            player_numbers = player_numbers.split(VALUE_SEPARATOR)
            players[row] = np.array(player_numbers, dtype=np.float32).reshape(n_slots, 3)

            ball[row] = np.array(ball_text.split(VALUE_SEPARATOR), dtype=np.float32)

    return players, ball, team


def load(game=GAME, data_dir=None, refresh=False):
    """The tracking as arrays: from the cache if it is up to date, otherwise from the file.

    The cache remembers the tracking file's size and date. If the file changes, it is read again.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))], dtype=np.int64)
    path = os.path.join(cache_dir(data_dir), "{}_atd.npz".format(game))

    if not refresh and os.path.isfile(path):
        with np.load(path) as stored:
            if np.array_equal(stored["stamp"], stamp):
                return {
                    "players": stored["players"],
                    "ball": stored["ball"],
                    "team": stored["team"],
                    "direction": int(stored["direction"]),
                    "names": team_names(meta_path),
                }

    players, ball, team = parse(meta_path, raw_path, source_frames("tactical", game))
    direction = attack_direction(meta_path)

    if not os.path.isdir(cache_dir(data_dir)):
        os.makedirs(cache_dir(data_dir))
    np.savez(path, players=players, ball=ball, team=team, direction=direction, stamp=stamp)

    return {
        "players": players,
        "ball": ball,
        "team": team,
        "direction": direction,
        "names": team_names(meta_path),
    }


__all__ = ["HOME", "AWAY", "attack_direction", "parse", "load", "cache_dir"]
