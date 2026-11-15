"""IDSSE observed x/y adapter; status references never enter the input store."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/external/idsse'
OUTPUT = ROOT/'data/processed/external/idsse'
PERIODS = ('firstHalf', 'secondHalf')


def frames(path):
    """Stream object-grouped XML with bounded memory, preserving raw attributes."""
    stack = []
    context = {}
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if event == 'start':
            stack.append(element)
            if element.tag == 'FrameSet':
                context = dict(element.attrib)
        else:
            if element.tag == 'Frame' and context:
                yield context, dict(element.attrib)
            if len(stack) > 1:
                stack[-2].remove(element)
            element.clear()
            stack.pop()
            if element.tag == 'FrameSet':
                context = {}


def encode_xy(frame, pitch):
    """Sportec centre-origin metres -> normalised x/y; no S or Z channel."""
    if frame.get('X') is None or frame.get('Y') is None:
        return np.array([np.nan, np.nan])
    return np.array([float(frame['X']), float(frame['Y'])])/pitch + .5


def adapt(match, source=SOURCE, output=OUTPUT):
    meta_path, = source.glob(f'DFL_02_01_*{match}.xml')
    tracking_path, = source.glob(f'DFL_04_03_*{match}.xml')
    provenance = json.loads((source/'provenance.json').read_text())
    hashes = {}
    for path in (meta_path, tracking_path):
        sha = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                sha.update(chunk)
        hashes[path.name] = sha.hexdigest()
        expected = next(f['sha256'] for f in provenance['files'] if f['path'] == path.name)
        if sha.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {path}')
    meta = ET.parse(meta_path)
    general = next(meta.iter('General'))
    sides = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    environment = next(meta.iter('Environment'))
    pitch = np.array([float(environment.get('PitchX')), float(environment.get('PitchY'))])
    ids, teams = [], []
    for team in meta.iter('Team'):
        for player in team.iter('Player'):
            ids.append(player.get('PersonId'))
            teams.append(sides.index(team.get('TeamId')))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate player IDs in metadata')
    slot_of = {pid: i for i, pid in enumerate(ids)}
    frame_ids = {period: set() for period in PERIODS}
    print(f'{match}: indexing frames...', flush=True)
    for context, frame in frames(tracking_path):
        period = context.get('GameSection')
        if period in frame_ids:
            frame_ids[period].add(int(frame['N']))
    keys = [(p, f) for p in PERIODS for f in sorted(frame_ids[p])]
    row_of = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    data = np.full((n, len(ids)+1, 3), np.nan, dtype=np.float32)
    times = np.full(n, np.nan)
    dead = np.full(n, -1, dtype=np.int8)
    ball_seen = np.zeros(n, dtype=bool)
    print(f'{match}: adapting {n} frames...', flush=True)
    for context, frame in frames(tracking_path):
        key = (context.get('GameSection'), int(frame['N']))
        if key not in row_of:
            continue
        row = row_of[key]
        if frame.get('T'):
            timestamp = datetime.fromisoformat(frame['T']).timestamp()
            if np.isfinite(times[row]) and abs(times[row]-timestamp) > 1e-5:
                raise ValueError('Inconsistent timestamps for one frame ID')
            times[row] = timestamp
        team = context.get('TeamId')
        if team == 'BALL':
            if ball_seen[row]:
                raise ValueError('Duplicate ball frame')
            ball_seen[row] = True
            slot = len(ids)
            dead[row] = {'0': 1, '1': 0}.get(frame.get('BallStatus'), -1)
        elif team in sides:
            slot = slot_of[context['PersonId']]
            if teams[slot] != sides.index(team):
                raise ValueError('Player team mismatch')
        else:
            continue
        data[row, slot, :2] = encode_xy(frame, pitch)
    # Split on period, ID gap or timestamp gap: never invent a motion link.
    breaks = [0]
    for i in range(1, n):
        same = keys[i][0] == keys[i-1][0] and keys[i][1] == keys[i-1][1]+1
        if not same or not np.isclose(times[i]-times[i-1], .04, atol=1e-5):
            breaks.append(i)
    breaks.append(n)
    if not np.isfinite(times).all():
        raise ValueError('Missing source timestamps')
    segments = np.array(list(zip(breaks[:-1], breaks[1:])), dtype=np.int64)
    destination = output/match
    destination.mkdir(parents=True, exist_ok=True)
    source_frames = np.array([k[1] for k in keys])
    np.savez_compressed(destination/'inputs.npz', data=data, slots=np.array(teams),
                        pitch=pitch, fps=np.array(25.), frames=source_frames, segments=segments,
                        player_ids=ids, timestamp_seconds=times)
    np.savez_compressed(destination/'reference.npz', dead=dead, known=dead >= 0,
                        frames=source_frames)
    report = dict(match_id=match, frames=n, source_sha256=hashes,
                  segments_half_open=segments.tolist(), fps=25,
                  alive=int((dead == 0).sum()), dead=int((dead == 1).sum()),
                  unknown=int((dead < 0).sum()),
                  inputs='Only raw observed x/y; centre metres normalised; S/Z/status/possession/events excluded',
                  missingness='No interpolation or extrapolation; M flag semantics not assumed',
                  reference='Explicit BallStatus only, missing/other values unknown')
    (destination/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WPY')
    args = parser.parse_args()
    adapt(args.match)
