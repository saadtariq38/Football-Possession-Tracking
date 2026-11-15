"""Observed-only external adapter. No challenge data, fitting or event-derived inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/external/skillcorner'
OUTPUT = ROOT/'data/processed/external/skillcorner'


def seconds(timestamp):
    if timestamp is None:
        return np.nan
    hours, minutes, sec = timestamp.split(':')
    return int(hours)*3600+int(minutes)*60+float(sec)


def detection(value):
    return 1 if value is True else 0 if value is False else -1


def encode_frame(row, metadata, slot_of):
    """Keep source detection flags; only explicitly observed positions enter inputs."""
    pitch = np.array([metadata['pitch_length'], metadata['pitch_width']], dtype=float)
    positions = np.full((len(slot_of)+1, 3), np.nan)
    flags = np.full(len(slot_of)+1, -1, dtype=np.int8)
    present = np.zeros(len(slot_of)+1, dtype=bool)
    seen = set()
    active = row.get('period') in (1, 2)
    objects = []
    for player in row.get('player_data', []):
        player_id = player['player_id']
        if player_id not in slot_of or player_id in seen:
            raise ValueError('Unknown or duplicated player ID')
        seen.add(player_id)
        objects.append((slot_of[player_id], player))
    objects.append((len(slot_of), row.get('ball_data') or {}))
    for slot, item in objects:
        flags[slot] = detection(item.get('is_detected'))
        present[slot] = item.get('x') is not None and item.get('y') is not None
        if active and present[slot] and flags[slot] == 1:
            # Metres about pitch centre -> normalised coordinates. Retain source orientation.
            positions[slot, :2] = np.array([item['x'], item['y']])/pitch + .5
    # Third channel is speed, deliberately NaN: no altitude or provider-generated speed input.
    # Our motion features compute velocity from adjacent observations at the source fps.
    possession = row.get('possession') or {}
    group = possession.get('group')
    team_ref = {'home team': 0, 'away team': 1}.get(group, -1) if active else -1
    player_ref = slot_of.get(possession.get('player_id'), -1) if active else -1
    return positions, flags, present, team_ref, player_ref


def adapt(match):
    folder = SOURCE/match
    meta_path = folder/f'{match}_match.json'
    tracking_path = folder/f'{match}_tracking_extrapolated.jsonl'
    provenance = json.loads((SOURCE/'provenance.json').read_text())
    hashes = {}
    for path in (meta_path, tracking_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = next(r for r in provenance['files'] if r['path']==str(path.relative_to(SOURCE)))
        if digest != record['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
        hashes[path.name] = digest
    metadata = json.loads(meta_path.read_text())
    player_ids = [p['id'] for p in metadata['players']]
    if len(set(player_ids)) != len(player_ids):
        raise ValueError('Duplicate metadata player IDs')
    slot_of = {pid: i for i, pid in enumerate(player_ids)}
    sides = [metadata['home_team']['id'], metadata['away_team']['id']]
    slots = np.array([sides.index(p['team_id']) for p in metadata['players']], dtype=np.int8)
    n = sum(1 for _ in tracking_path.open())
    data = np.full((n, len(slots)+1, 3), np.nan, dtype=np.float64)
    flags = np.full((n, len(slots)+1), -1, dtype=np.int8)
    present = np.zeros_like(flags, dtype=bool)
    frames = np.empty(n, dtype=np.int64)
    periods = np.zeros(n, dtype=np.int8)
    times = np.full(n, np.nan)
    team_ref = np.full(n, -1, dtype=np.int8)
    player_ref = np.full(n, -1, dtype=np.int16)
    with tracking_path.open() as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            data[i], flags[i], present[i], team_ref[i], player_ref[i] = encode_frame(row, metadata, slot_of)
            frames[i], periods[i], times[i] = row['frame'], row.get('period') or 0, seconds(row['timestamp'])
    if not np.all(np.diff(frames)==1):
        raise ValueError('Non-contiguous frame IDs; adapter must not bridge gaps')
    adjacent = (periods[1:]>0)&(periods[1:]==periods[:-1])&np.isfinite(times[1:])&np.isfinite(times[:-1])
    diffs = np.diff(times)[adjacent]
    if not len(diffs) or not np.allclose(diffs, .1, atol=1e-6):
        raise ValueError('Timestamp increments do not confirm documented 10 fps')
    segments = []
    for period in (1, 2):
        indices = np.flatnonzero(periods==period)
        if not len(indices) or not np.all(np.diff(indices)==1):
            raise ValueError('Expected contiguous period block')
        segments.append([int(indices[0]), int(indices[-1]+1)])
    destination = OUTPUT/match
    destination.mkdir(parents=True, exist_ok=True)
    # Input and reference stores are separate. No possession/phase/event fields in inputs.
    np.savez_compressed(destination/'inputs.npz', data=data, slots=slots,
        player_ids=player_ids, frames=frames, periods=periods, timestamp_seconds=times,
        detection_flags=flags, source_position_present=present,
        segments=np.array(segments), fps=np.array(10.),
        pitch=np.array([metadata['pitch_length'], metadata['pitch_width']]))
    np.savez_compressed(destination/'reference.npz', team=team_ref, player_slot=player_ref,
        team_known=team_ref>=0, player_known=player_ref>=0, frames=frames)
    report = dict(match_id=match, source_revision=provenance['revision'], source_sha256=hashes,
        frames=n, period_segments_half_open=segments, fps_verified=10,
        known_team_frames=int((team_ref>=0).sum()), unknown_team_frames=int((team_ref<0).sum()),
        dead_reference_frames=0, observed_ball_frames=int(np.isfinite(data[:, -1, :2]).all(axis=1).sum()),
        coordinate_convention='Normalised from centred metres; source x/y orientation preserved',
        reference='Provider-generated possession, unknown=-1; no D inferred from null or phase gaps',
        consumer_requirement='Run motion and decoding separately on each half-open period segment.',
        leakage_policy='Inputs exclude possession, player control labels, phases, events and ball altitude.',
        calibration_status='Not fitted or scored; input preparation only')
    (destination/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match')
    args = parser.parse_args()
    split = json.loads((Path(__file__).parent/'external_split.json').read_text())
    development, validation = split['development'], split['validation']
    if set(development)&set(validation):
        raise ValueError('Match leakage between development and validation')
    for match in [args.match] if args.match else development+validation:
        adapt(match)
