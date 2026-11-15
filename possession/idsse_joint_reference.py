"""Reference-only extension; never rewrites observed inputs or the binary references."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json

import numpy as np

from possession.idsse import SOURCE, OUTPUT, frames


def joint_state(frame):
    status = frame.get('BallStatus')
    if status == '0':
        return 2
    if status == '1':
        return {'1': 0, '2': 1}.get(frame.get('BallPossession'), -1)
    return -1


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_reference(match, source=SOURCE, output=OUTPUT):
    folder = output/match
    destination = folder/'reference_had_v1.npz'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite joint references: {destination}')
    path, = source.glob(f'DFL_04_03_*{match}.xml')
    expected = next(f['sha256'] for f in json.loads((source/'provenance.json').read_text())['files']
                    if f['path'] == path.name)
    source_hash = sha(path)
    if source_hash != expected:
        raise ValueError('Raw XML does not match recorded checksum')
    inputs_path = folder/'inputs.npz'
    old_reference_path = folder/'reference.npz'
    original_hashes = {p.name: sha(p) for p in (inputs_path, old_reference_path)}
    with np.load(inputs_path) as inputs:
        frame_ids = inputs['frames']
        timestamps = inputs['timestamp_seconds']
    with np.load(old_reference_path) as old:
        if not np.array_equal(old['frames'], frame_ids):
            raise ValueError('Old reference frame alignment differs')
        old_dead = old['dead']
    # Timestamp key also permits frame numbers restarting between periods.
    keys = [(int(f), round(float(t)*1e6)) for f, t in zip(frame_ids, timestamps)]
    row_of = {key: i for i, key in enumerate(keys)}
    if len(row_of) != len(keys):
        raise ValueError('Duplicate source frame/timestamp keys')
    state = np.full(len(keys), -1, dtype=np.int8)
    dead = np.full(len(keys), -1, dtype=np.int8)
    seen = np.zeros(len(keys), dtype=bool)
    statuses, possession_codes = Counter(), Counter()
    for context, frame in frames(path):
        if context.get('TeamId') != 'BALL' or not frame.get('T'):
            continue
        key = (int(frame['N']), round(datetime.fromisoformat(frame['T']).timestamp()*1e6))
        if key not in row_of:
            continue
        row = row_of[key]
        if seen[row]:
            raise ValueError('Duplicate ball reference frame')
        seen[row] = True
        state[row] = joint_state(frame)
        dead[row] = {'0': 1, '1': 0}.get(frame.get('BallStatus'), -1)
        statuses[frame.get('BallStatus', '<missing>')] += 1
        possession_codes[frame.get('BallPossession', '<missing>')] += 1
    if not np.array_equal(old_dead, dead):
        raise ValueError('Raw statuses no longer agree with frozen binary references')
    if any(sha(folder/name) != digest for name, digest in original_hashes.items()):
        raise ValueError('Original prepared files changed')
    np.savez_compressed(destination, state=state, known=state >= 0,
                        dead=dead, frames=frame_ids)
    report = dict(match=match, source_sha256=source_hash, original_sha256=original_hashes,
                  reference_sha256=sha(destination), frames=len(state),
                  counts={label: int((state == code).sum())
                          for code, label in ((0, 'H'), (1, 'A'), (2, 'D'), (-1, 'unknown'))},
                  missing_ball_reference_frames=int((~seen).sum()),
                  raw_status_counts=dict(statuses), raw_possession_counts=dict(possession_codes),
                  definition='D requires explicit status 0; H/A require explicit alive status and team 1/2; all else unknown',
                  source_definition='https://www.nature.com/articles/s41597-025-04505-y',
                  licence='CC BY 4.0; Bassek, Rein, Weber & Memmert (2025); DFL / Sportec Solutions',
                  leakage_policy='Reference-only addition; no input or original reference files rewritten')
    (folder/'reference_had_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WQQ')
    args = parser.parse_args()
    write_reference(args.match)
