"""Audit one external match. Does not read challenge data or fit parameters."""
import csv
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/external/skillcorner'
MATCH = '2017461'


def main():
    folder = BASE/MATCH
    metadata = json.loads((folder/f'{MATCH}_match.json').read_text())
    counts = Counter()
    groups = Counter()
    period_counts = Counter()
    first_active = None
    phase_coverage = set()
    with (folder/f'{MATCH}_phases_of_play.csv').open() as handle:
        phases = list(csv.DictReader(handle))
    # Endpoint convention has not yet been validated. Inclusive for coverage audit only.
    for phase in phases:
        phase_coverage.update(range(int(phase['frame_start']), int(phase['frame_end'])+1))
    previous = None
    frame_steps = Counter()
    with (folder/f'{MATCH}_tracking_extrapolated.jsonl').open() as handle:
        for line in handle:
            d = json.loads(line)
            counts['frames'] += 1
            if previous is not None:
                frame_steps[d['frame']-previous] += 1
            previous = d['frame']
            active = d['period'] in (1, 2)
            period_counts[str(d['period'])] += 1
            if not active:
                continue
            counts['period_frames'] += 1
            group = d['possession'].get('group')
            groups[str(group)] += 1
            ball = d['ball_data']
            position = ball.get('x') is not None and ball.get('y') is not None
            counts['ball_position_present'] += position
            counts['ball_explicitly_detected'] += ball.get('is_detected') is True
            counts['player_possession_present'] += d['possession'].get('player_id') is not None
            counts['phase_covered_period_frames'] += d['frame'] in phase_coverage
            counts['unknown_possession_inside_phase'] += group is None and d['frame'] in phase_coverage
            counts['known_possession_outside_phase'] += group is not None and d['frame'] not in phase_coverage
            for player in d['player_data']:
                counts['player_observations'] += 1
                counts['player_explicitly_detected'] += player.get('is_detected') is True
            if first_active is None and position:
                first_active = {k: d[k] for k in ('frame', 'timestamp', 'period', 'ball_data', 'possession')}
                first_active['player_sample'] = d['player_data'][:1]
    with (folder/f'{MATCH}_dynamic_events.csv').open() as handle:
        events = list(csv.DictReader(handle))
    report = dict(match_id=MATCH, teams=[metadata['home_team']['short_name'], metadata['away_team']['short_name']],
        pitch_m=[metadata['pitch_length'], metadata['pitch_width']], documented_fps=10,
        counts=dict(counts), period_counts=dict(period_counts), frame_steps=dict(frame_steps),
        team_possession_groups=dict(groups), phase_intervals=len(phases),
        event_types=dict(Counter(e['event_type'] for e in events)),
        event_game_state_values=dict(Counter(e['game_state'] for e in events)),
        sample=first_active,
        conclusions=[
            'External provider-generated possession, not independent human ground truth.',
            'No explicit per-frame in-play/dead flag in tracking schema.',
            'Null possession must remain unknown, not D.',
            'Phase coverage is a live-play proxy, not an independently verified dead-ball reference.',
            'game_state in event data is score state, not live/dead status.',
            'Filter extrapolated positions using is_detected when comparing with observed challenge inputs.',
            '10 fps source requires timing-aware adaptation; phase endpoints still need confirmation.',
            'No parameter calibration performed. Single-match schema audit is not held-out validation.'
        ])
    (BASE/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
