"""Select sequence parameters by official spellF1 on external development data."""
import itertools
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession import spell, stoppage
from possession.calibrate_restart import offset_counts
from possession.idsse import OUTPUT
from possession.idsse_joint_reference import sha
from possession.run_restart import load_models

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('spell_protocol_v3.json')
CONFIG = Path(__file__).with_name('config_spell_v3.json')
REPORT = OUTPUT/'spell_v3_development.json'


def build(match, bundle, allowed):
    if match not in allowed:
        raise ValueError('Only declared development matches may be opened here')
    folder = OUTPUT/match
    source, reference = np.load(folder/'inputs.npz'), np.load(folder/'reference_had_v1.npz')
    if not np.array_equal(source['frames'], reference['frames']):
        raise ValueError('Reference alignment differs')
    pieces = []
    raw_data, pitch, fps, slots = source['data'], source['pitch'], float(source['fps']), source['slots']
    for scenario in ('original', 'degraded'):
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = raw_data[lo:hi]
            if scenario == 'degraded':
                raw = stoppage.degraded_data(raw, pitch, fps, 20261114+sum(ord(c) for c in match)*10+segment)
            print('Extracting spell evidence:', match, scenario, segment, flush=True)
            evidence = spell.extract(raw, slots, pitch, fps, bundle)
            pieces.append(dict(match=match, scenario=scenario, fps=fps, evidence=evidence,
                y=reference['state'][lo:hi], dead_reference=reference['dead'][lo:hi],
                observed=evidence['ball_observed']))
    return pieces


def group_arrays(pieces, predictions):
    gt, pred, runs, offset = [], [], [], 0
    for piece, state in zip(pieces, predictions):
        target = np.where(piece['y'] >= 0, piece['y'], pm.X).astype(np.int8)
        gt.append(target)
        pred.append(state)
        runs.extend([(offset+int(a), offset+int(b)) for a, b in pm.valid_runs(target)])
        offset += len(target)
    return np.concatenate(gt), np.concatenate(pred), runs


def score(pieces, predictions, full=False):
    result = {}
    for match in sorted(set(p['match'] for p in pieces)):
        for scenario in ('original', 'degraded'):
            selected = [(p, s) for p, s in zip(pieces, predictions)
                        if p['match'] == match and p['scenario'] == scenario]
            pp, ss = zip(*selected)
            gt, pred, runs = group_arrays(pp, ss)
            metric = pm.spell_scores(gt, pred, runs)
            macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
            row = dict(spell_f1=metric['mean_f1'], macro_f1=macro,
                       n_gt_spells=metric['n_gt'], n_pred_spells=metric['n_pred'])
            if full:
                counts = [offset_counts(p, s, [.5, 1., 2.]) for p, s in selected]
                eligible = sum(c['missing_eligible'] for c in counts)
                correct = sum(c['missing_correct'] for c in counts)
                row.update(missing_offset_eligible=eligible, missing_offset_correct=correct,
                           missing_offset_accuracy=correct/eligible if eligible else None)
            result[f'{match}/{scenario}'] = row
    return result


def admissible(scores, baseline, full=False):
    for name, row in scores.items():
        base = baseline[name]
        if row['macro_f1'] < base['macro_f1']-.015 or row['spell_f1'] < base['spell_f1']-.015:
            return False
        if full and row['missing_offset_accuracy'] is not None:
            if row['missing_offset_accuracy'] < base['missing_offset_accuracy']-.02:
                return False
    return True


def tie_key(row):
    c = row['config']
    return (row['objective'], c['kind'] == 'v1', -c.get('learned_weight', 0),
            -c.get('missing_multiplier', 0), -c.get('switch_penalty', 0),
            -c.get('smoothing_seconds', 0), c.get('mode') == 'continuous')


def candidates(protocol):
    result = [dict(kind='v1')]
    for learned, missing, smooth, penalty, mode in itertools.product(
        protocol['learned_weight_candidates'], protocol['missing_multiplier_candidates'],
        protocol['smoothing_seconds_candidates'], protocol['switch_penalty_5hz_candidates'],
        protocol['sequence_mode_candidates']):
        if learned == 0 and missing != 0:
            continue
        result.append(dict(kind='hybrid', learned_weight=learned, missing_multiplier=missing,
                           smoothing_seconds=smooth, switch_penalty=penalty, mode=mode))
    return result


def main():
    if CONFIG.exists() or REPORT.exists():
        raise FileExistsError('Frozen spell experiment exists; no reselection')
    protocol = json.loads(PROTOCOL.read_text())
    allowed = protocol['development_matches']
    if set(allowed) & set(protocol['excluded_seen_audit_matches']+[protocol['fresh_validation_match']]):
        raise ValueError('Development and audit matches overlap')
    bundle, _ = load_models()
    pieces = sum([build(match, bundle, allowed) for match in allowed], [])
    coarse = []
    for piece in pieces:
        coarse.append(dict(piece, fps=piece['fps']/5, y=piece['y'][::5],
            dead_reference=piece['dead_reference'][::5], observed=piece['observed'][::5],
            evidence={k: v[::5] for k, v in piece['evidence'].items()}))
    grid = candidates(protocol)
    baseline_coarse = score(coarse, [spell.apply(p['evidence'], grid[0], p['fps'])['state'] for p in coarse])
    rows = []
    for index, config in enumerate(grid):
        predicted = [spell.apply(p['evidence'], config, p['fps'])['state'] for p in coarse]
        scores = score(coarse, predicted)
        row = dict(config=config, scores=scores, allowed=admissible(scores, baseline_coarse),
                   objective=float(np.mean([s['spell_f1'] for s in scores.values()])))
        rows.append(row)
        if index % 10 == 0:
            print(f'Coarse {index+1}/{len(grid)}: {row["objective"]:.4f}, guard={row["allowed"]}', flush=True)
    shortlisted = sorted([r for r in rows if r['allowed']], key=tie_key, reverse=True)[:8]
    if not any(r['config']['kind'] == 'v1' for r in shortlisted):
        shortlisted.append(rows[0])
    baseline_full = score(pieces, [spell.apply(p['evidence'], grid[0], p['fps'])['state'] for p in pieces], full=True)
    full_rows = []
    for index, row in enumerate(shortlisted):
        predicted = [spell.apply(p['evidence'], row['config'], p['fps'])['state'] for p in pieces]
        scores = score(pieces, predicted, full=True)
        record = dict(config=row['config'], scores=scores,
            objective=float(np.mean([s['spell_f1'] for s in scores.values()])),
            allowed=admissible(scores, baseline_full, full=True))
        full_rows.append(record)
        print(f'Full-rate {index+1}/{len(shortlisted)}:', record['config'],
              record['objective'], 'guard', record['allowed'], flush=True)
    best = max((r for r in full_rows if r['allowed']), key=tie_key)
    config = dict(parameters=best['config'], protocol_sha256=sha(PROTOCOL),
        code_sha256=sha(Path(spell.__file__)), metric_code_sha256=sha(ROOT/'evaluation/possession_metrics.py'),
        base_model_sha256=sha(Path(__file__).with_name('model_external_restart_v2.json')),
        development_matches=allowed, fresh_validation_match=protocol['fresh_validation_match'],
        attribution=protocol['attribution'], status='Frozen spellF1-selected sequence parameters; fresh validation pending')
    CONFIG.write_text(json.dumps(config, indent=2)+'\n')
    REPORT.write_text(json.dumps(dict(protocol=protocol, config_sha256=sha(CONFIG),
        coarse_results=rows, full_rate_results=full_rows, selected=best,
        calibration_code_sha256=sha(Path(__file__))), indent=2)+'\n')
    print('Frozen spell configuration:', best['config'], flush=True)


if __name__ == '__main__':
    main()
