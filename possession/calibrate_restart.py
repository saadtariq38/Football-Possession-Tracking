"""Development-only restart experiment. Never opens either audit match."""
import json
from pathlib import Path

import numpy as np

from possession import restart, stoppage
from possession.idsse import OUTPUT
from possession.idsse_joint_reference import sha
from possession.joint_metrics import frame_metrics, merge_frame_metrics, restart_events
from possession.validate_joint import infer_segment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('restart_protocol_v2.json')
MODEL = Path(__file__).with_name('model_external_restart_v2.json')
FIT_REPORT = OUTPUT/'restart_v2_fit.json'
CONFIG_PATH = Path(__file__).with_name('config_external_control_v1.json')
V1_PATH = Path(__file__).with_name('model_external_stoppage_v1.json')


def serialise(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialise(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build(match, config, v1, allowed):
    if match not in allowed:
        raise ValueError('Calibration may only open the two declared development matches')
    folder = OUTPUT/match
    source = np.load(folder/'inputs.npz')
    references = np.load(folder/'reference_had_v1.npz')
    if not np.array_equal(source['frames'], references['frames']):
        raise ValueError('Input/reference alignment differs')
    data, fps, pitch, slots = source['data'], float(source['fps']), source['pitch'], source['slots']
    pieces = []
    for scenario in ('original', 'degraded'):
        for segment, (lo, hi) in enumerate(source['segments']):
            print(f'Extracting {match} {scenario} half {segment}...', flush=True)
            raw = data[lo:hi]
            if scenario == 'degraded':
                raw = stoppage.degraded_data(raw, pitch, fps,
                    20261114+sum(ord(c) for c in match)*10+segment)
            sx, sn = restart.stop_features(raw, pitch, fps)
            tx, tn = restart.team_features(raw, slots, pitch, fps, config)
            baseline = infer_segment(raw, slots, pitch, fps, config, v1)
            pieces.append(dict(match=match, scenario=scenario, segment=segment, fps=fps,
                sx=sx.astype(np.float32), tx=tx.astype(np.float32),
                y=references['state'][lo:hi], dead_reference=references['dead'][lo:hi],
                observed=np.isfinite(raw[:, -1, :2]).all(axis=1),
                v1_team=baseline['team'], v1_dead=baseline['dead'], v1_combined=baseline['current_combined']))
    return pieces, sn, tn


def offset_counts(piece, predicted, offsets):
    eligible = correct = missing_eligible = missing_correct = missing_false_dead = 0
    events = restart_events(piece['dead_reference'], piece['y'], piece['fps'])
    for event in events:
        for offset in offsets:
            index = event['index']+round(offset*piece['fps'])
            if index >= event['live_end'] or piece['y'][index] not in (0, 1):
                continue
            eligible += 1
            correct += int(predicted[index] == piece['y'][index])
            if not piece['observed'][index]:
                missing_eligible += 1
                missing_correct += int(predicted[index] == piece['y'][index])
                missing_false_dead += int(predicted[index] == 2)
    return dict(eligible=eligible, correct=correct, missing_eligible=missing_eligible,
                missing_correct=missing_correct, missing_false_dead=missing_false_dead)


def evaluate(pieces, states, offsets):
    result = {}
    for scenario in ('original', 'degraded'):
        chosen = [(piece, state) for piece, state in zip(pieces, states) if piece['scenario'] == scenario]
        whole = merge_frame_metrics([frame_metrics(p['y'], state) for p, state in chosen])
        cm = np.array(whole['confusion_reference_rows_predicted_columns_HAD'])
        binary = np.array([[cm[:2, :2].sum(), cm[:2, 2].sum()], [cm[2, :2].sum(), cm[2, 2]]])
        denominator = binary.sum(axis=0)+binary.sum(axis=1)
        f1 = np.divide(2*np.diag(binary), denominator, out=np.zeros(2), where=denominator > 0)
        offset_parts = [offset_counts(p, state, offsets) for p, state in chosen]
        counts = {key: sum(part[key] for part in offset_parts) for key in offset_parts[0]}
        alive = int(cm[:2].sum())
        result[scenario] = dict(whole=whole, binary_macro_f1=float(f1.mean()), dead_f1=float(f1[1]),
            dead_recall=float(binary[1, 1]/binary[1].sum()),
            false_dead_rate=float(binary[0, 1]/alive),
            alive_accuracy=float((cm[0, 0]+cm[1, 1])/alive), offsets=counts,
            offset_accuracy=counts['correct']/counts['eligible'] if counts['eligible'] else None,
            missing_offset_accuracy=counts['missing_correct']/counts['missing_eligible'] if counts['missing_eligible'] else None,
            missing_offset_false_dead_rate=counts['missing_false_dead']/counts['missing_eligible'] if counts['missing_eligible'] else None)
    return result


def dead_states(pieces, head, v1):
    return [(p['v1_dead'] if head['kind'] == 'v1' else stoppage.decode_binary(
        restart.stop_logits(p['sx'], head, v1), head['penalty'], p['fps'])) for p in pieces]


def combine(teams, deads):
    result = []
    for team, dead in zip(teams, deads):
        state = team.copy()
        state[dead == 1] = 2
        result.append(state)
    return result


def stop_objective(scores):
    return float(np.mean([.5*s['binary_macro_f1']+.25*s['offset_accuracy']+
        .25*(s['missing_offset_accuracy'] if s['missing_offset_accuracy'] is not None else s['offset_accuracy'])
        for s in scores.values()]))


def stop_allowed(scores, baseline):
    return all(s['dead_f1'] >= baseline[name]['dead_f1']-.005 and
               s['dead_recall'] >= baseline[name]['dead_recall']-.01 and
               s['false_dead_rate'] <= baseline[name]['false_dead_rate']+.005
               for name, s in scores.items())


def team_objective(scores):
    return float(np.mean([.6*s['alive_accuracy']+.4*s['offset_accuracy'] for s in scores.values()]))


def training_arrays(pieces, column):
    xs, ys = [], []
    for piece in pieces:
        indices = np.arange(0, len(piece['y']), round(piece['fps']/5))
        target = piece['dead_reference'] if column == 'sx' else piece['y']
        valid = target[indices] >= 0 if column == 'sx' else np.isin(target[indices], [0, 1])
        indices = indices[valid]
        xs.append(piece[column][indices])
        ys.append(target[indices] if column == 'sx' else (target[indices] == 0).astype(int))
    return np.concatenate(xs), np.concatenate(ys)


def main():
    if MODEL.exists() or FIT_REPORT.exists():
        raise FileExistsError('Frozen v2 experiment exists; no refit or reselection')
    protocol = json.loads(PROTOCOL.read_text())
    allowed = [protocol['training_match'], protocol['selection_match']]
    if len(set(allowed+[protocol['fresh_validation_match'], protocol['excluded_previous_audit_match']])) != 4:
        raise ValueError('Development and audit matches must be disjoint')
    config, v1 = json.loads(CONFIG_PATH.read_text()), json.loads(V1_PATH.read_text())
    initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
        (CONFIG_PATH, V1_PATH, Path(__file__).with_name('pipeline.py'), Path(stoppage.__file__))}
    train, sn, tn = build(allowed[0], config, v1, allowed)
    selection, ssn, stn = build(allowed[1], config, v1, allowed)
    if sn != ssn or tn != stn:
        raise ValueError('Feature schemas differ')
    offsets = protocol['reference_restart_offsets_seconds']
    baseline = evaluate(selection, [p['v1_combined'] for p in selection], offsets)
    # For binary selection, encode every predicted alive frame as the reference H/A.
    # This is a scoring-only construction, equivalent to alive/dead correctness at offsets.
    def stop_score(ds):
        states = [np.where(d == 1, 2, np.where(np.isin(p['y'], [0, 1]), p['y'], 0)).astype(np.int8)
                  for p, d in zip(selection, ds)]
        return evaluate(selection, states, offsets)
    v1_head = dict(kind='v1', penalty=float(v1['switch_penalty']), missing_shift=0.)
    sb = stop_score([p['v1_dead'] for p in selection])
    stop_candidates = [dict(settings=v1_head, objective=stop_objective(sb), allowed=True, scores=sb)]
    sx, sy = training_arrays(train, 'sx')
    for ridge in protocol['ridge_candidates']:
        print(f'Fitting directional stop head ridge={ridge}...', flush=True)
        model = stoppage.fit(sx, sy, ridge)
        for penalty in protocol['stop_switch_penalty_candidates_5hz']:
            for shift in protocol['missing_dead_logit_shift_candidates']:
                settings = dict(kind='learned', ridge=ridge, penalty=penalty, missing_shift=shift)
                scores = stop_score(dead_states(selection, dict(**settings, model=model), v1))
                row = dict(settings=settings, objective=stop_objective(scores),
                           allowed=stop_allowed(scores, sb), scores=scores)
                stop_candidates.append(row)
                print('Stop', settings, row['objective'], 'guard', row['allowed'], flush=True)
    chosen_stop = max((r for r in stop_candidates if r['allowed']), key=lambda r:
        (r['objective'], r['settings']['kind'] == 'v1', r['settings'].get('ridge', 0),
         -r['settings']['penalty'], r['settings']['missing_shift']))
    sh = dict(chosen_stop['settings'])
    if sh['kind'] == 'learned':
        sh['model'] = stoppage.fit(sx, sy, sh['ridge'])
    ds = dead_states(selection, sh, v1)
    baseline_team_scores = evaluate(selection, combine([p['v1_team'] for p in selection], ds), offsets)
    legacy_permitted = all(s['whole']['accuracy'] >= baseline[n]['whole']['accuracy']-.002
                           for n, s in baseline_team_scores.items())
    team_candidates = [dict(settings=dict(kind='legacy'), objective=team_objective(baseline_team_scores),
                            allowed=legacy_permitted, scores=baseline_team_scores)]
    tx, ty = training_arrays(train, 'tx')
    for ridge in protocol['ridge_candidates']:
        print(f'Fitting symmetric team head ridge={ridge}...', flush=True)
        model = restart.fit_team(tx, ty, ridge)
        home_odds = [stoppage.logits(p['tx'], model) for p in selection]
        for penalty in protocol['team_switch_penalty_candidates_5hz']:
            for mode in protocol['team_sequence_candidates']:
                teams = [restart.team_decode(odds, dead, penalty, p['fps'], mode)
                         for p, odds, dead in zip(selection, home_odds, ds)]
                scores = evaluate(selection, combine(teams, ds), offsets)
                permitted = all(s['whole']['accuracy'] >= baseline[n]['whole']['accuracy']-.002
                                for n, s in scores.items())
                row = dict(settings=dict(kind='learned', ridge=ridge, penalty=penalty, mode=mode),
                           objective=team_objective(scores), allowed=permitted, scores=scores)
                team_candidates.append(row)
                print('Team', row['settings'], row['objective'], 'guard', permitted, flush=True)
    permitted_teams = [r for r in team_candidates if r['allowed']]
    if permitted_teams:
        chosen_team = max(permitted_teams, key=lambda r:
            (r['objective'], r['settings']['kind'] == 'legacy', r['settings'].get('ridge', 0),
             -r['settings'].get('penalty', 0), r['settings'].get('mode') == 'continuous'))
    else:
        chosen_stop, sh = stop_candidates[0], dict(v1_head)
        chosen_team = dict(settings=dict(kind='legacy'), objective=team_objective(baseline),
                           allowed=True, scores=baseline, fallback='No new joint option passed guards')
    th = dict(chosen_team['settings'])
    print('Selected stop:', chosen_stop['settings'], 'team:', th, flush=True)
    # Refit selected learned heads on both development matches; no held-out arrays opened.
    if sh['kind'] == 'learned':
        fx, fy = training_arrays(train+selection, 'sx')
        sh['model'] = stoppage.fit(fx, fy, sh['ridge'])
        del fx, fy
    if th['kind'] == 'learned':
        fx, fy = training_arrays(train+selection, 'tx')
        th['model'] = restart.fit_team(fx, fy, th['ridge'])
        del fx, fy
    bundle = dict(version=2, stop_head=sh, team_head=th, stop_feature_names=sn, team_feature_names=tn,
        control_config=config, v1_stoppage_model=v1, protocol_sha256=sha(PROTOCOL),
        inference_code_sha256=sha(Path(restart.__file__)), frozen_v1_sha256=initial_hashes,
        fitted_matches=allowed, fresh_validation_match=protocol['fresh_validation_match'],
        development_sources={match: json.loads((OUTPUT/match/'reference_had_v1.json').read_text()) for match in allowed},
        status='Frozen before fresh validation; external-only calibration; challenge transfer unverified',
        attribution=protocol['attribution'])
    MODEL.write_text(json.dumps(serialise(bundle), indent=2)+'\n')
    report = dict(protocol=protocol, model_sha256=sha(MODEL), initial_hashes=initial_hashes,
        stop_candidates=stop_candidates, team_candidates=team_candidates,
        chosen_stop=chosen_stop['settings'], chosen_team=chosen_team['settings'],
        calibration_code_sha256=sha(Path(__file__)),
        note='Existing v1 stop baseline was previously refitted on both development matches. These are development comparisons; only fresh validation tests generalisation.')
    FIT_REPORT.write_text(json.dumps(serialise(report), indent=2)+'\n')
    for relative, expected in initial_hashes.items():
        if sha(ROOT/relative) != expected:
            raise ValueError('A frozen v1 artifact changed')
    print(f'Frozen model: {MODEL}\nDevelopment report: {FIT_REPORT}', flush=True)


if __name__ == '__main__':
    main()
