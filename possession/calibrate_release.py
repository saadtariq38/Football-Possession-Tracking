"""Bounded release refinement on declared external development matches only."""
import json
from pathlib import Path

import numpy as np

from possession import restart, restart_release, stoppage
from possession.calibrate_restart import MODEL as BASE, evaluate, serialise
from possession.idsse import OUTPUT
from possession.idsse_joint_reference import sha

PROTOCOL = Path(__file__).with_name('restart_release_protocol_v2a.json')
MODEL = Path(__file__).with_name('model_external_release_v2a.json')
REPORT = OUTPUT/'restart_release_v2a_fit.json'


def build(match, allowed, bundle):
    if match not in allowed:
        raise ValueError('Release calibration cannot read validation matches')
    folder = OUTPUT/match
    source, reference = np.load(folder/'inputs.npz'), np.load(folder/'reference_had_v1.npz')
    if not np.array_equal(source['frames'], reference['frames']):
        raise ValueError('Alignment mismatch')
    data, pitch, fps, slots = source['data'], source['pitch'], float(source['fps']), source['slots']
    pieces = []
    for scenario in ('original', 'degraded'):
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = data[lo:hi]
            if scenario == 'degraded':
                raw = stoppage.degraded_data(raw, pitch, fps, 20261114+sum(ord(c) for c in match)*10+segment)
            print('Release development:', match, scenario, segment, flush=True)
            base = restart.predict(raw, slots, pitch, fps, bundle)
            x, names, region, observed = restart_release.feature_matrix(raw, pitch, fps,
                base['dead'], base['dead_log_odds'])
            pieces.append(dict(x=x, region=region, observed=observed, base=base,
                y=reference['state'][lo:hi], dead_reference=reference['dead'][lo:hi],
                scenario=scenario, fps=fps))
    return pieces, names


def training(pieces):
    xs, ys = [], []
    for p in pieces:
        ids = np.arange(0, len(p['y']), round(p['fps']/5))
        ids = ids[p['region'][ids] & (p['dead_reference'][ids] >= 0)]
        xs.append(p['x'][ids])
        ys.append(p['dead_reference'][ids])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    if MODEL.exists() or REPORT.exists():
        raise FileExistsError('Frozen release experiment exists')
    protocol = json.loads(PROTOCOL.read_text())
    if sha(BASE) != protocol['base_model_sha256']:
        raise ValueError('Base model changed')
    bundle = json.loads(BASE.read_text())
    allowed = [protocol['training_match'], protocol['selection_match']]
    if set(allowed) & {protocol['fresh_validation_match'], protocol['excluded_previous_audit_match']}:
        raise ValueError('Development/validation overlap')
    train, names = build(allowed[0], allowed, bundle)
    selection, selection_names = build(allowed[1], allowed, bundle)
    if names != selection_names:
        raise ValueError('Release feature schemas differ')
    offsets = [.5, 1., 2.]
    baseline = evaluate(selection, [p['base']['state'] for p in selection], offsets)
    def objective(scores):
        missing = [s['missing_offset_false_dead_rate'] for s in scores.values()
                   if s['missing_offset_false_dead_rate'] is not None]
        return (-float(np.mean(missing)) if missing else 0.,
                float(np.mean([s['whole']['accuracy'] for s in scores.values()])))
    candidates = [dict(settings=dict(kind='none'), allowed=True, objective=objective(baseline), scores=baseline)]
    x, y = training(train)
    models = {}
    for ridge in protocol['ridge_candidates']:
        models[ridge] = stoppage.fit(x, y, ridge)
        probabilities = [stoppage.sigmoid(stoppage.logits(p['x'], models[ridge])) for p in selection]
        for advance in protocol['maximum_advance_seconds_candidates']:
            for threshold in protocol['maximum_dead_probability_candidates']:
                head = dict(kind='learned', ridge=ridge, maximum_advance_seconds=advance,
                    maximum_dead_probability=threshold, sustain_seconds=protocol['sustained_missing_live_evidence_seconds'])
                states = []
                for p, probability in zip(selection, probabilities):
                    dead = restart_release.apply(p['base']['dead'], p['observed'], probability, p['fps'], head)
                    team = p['base']['team'].copy()
                    th = bundle['team_head']
                    if th['kind'] == 'learned' and th['mode'] == 'reset_at_predicted_dead':
                        team = restart.team_decode(p['base']['home_log_odds'], dead, th['penalty'], p['fps'], th['mode'])
                    team[dead == 1] = 2
                    states.append(team)
                scores = evaluate(selection, states, offsets)
                permitted = all(s['whole']['accuracy'] >= baseline[n]['whole']['accuracy']-.002 and
                    s['dead_f1'] >= baseline[n]['dead_f1']-.005 and
                    s['dead_recall'] >= baseline[n]['dead_recall']-.01 for n, s in scores.items())
                row = dict(settings=head, allowed=permitted, objective=objective(scores), scores=scores)
                candidates.append(row)
                print('Release', head, row['objective'], 'guard', permitted, flush=True)
    best = max((r for r in candidates if r['allowed']), key=lambda r:
        (tuple(r['objective']), r['settings']['kind'] == 'none', r['settings'].get('ridge', 0),
         -r['settings'].get('maximum_advance_seconds', 0), -r['settings'].get('maximum_dead_probability', 0)))
    head = dict(best['settings'])
    if head['kind'] == 'learned':
        x, y = training(train+selection)
        head['model'] = stoppage.fit(x, y, head['ridge'])
    model = dict(head=head, feature_names=names, base_model_sha256=sha(BASE),
        protocol_sha256=sha(PROTOCOL), code_sha256=sha(Path(restart_release.__file__)),
        fitted_matches=allowed, fresh_validation_match=protocol['fresh_validation_match'],
        attribution=protocol['attribution'], status='Frozen external-only bounded release model; fresh validation pending')
    MODEL.write_text(json.dumps(serialise(model), indent=2)+'\n')
    REPORT.write_text(json.dumps(serialise(dict(protocol=protocol, candidates=candidates,
        selected=best['settings'], model_sha256=sha(MODEL),
        code_sha256=sha(Path(__file__)))), indent=2)+'\n')
    print('Frozen release refinement:', best['settings'], flush=True)


if __name__ == '__main__':
    main()
