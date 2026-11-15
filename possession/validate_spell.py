"""Fresh official spellF1 evaluation with no fitting or model selection."""
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession import spell, restart_release, stoppage
from possession.calibrate_spell import CONFIG, PROTOCOL, score, group_arrays
from possession.idsse import OUTPUT
from possession.idsse_joint_reference import sha, write_reference
from possession.run_restart import load_models
from possession.validate_restart import verify as verify_previous

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = OUTPUT/'spell_v3_validation'
REPORT = DESTINATION/'report.json'


def load_config():
    config = json.loads(CONFIG.read_text())
    if (config['code_sha256'] != sha(Path(spell.__file__)) or
        config['metric_code_sha256'] != sha(ROOT/'evaluation/possession_metrics.py') or
        config['protocol_sha256'] != sha(PROTOCOL) or
        config['base_model_sha256'] != sha(Path(__file__).with_name('model_external_restart_v2.json'))):
        raise ValueError('Frozen spell code, metric, protocol or model changed')
    return config


def fresh_gate(baseline, candidate):
    gain = float(np.mean([candidate[k]['spell_f1']-baseline[k]['spell_f1'] for k in baseline]))
    checks = {}
    for key, b in baseline.items():
        c = candidate[key]
        checks[key] = dict(spell_noninferior=c['spell_f1'] >= b['spell_f1']-.01,
            macro_noninferior=c['macro_f1'] >= b['macro_f1']-.015,
            missing_restart_noninferior=(c['missing_offset_accuracy'] is None or
                c['missing_offset_accuracy'] >= b['missing_offset_accuracy']-.02),
            spell_gain=c['spell_f1']-b['spell_f1'],
            macro_gain=c['macro_f1']-b['macro_f1'],
            missing_restart_gain=(c['missing_offset_accuracy']-b['missing_offset_accuracy']
                if c['missing_offset_accuracy'] is not None else None))
    passed = gain >= .02 and all(row[k] for row in checks.values() for k in
        ('spell_noninferior', 'macro_noninferior', 'missing_restart_noninferior'))
    return dict(passed=passed, mean_spell_gain=gain, required_gain=.02, checks=checks)


def main():
    if REPORT.exists():
        raise FileExistsError('Fresh spell validation already exists; no repeat or reselection')
    config = load_config()
    protocol = json.loads(PROTOCOL.read_text())
    bundle, refinement = load_models()
    verify_previous(bundle)
    match = protocol['fresh_validation_match']
    if match in protocol['development_matches']+protocol['excluded_seen_audit_matches']:
        raise ValueError('Fresh match overlaps previously used matches')
    folder = OUTPUT/match
    if not (folder/'reference_had_v1.npz').exists():
        write_reference(match)
    reference_report = json.loads((folder/'reference_had_v1.json').read_text())
    if sha(folder/'reference_had_v1.npz') != reference_report['reference_sha256']:
        raise ValueError('Reference checksum differs')
    source = np.load(folder/'inputs.npz')
    data, pitch, fps, slots = source['data'], source['pitch'], float(source['fps']), source['slots']
    DESTINATION.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for scenario in protocol['scenarios']:
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = data[lo:hi]
            if scenario == 'degraded':
                raw = stoppage.degraded_data(raw, pitch, fps, 20261114+sum(ord(c) for c in match)*10+segment)
            print('Fresh spell predictions:', scenario, segment, flush=True)
            evidence = spell.extract(raw, slots, pitch, fps, bundle)
            candidate = spell.apply(evidence, config['parameters'], fps)
            baseline = spell.apply(evidence, dict(kind='v1'), fps)
            v2a = restart_release.predict(raw, slots, pitch, fps, bundle, refinement)
            path = DESTINATION/f'{scenario}_half_{segment}.npz'
            np.savez_compressed(path, **evidence, v3=candidate['state'], v1=baseline['state'],
                v2a=v2a['state'], team_v3=candidate['team'], home_log_odds=candidate['home_log_odds'],
                frames=source['frames'][lo:hi])
            artifacts.append(dict(scenario=scenario, segment=segment, lo=int(lo), hi=int(hi),
                                  path=str(path.relative_to(ROOT)), sha256=sha(path)))
    # Reference arrays are opened only after every external prediction is saved.
    with np.load(folder/'reference_had_v1.npz') as reference:
        target, dead = reference['state'], reference['dead']
        if not np.array_equal(reference['frames'], source['frames']):
            raise ValueError('Reference alignment differs')
    pieces, predictions = [], {name: [] for name in ('v1', 'v2a', 'v3')}
    for record in artifacts:
        lo, hi = record['lo'], record['hi']
        with np.load(ROOT/record['path']) as saved:
            if not np.array_equal(saved['frames'], source['frames'][lo:hi]):
                raise ValueError('Saved prediction alignment differs')
            pieces.append(dict(match=match, scenario=record['scenario'], fps=fps,
                y=target[lo:hi], dead_reference=dead[lo:hi], observed=saved['ball_observed']))
            for name in predictions:
                predictions[name].append(saved[name])
    results = {name: score(pieces, states, full=True) for name, states in predictions.items()}
    details = {}
    for name, states in predictions.items():
        details[name] = {}
        for scenario in protocol['scenarios']:
            selected = [(p, s) for p, s in zip(pieces, states) if p['scenario'] == scenario]
            pp, ss = zip(*selected)
            gt, pred, runs = group_arrays(pp, ss)
            details[name][scenario] = dict(
                spell_per_iou=pm.spell_scores(gt, pred, runs)['per_iou'],
                spell_bootstrap_5_95=list(pm.bootstrap_spell_f1(gt, pred, runs, 1000)),
                reference_stats=pm.possession_stats(gt, runs), prediction_stats=pm.possession_stats(pred, runs),
                accuracy=float((gt[gt != pm.X] == pred[gt != pm.X]).mean()))
    gate = fresh_gate(results['v1'], results['v3'])
    load_config()
    verify_previous(bundle)
    report = dict(config=config, config_sha256=sha(CONFIG), protocol_sha256=sha(PROTOCOL),
        fresh_match=match, reference=reference_report, prediction_artifacts=artifacts,
        results=results, details=details, gate=gate, validation_code_sha256=sha(Path(__file__)),
        status=('Passed prespecified fresh external spellF1 gate; challenge transfer unverified' if gate['passed'] else
                'Failed fresh external spellF1 gate; experimental only, preserve v1 fallback'),
        limitations=['Single fresh match with shared-reference original/degraded scenarios',
            'External provider possession definitions and tracking quality may differ from challenge',
            'No Smart Tagging equivalent externally; challenge keeps fixed tag contribution',
            'No parameters or candidate selection changed after fresh scoring'])
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(gate=gate, results=results), indent=2), flush=True)
    print(f'Wrote {REPORT}', flush=True)


if __name__ == '__main__':
    main()
