"""One fresh external evaluation of frozen v2, with prespecified ablations and gate."""
import copy
import argparse
import json
from pathlib import Path

import numpy as np

from possession import restart, stoppage
from possession.calibrate_restart import evaluate, PROTOCOL, MODEL, FIT_REPORT
from possession.idsse import OUTPUT
from possession.idsse_joint_reference import sha, write_reference
from possession.joint_metrics import (frame_metrics, merge_frame_metrics, merge_restart_diagnostics,
    restart_events, restart_diagnostics, transition_indices, window_mask)
from possession.validate_joint import infer_segment

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = OUTPUT/'restart_v2_validation'
REPORT = DESTINATION/'report.json'
METHODS = ('v1', 'team_only_change', 'stop_only_change', 'v2')


def release_gate(before, after):
    details = {}
    for scenario in ('original', 'degraded'):
        b, a = before[scenario], after[scenario]
        missing = a['missing_offset_false_dead_rate']
        old_missing = b['missing_offset_false_dead_rate']
        details[scenario] = dict(
            whole_accuracy_noninferior=a['whole']['accuracy'] >= b['whole']['accuracy']-.002,
            dead_f1_noninferior=a['dead_f1'] >= b['dead_f1']-.005,
            missing_restart_false_dead_noninferior=(True if missing is None or old_missing is None else missing <= old_missing),
            whole_accuracy_delta=a['whole']['accuracy']-b['whole']['accuracy'],
            dead_f1_delta=a['dead_f1']-b['dead_f1'],
            missing_restart_false_dead_delta=(missing-old_missing if missing is not None and old_missing is not None else None))
    gain = float(np.mean([after[s]['offset_accuracy']-before[s]['offset_accuracy']
                         for s in ('original', 'degraded')]))
    passed = gain >= .01 and all(row[key] for row in details.values() for key in
        ('whole_accuracy_noninferior', 'dead_f1_noninferior', 'missing_restart_false_dead_noninferior'))
    return dict(passed=passed, mean_restart_offset_accuracy_gain=gain,
                required_gain=.01, scenarios=details)


def diagnostics(pieces, predictions):
    result = {}
    for scenario in ('original', 'degraded'):
        rows = [(p, v) for p, v in zip(pieces, predictions) if p['scenario'] == scenario]
        windows, restarts, transitions, missing = [], [], [], []
        for piece, state in rows:
            anchors, _ = transition_indices(piece['dead_reference'])
            mask = window_mask(anchors, len(state), piece['fps'], 5., 5.)
            windows.append(frame_metrics(piece['y'], state, mask))
            missing.append(frame_metrics(piece['y'], state, ~piece['observed']))
            events = restart_events(piece['dead_reference'], piece['y'], piece['fps'])
            restarts.append(restart_diagnostics(events, piece['y'], state, piece['fps'],
                                                [0., .5, 1., 2., 5.]))
            transitions.append(stoppage.metrics(piece['dead_reference'], (state == 2).astype(int),
                                                piece['fps'])['transitions'])
        result[scenario] = dict(restart_window=merge_frame_metrics(windows),
            ball_missing_frames=merge_frame_metrics(missing), restart=merge_restart_diagnostics(restarts),
            per_half_transitions=transitions)
    return result


def verify(bundle):
    if sha(PROTOCOL) != bundle['protocol_sha256']:
        raise ValueError('Experiment protocol changed after model freeze')
    if sha(Path(restart.__file__)) != bundle['inference_code_sha256']:
        raise ValueError('V2 inference code changed after freeze')
    for path, expected in bundle['frozen_v1_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'V1 artifact changed: {path}')
    # Also preserve all files frozen by the previous audit, including the v1 submission.
    old_protocol = json.loads(Path(__file__).with_name('joint_validation_protocol.json').read_text())
    for path, expected in old_protocol['frozen_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Previous audit artifact changed: {path}')


def main(release_path=None):
    destination = OUTPUT/'restart_v2a_validation' if release_path else DESTINATION
    report_path = destination/'report.json'
    if REPORT.exists() or (OUTPUT/'restart_v2a_validation/report.json').exists():
        raise FileExistsError('Fresh v2 validation exists; no repeated evaluation or selection')
    bundle = json.loads(MODEL.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    if sha(MODEL) != fit_report['model_sha256']:
        raise ValueError('Frozen model differs from fit report')
    verify(bundle)
    refinement = None
    if release_path:
        from possession import restart_release
        refinement = json.loads(Path(release_path).read_text())
        release_protocol = Path(__file__).with_name('restart_release_protocol_v2a.json')
        if (refinement['base_model_sha256'] != sha(MODEL) or
            refinement['code_sha256'] != sha(Path(restart_release.__file__)) or
            refinement['protocol_sha256'] != sha(release_protocol)):
            raise ValueError('Frozen release model, protocol or code changed')
    match = protocol['fresh_validation_match']
    if match in bundle['fitted_matches'] or match == protocol['excluded_previous_audit_match']:
        raise ValueError('Fresh validation overlaps a development or previous audit match')
    folder = OUTPUT/match
    if not (folder/'reference_had_v1.npz').exists():
        write_reference(match)
    reference_report = json.loads((folder/'reference_had_v1.json').read_text())
    if sha(folder/'reference_had_v1.npz') != reference_report['reference_sha256']:
        raise ValueError('Fresh joint reference hash changed')
    source = np.load(folder/'inputs.npz')
    data, slots, pitch, fps = source['data'], source['slots'], source['pitch'], float(source['fps'])
    config, v1 = bundle['control_config'], bundle['v1_stoppage_model']
    team_only_bundle = copy.deepcopy(bundle)
    team_only_bundle['stop_head'] = dict(kind='v1', penalty=v1['switch_penalty'], missing_shift=0.)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    # No validation reference arrays are opened until every prediction is saved.
    for scenario in protocol['scenarios']:
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = data[lo:hi]
            if scenario == 'degraded':
                raw = stoppage.degraded_data(raw, pitch, fps,
                    20261114+sum(ord(c) for c in match)*10+segment)
            print(f'Fresh predictions: {scenario}, segment {segment}...', flush=True)
            baseline = infer_segment(raw, slots, pitch, fps, config, v1)
            candidate = (restart_release.predict(raw, slots, pitch, fps, bundle, refinement)
                         if refinement else restart.predict(raw, slots, pitch, fps, bundle))
            changed_team_only = restart.predict(raw, slots, pitch, fps, team_only_bundle)
            changed_stop_only = baseline['team'].copy()
            changed_stop_only[candidate['dead'] == 1] = 2
            path = destination/f'{scenario}_half_{segment}.npz'
            extra = {}
            if refinement:
                extra.update(unrefined_v2=candidate['base_state'],
                    release_probability_dead=candidate['release_probability_dead'],
                    release_changed=candidate['release_changed'])
            np.savez_compressed(path, v1=baseline['current_combined'],
                team_only_change=changed_team_only['state'], stop_only_change=changed_stop_only,
                v2=candidate['state'], v2_team=candidate['team'], v2_dead=candidate['dead'],
                v2_home_log_odds=candidate['home_log_odds'], v2_dead_log_odds=candidate['dead_log_odds'],
                observed=candidate['ball_observed'], frames=source['frames'][lo:hi], **extra)
            artifacts.append(dict(scenario=scenario, segment=segment, lo=int(lo), hi=int(hi),
                                  path=str(path.relative_to(ROOT)), sha256=sha(path)))
    with np.load(folder/'reference_had_v1.npz') as references:
        y, dead = references['state'], references['dead']
        if not np.array_equal(references['frames'], source['frames']):
            raise ValueError('Fresh reference alignment mismatch')
    pieces = []
    methods = METHODS+('unrefined_v2',) if refinement else METHODS
    predictions = {method: [] for method in methods}
    for artifact in artifacts:
        lo, hi = artifact['lo'], artifact['hi']
        with np.load(ROOT/artifact['path']) as saved:
            if not np.array_equal(saved['frames'], source['frames'][lo:hi]):
                raise ValueError('Saved prediction frame alignment mismatch')
            pieces.append(dict(y=y[lo:hi], dead_reference=dead[lo:hi], fps=fps,
                               observed=saved['observed'], scenario=artifact['scenario']))
            for method in methods:
                predictions[method].append(saved[method])
    scores = {method: evaluate(pieces, predictions[method], protocol['reference_restart_offsets_seconds'])
              for method in methods}
    detailed = {method: diagnostics(pieces, predictions[method]) for method in methods}
    gate = release_gate(scores['v1'], scores['v2'])
    verify(bundle)
    report = dict(model_sha256=sha(MODEL), protocol_sha256=sha(PROTOCOL), fresh_match=match,
        fitted_matches=bundle['fitted_matches'], model_settings=dict(
            stop={k: v for k, v in bundle['stop_head'].items() if k != 'model'},
            team={k: v for k, v in bundle['team_head'].items() if k != 'model'}),
        reference=reference_report, prediction_artifacts=artifacts, scores=scores,
        diagnostics=detailed, release_gate=gate, validation_code_sha256=sha(Path(__file__)),
        status=('Passed prespecified fresh external gate; challenge transfer unverified' if gate['passed'] else
                'Experimental candidate only: failed prespecified fresh external gate; keep v1 recommendation'),
        limitations=['Single fresh external match, plus synthetic corruption; no challenge accuracy established',
                     'External provider ownership semantics may differ from challenge possession definitions',
                     'Smart Tagging unavailable externally; challenge retains frozen tag strength',
                     'Ablations are diagnostics and cannot be selected retrospectively as a winner',
                     'V1 was refitted on both development matches; all fresh results exclude them'])
    if refinement:
        report.update(release_model_sha256=sha(Path(release_path)),
                      release_protocol_sha256=refinement['protocol_sha256'],
                      release_settings={k: v for k, v in refinement['head'].items() if k != 'model'},
                      variant='v2a bounded release refinement; v2 key is the prespecified primary candidate')
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(gate=gate, scores={method:{scenario:dict(
        accuracy=s['whole']['accuracy'], macro_f1=s['whole']['macro_f1_HAD'],
        restart_offset_accuracy=s['offset_accuracy'], missing_restart_false_dead=s['missing_offset_false_dead_rate'])
        for scenario, s in rows.items()} for method, rows in scores.items()}), indent=2), flush=True)
    print(f'Wrote {report_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-model', default=None)
    args = parser.parse_args()
    main(args.release_model)
