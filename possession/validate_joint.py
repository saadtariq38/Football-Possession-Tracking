"""Frozen external H/A/D restart audit. No optimisation, label repair or fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

from possession.idsse import OUTPUT
from possession.pipeline import features, emissions, decode
from possession import stoppage
from possession.idsse_joint_reference import sha
from possession.joint_metrics import (error_decomposition, frame_metrics, merge_frame_metrics,
    merge_restart_diagnostics, restart_diagnostics, restart_events, transition_indices, window_mask)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('joint_validation_protocol.json')
DESTINATION = OUTPUT/'joint_audit_v1'
REPORT = DESTINATION/'report.json'


def verify_frozen(protocol):
    for relative, expected in protocol['frozen_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen file changed: {relative}')


def infer_segment(data, slots, pitch, fps, config, model):
    """Match production inference with unavailable Smart Tagging set to zero."""
    f = features(data, slots, pitch, fps, config)
    evidence, _ = emissions(f, np.zeros(len(data)), np.zeros(len(data)), fps, config)
    legacy, _ = decode(evidence, config['switch_penalty'], marginals=False)
    evidence[:, 2] = -1e6
    team, _ = decode(evidence, config['switch_penalty'], marginals=False)
    dead, log_odds = stoppage.predict(data, pitch, fps, model)
    combined = team.copy()
    combined[dead == 1] = 2
    return dict(legacy_control=legacy, current_combined=combined,
                team=team, dead=dead, dead_log_odds=log_odds)


def score_segment(reference, dead, predictions, fps, protocol):
    restarts, stops = transition_indices(dead)
    events = restart_events(dead, reference, fps)
    before, after = protocol['window_seconds_before'], protocol['window_seconds_after']
    masks = dict(whole=np.ones(len(reference), dtype=bool),
                 restart_window=window_mask(restarts, len(reference), fps, before, after),
                 stop_window=window_mask(stops, len(reference), fps, before, after))
    result = {}
    for name in protocol['comparisons']:
        predicted = predictions[name]
        binary = (predicted == 2).astype(np.int8)
        result[name] = dict(
            frame_metrics={key: frame_metrics(reference, predicted, mask) for key, mask in masks.items()},
            restart=restart_diagnostics(events, reference, predicted, fps,
                protocol['restart_offsets_seconds'], protocol['recovery_horizon_seconds'],
                protocol['recovery_sustain_seconds']),
            binary_transition_metrics=stoppage.metrics(dead, binary, fps)['transitions'])
    result['error_decomposition'] = error_decomposition(
        reference, predictions['current_combined'], predictions['team'])
    result['reference_events'] = dict(restarts=len(restarts), stops=len(stops),
                                    restart_details=events)
    return result


def aggregate(parts, protocol):
    result = {}
    for name in protocol['comparisons']:
        result[name] = dict(
            frame_metrics={key: merge_frame_metrics([p[name]['frame_metrics'][key] for p in parts])
                           for key in ('whole', 'restart_window', 'stop_window')},
            restart=merge_restart_diagnostics([p[name]['restart'] for p in parts]))
        transitions = {}
        for transition in ('stop', 'resume'):
            rows = [p[name]['binary_transition_metrics'][transition] for p in parts]
            counts = {key: sum(r[key] for r in rows) for key in
                      ('reference_count', 'predicted_count', 'matched_within_2s')}
            counts['unmatched_reference'] = counts['reference_count']-counts['matched_within_2s']
            counts['unmatched_predictions'] = counts['predicted_count']-counts['matched_within_2s']
            counts['matched_mean_absolute_seconds'] = (sum(
                (r['matched_mean_absolute_seconds'] or 0)*r['matched_within_2s'] for r in rows
            )/counts['matched_within_2s'] if counts['matched_within_2s'] else None)
            transitions[transition] = counts
        result[name]['transitions'] = transitions
    result['error_decomposition'] = {key: sum(p['error_decomposition'][key] for p in parts)
                                   for key in parts[0]['error_decomposition']}
    return result


def main():
    if REPORT.exists():
        raise FileExistsError('Joint audit report already exists; do not overwrite or use for tuning')
    protocol = json.loads(PROTOCOL.read_text())
    verify_frozen(protocol)
    folder = OUTPUT/protocol['match']
    reference_path = folder/'reference_had_v1.npz'
    reference_report_path = folder/'reference_had_v1.json'
    reference_report = json.loads(reference_report_path.read_text())
    if sha(reference_path) != reference_report['reference_sha256']:
        raise ValueError('Joint reference checksum changed')
    config = json.loads(Path(__file__).with_name('config_external_control_v1.json').read_text())
    model = json.loads(Path(__file__).with_name('model_external_stoppage_v1.json').read_text())
    if model['extraction_code_sha256'] != sha(Path(stoppage.__file__)):
        raise ValueError('Feature code differs from frozen model')
    source = np.load(folder/'inputs.npz')
    raw_data, slots, pitch, fps = source['data'], source['slots'], source['pitch'], float(source['fps'])
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    # Complete inference for both scenarios before opening any joint reference arrays.
    for scenario in protocol['scenarios']:
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = raw_data[lo:hi]
            if scenario == 'degraded':
                seed = 20261114 + sum(ord(c) for c in protocol['match'])*10 + segment
                raw = stoppage.degraded_data(raw, pitch, fps, seed)
            print(f'Predicting {scenario}, segment {segment}, {len(raw)} frames...', flush=True)
            predictions = infer_segment(raw, slots, pitch, fps, config, model)
            path = DESTINATION/f'{scenario}_segment_{segment}_predictions.npz'
            np.savez_compressed(path, **predictions, frames=source['frames'][lo:hi])
            prediction_records.append(dict(scenario=scenario, segment=segment,
                lo=int(lo), hi=int(hi), path=str(path.relative_to(ROOT)), sha256=sha(path)))
    del raw_data
    with np.load(reference_path) as references:
        reference, dead, frame_ids = references['state'], references['dead'], references['frames']
    if not np.array_equal(frame_ids, source['frames']):
        raise ValueError('Joint reference frame alignment differs')
    details = {}
    for scenario in protocol['scenarios']:
        pieces = []
        for record in prediction_records:
            if record['scenario'] != scenario:
                continue
            lo, hi = record['lo'], record['hi']
            with np.load(ROOT/record['path']) as predictions:
                if not np.array_equal(predictions['frames'], frame_ids[lo:hi]):
                    raise ValueError('Saved prediction frame alignment differs')
                piece = score_segment(reference[lo:hi], dead[lo:hi], predictions, fps, protocol)
            pieces.append(piece)
        details[scenario] = dict(aggregate=aggregate(pieces, protocol), segments=pieces)
    verify_frozen(protocol)
    report = dict(protocol=protocol, protocol_sha256=sha(PROTOCOL),
                  audit_code_sha256={p.name: sha(p) for p in
                      (Path(__file__), Path(__file__).with_name('joint_metrics.py'),
                       Path(__file__).with_name('idsse_joint_reference.py'))},
                  reference=reference_report, prediction_artifacts=prediction_records,
                  results=details, status='Validation only; model, configuration and challenge candidate unchanged',
                  limitations=['Same held-out IDSSE match as binary audit, not a new independent sample',
                    'IDSSE possession follows provider control semantics; challenge ownership definitions may differ',
                    'No challenge Smart Tagging equivalent is available; both methods receive zero tag evidence',
                    'Original/degraded scenarios share references and are not independent matches',
                    'Reference-based windows, strata and transitions are metrics only; never inputs',
                    'All claimed accuracy is external, not challenge H/A/D accuracy'])
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    for scenario, detail in details.items():
        result = detail['aggregate']['current_combined']
        print(scenario, json.dumps(dict(
            whole=result['frame_metrics']['whole'],
            restart_window=result['frame_metrics']['restart_window'],
            restart_offsets=result['restart']['all']['offsets_seconds'],
            recovery=result['restart']['all']['recovery'])), flush=True)
    print(f'Wrote {REPORT}', flush=True)


if __name__ == '__main__':
    main()
