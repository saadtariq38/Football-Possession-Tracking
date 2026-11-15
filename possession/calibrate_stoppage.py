"""Predeclared external-only live/dead experiment; refuse to overwrite frozen results."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

from possession.idsse import OUTPUT
from possession import stoppage
from possession.pipeline import features, emissions, decode

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('stoppage_protocol.json')
MODEL = Path(__file__).with_name('model_external_stoppage_v1.json')
REPORT = ROOT/'data/processed/external/idsse/stoppage_report_v1.json'
FIT_RECORD = REPORT.with_name('stoppage_fit_record_v1.json')
CONFIG = json.loads(Path(__file__).with_name('config_external_control_v1.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(match, sampled=True, baseline=False):
    """Feature extraction never receives the reference arrays."""
    source = np.load(OUTPUT/match/'inputs.npz')
    reference = np.load(OUTPUT/match/'reference.npz')
    if not np.array_equal(source['frames'], reference['frames']):
        raise ValueError('Input/reference frames differ')
    fps, pitch = float(source['fps']), source['pitch']
    data = source['data']
    result = []
    for scenario in (0, 1):
        for segment, (lo, hi) in enumerate(source['segments']):
            raw = data[lo:hi]
            if scenario:
                # Stable seeds, no status dependence, same degradation at selection and audit.
                seed = 20261114 + sum(ord(c) for c in match)*10 + segment
                raw = stoppage.degraded_data(raw, pitch, fps, seed)
            x, names = stoppage.motion_features(raw, pitch, fps)
            step = round(fps/5) if sampled else 1
            piece = dict(match=match, scenario=('original', 'degraded')[scenario],
                         segment=segment, x=x[::step].astype(np.float32),
                         y=reference['dead'][lo:hi:step], fps=fps/step)
            if baseline:
                f = features(raw, source['slots'], pitch, fps, CONFIG)
                evidence, _ = emissions(f, np.zeros(len(raw)), np.zeros(len(raw)), fps, CONFIG)
                state, _ = decode(evidence, CONFIG['switch_penalty'], marginals=False)
                piece['baseline'] = (state[::step] == 2).astype(np.int8)
            result.append(piece)
    return result, names


def summary(parts):
    cm = np.sum([p['confusion_alive_dead'] for p in parts], axis=0)
    denominator = cm.sum(axis=0)+cm.sum(axis=1)
    f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(2), where=denominator > 0)
    return dict(known_frames=int(cm.sum()), confusion_alive_dead=cm.tolist(),
                accuracy=float(np.trace(cm)/max(1, cm.sum())), macro_f1=float(f1.mean()),
                dead_f1=float(f1[1]), segments=parts)


def score(pieces, model, penalty, baseline=False):
    scenarios = {}
    for scenario in ('original', 'degraded'):
        parts = []
        for piece in pieces:
            if piece['scenario'] != scenario:
                continue
            predicted = (piece['baseline'] if baseline else stoppage.decode_binary(
                stoppage.logits(piece['x'], model), penalty, piece['fps']))
            parts.append(stoppage.metrics(piece['y'], predicted, piece['fps']))
        scenarios[scenario] = summary(parts)
    return scenarios


def training_arrays(pieces):
    return (np.concatenate([p['x'][p['y'] >= 0] for p in pieces]),
            np.concatenate([p['y'][p['y'] >= 0] for p in pieces]))


def main():
    if MODEL.exists() or REPORT.exists():
        raise RuntimeError('Frozen experiment exists. Do not reselect against held-out results.')
    protocol = json.loads(PROTOCOL.read_text())
    splits = [protocol[k] for k in ('training_matches', 'selection_matches', 'validation_matches')]
    if any(len(split) != 1 for split in splits):
        raise ValueError('This version expects exactly one match in each of the three splits')
    if len(set(sum(splits, []))) != len(sum(splits, [])):
        raise ValueError('Matches overlap across splits')
    for match in sum(splits, []):
        for filename in ('inputs.npz', 'reference.npz', 'summary.json'):
            if not (OUTPUT/match/filename).is_file():
                raise FileNotFoundError(f'Prepare all match inputs before fitting: {match}/{filename}')
    train, names = build(splits[0][0])
    selection, selection_names = build(splits[1][0])
    if names != selection_names:
        raise ValueError('Inconsistent feature schema')
    x, y = training_arrays(train)
    candidates = []
    for ridge in protocol['ridge_candidates']:
        print(f'Fitting training match, ridge={ridge}...', flush=True)
        model = stoppage.fit(x, y, ridge)
        for penalty in protocol['binary_switch_penalty_candidates_at_5fps']:
            scores = score(selection, model, penalty)
            objective = np.mean([v['macro_f1'] for v in scores.values()])
            row = dict(ridge=ridge, switch_penalty=penalty, objective=float(objective), scores=scores)
            candidates.append(row)
            print(f'  switch={penalty}: selection macro F1 {objective:.4f}', flush=True)
    best = max(candidates, key=lambda c: (c['objective'], c['ridge'], -c['switch_penalty']))
    print('Selected:', {k: best[k] for k in ('ridge', 'switch_penalty', 'objective')}, flush=True)
    x, y = training_arrays(train+selection)
    model = stoppage.fit(x, y, best['ridge'])
    model.update(feature_names=names, switch_penalty=best['switch_penalty'],
                 protocol_sha256=sha(PROTOCOL),
                 extraction_code_sha256=sha(stoppage.__file__),
                 fitted_matches=splits[0]+splits[1], validation_matches=splits[2],
                 source_summaries={match: json.loads((OUTPUT/match/'summary.json').read_text())
                                   for match in splits[0]+splits[1]},
                 attribution='Bassek, Rein, Weber & Memmert (2025); DFL / Sportec Solutions; CC BY 4.0',
                 source_doi='10.6084/m9.figshare.28196177.v1',
                 status='External motion-only stoppage model; frozen before held-out evaluation; challenge transfer unverified')
    serialisable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
    MODEL.write_text(json.dumps(serialisable, indent=2)+'\n')
    FIT_RECORD.write_text(json.dumps(dict(
        model_sha256=sha(MODEL), protocol_sha256=sha(PROTOCOL),
        selection_candidates=candidates,
        selected={k: best[k] for k in ('ridge', 'switch_penalty', 'objective')}
    ), indent=2)+'\n')
    del train, selection, x, y
    evaluate_frozen()


def evaluate_frozen():
    if REPORT.exists():
        raise RuntimeError('Held-out report already exists; no repeat evaluation or reselection')
    protocol = json.loads(PROTOCOL.read_text())
    model = json.loads(MODEL.read_text())
    record = json.loads(FIT_RECORD.read_text())
    if (model['protocol_sha256'] != sha(PROTOCOL) or record['model_sha256'] != sha(MODEL)
            or model['extraction_code_sha256'] != sha(stoppage.__file__)):
        raise ValueError('Frozen protocol, model or feature code changed')
    for key in ('imputation', 'mean', 'scale', 'coefficient'):
        model[key] = np.asarray(model[key])
    validation_match, = protocol['validation_matches']
    # The only held-out evaluation happens after the final model has been saved.
    print('Frozen. Evaluating untouched validation match at full 25 fps...', flush=True)
    validation, validation_names = build(validation_match, sampled=False, baseline=True)
    if validation_names != model['feature_names']:
        raise ValueError('Validation feature schema differs')
    report = dict(protocol=protocol, model_sha256=sha(MODEL), fit_record=record,
                  validation_motion_model=score(validation, model, model['switch_penalty']),
                  validation_previous_control_model=score(validation, model, 0, baseline=True),
                  validation_source=json.loads((OUTPUT/validation_match/'summary.json').read_text()),
                  limitations=['One held-out match; no guarantee of transfer to challenge',
                               'Synthetic faults are robustness tests, not challenge observations',
                               'Previous control baseline has no Smart Tagging input on IDSSE',
                               'Transition timing is one-to-one within fixed 2 seconds; unmatched counts retained',
                               'All metrics are binary live/dead, not challenge H/A/D accuracy'])
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    for scenario, result in report['validation_motion_model'].items():
        print(scenario, {k: result[k] for k in ('accuracy', 'macro_f1', 'dead_f1')}, flush=True)
    print(f'Model: {MODEL}\nReport: {REPORT}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-frozen', action='store_true',
                        help='Resume an interrupted first validation without refitting or reselecting')
    args = parser.parse_args()
    evaluate_frozen() if args.evaluate_frozen else main()
