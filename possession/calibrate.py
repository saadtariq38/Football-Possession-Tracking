"""Predeclared external-only search; freeze before a single held-out evaluation."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from possession.pipeline import features, emissions, decode

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT/'data/processed/external/skillcorner'
OUT = ROOT/'data/processed/calibration/control_v1'
FROZEN = ROOT/'possession/config_external_control_v1.json'


def metric(reference, prediction):
    known = np.isin(reference, [0, 1])
    gt, pred = reference[known], prediction[known]
    if not len(gt):
        raise ValueError('No known reference frames')
    f1 = []
    for side in (0, 1):
        tp = np.count_nonzero((gt==side)&(pred==side))
        fp = np.count_nonzero((gt!=side)&(pred==side))
        fn = np.count_nonzero((gt==side)&(pred!=side))
        f1.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return dict(macro_f1_known_HA=float(np.mean(f1)), accuracy_known_HA=float(np.mean(gt==pred)),
                reference_frames=len(gt), total_frames=len(reference), coverage=float(known.mean()),
                predicted_D_on_known_HA=int(np.count_nonzero(pred==2)))


def evaluate(match, config):
    folder = EXTERNAL/match
    with np.load(folder/'inputs.npz') as a:
        prediction = np.full(len(a['frames']), -1, dtype=np.int8)
        observed_ball = np.isfinite(a['data'][:, -1, :2]).all(axis=1)
        for lo, hi in a['segments']:
            fps = float(a['fps'])
            f = features(a['data'][lo:hi], a['slots'], a['pitch'], fps, config)
            evidence, _ = emissions(f, np.zeros(hi-lo), np.zeros(hi-lo), fps, config)
            # Integrate evidence on a 25 fps reference scale. Prevent a 10 fps source
            # from implicitly reducing evidence relative to the fixed switch cost.
            evidence *= 25./fps
            prediction[lo:hi], _ = decode(evidence, config['switch_penalty'], marginals=False)
    with np.load(folder/'reference.npz') as r:
        result = metric(r['team'], prediction)
        for name, mask in [('ball_observed', observed_ball), ('ball_missing', ~observed_ball)]:
            subset = np.where(mask, r['team'], -1)
            if np.isin(subset, [0,1]).any():
                result[name] = metric(subset, prediction)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists() or FROZEN.exists():
        raise RuntimeError('Run already exists; preserve the predeclared protocol and validation result')
    split = json.loads((ROOT/'possession/external_split.json').read_text())
    if set(split['development'])&set(split['validation']):
        raise ValueError('Split leakage')
    base = json.loads((ROOT/'possession/config.json').read_text())
    candidates = [dict(base)]
    for radius, support, decay, penalty in itertools.product([1.5,2.5],[.2,.6],[1.,3.],[2.,4.]):
        c = dict(base, control_radius_m=radius, control_support_seconds=support,
                 continuity_decay_seconds=decay, bridge_decay_seconds=decay, switch_penalty=penalty)
        candidates.append(c)
    protocol = dict(split=split, candidates=candidates,
        objective='Mean known-H/A macro F1 across whole development matches, equal match weights',
        tiebreak='First candidate in declared order',
        validation='Evaluate frozen selected configuration and unchanged baseline once; never reselect',
        unknown_policy='Unknown labels excluded without joining disjoint reference spans',
        inputs='Observed positions only; zero phase/restart inputs to avoid provider-label leakage',
        fps_policy='Multiply per-frame emissions by 25/source_fps before decoding',
        scope='Control/continuity only; no dead-ball or Smart Tagging calibration',
        source_revision=json.loads((ROOT/'data/external/skillcorner/provenance.json').read_text())['revision'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results = []
    for i,c in enumerate(candidates):
        matches = {m:evaluate(m,c) for m in split['development']}
        score = float(np.mean([r['macro_f1_known_HA'] for r in matches.values()]))
        results.append(dict(candidate=i, score=score, matches=matches))
        (OUT/'development.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'Candidate {i+1}/{len(candidates)}: development mean macro F1 {score:.4f}',flush=True)
    winner = max(range(len(results)), key=lambda i:results[i]['score'])
    FROZEN.write_text(json.dumps(candidates[winner],indent=2)+'\n')
    selection = dict(candidate=winner, development=results[winner],
        config_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        status='Frozen before any held-out algorithm scores')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    validation = {}
    for m in split['validation']:
        validation[m] = dict(baseline=evaluate(m,base), selected=evaluate(m,candidates[winner]))
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Frozen candidate:',winner,'Validation:',json.dumps(validation),flush=True)


if __name__=='__main__':
    main()
