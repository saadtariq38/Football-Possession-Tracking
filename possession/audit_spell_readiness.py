"""Ranking-metric diagnostic on saved external predictions; no tuning or model changes."""
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession.idsse_joint_reference import sha
from possession.idsse import OUTPUT

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = OUTPUT/'restart_v2a_validation/spell_readiness.json'


def main():
    if DESTINATION.exists():
        raise FileExistsError('Saved ranking-metric diagnostic already exists')
    validation_path = OUTPUT/'restart_v2a_validation/report.json'
    validation = json.loads(validation_path.read_text())
    folder = OUTPUT/validation['fresh_match']
    source, reference = np.load(folder/'inputs.npz'), np.load(folder/'reference_had_v1.npz')
    if not np.array_equal(source['frames'], reference['frames']):
        raise ValueError('Reference alignment differs')
    gt = np.where(reference['state'] >= 0, reference['state'], pm.X).astype(np.int8)
    runs = []
    for lo, hi in source['segments']:
        runs.extend([(int(lo+a), int(lo+b)) for a, b in pm.valid_runs(gt[lo:hi])])
    results = {}
    for scenario in ('original', 'degraded'):
        results[scenario] = {}
        for method in ('v1', 'v2', 'unrefined_v2', 'team_only_change', 'stop_only_change'):
            predicted = np.full(len(gt), pm.X, dtype=np.int8)
            for artifact in validation['prediction_artifacts']:
                if artifact['scenario'] != scenario:
                    continue
                path = ROOT/artifact['path']
                if sha(path) != artifact['sha256']:
                    raise ValueError('Saved external predictions changed')
                with np.load(path) as saved:
                    predicted[artifact['lo']:artifact['hi']] = saved[method]
            filled = pm.fill_abstentions(predicted, runs)
            scores = pm.spell_scores(gt, filled, runs)
            low, high = pm.bootstrap_spell_f1(gt, filled, runs, 1000)
            results[scenario][method] = dict(
                spell_f1=scores['mean_f1'], per_iou=scores['per_iou'],
                n_reference_spells=scores['n_gt'], n_predicted_spells=scores['n_pred'],
                bootstrap_5_95=[low, high],
                reference_stats=pm.possession_stats(gt, runs),
                prediction_stats=pm.possession_stats(filled, runs))
    report = dict(source_validation_sha256=sha(validation_path),
        official_metric_code_sha256=sha(ROOT/'evaluation/possession_metrics.py'),
        script_sha256=sha(Path(__file__)), match=validation['fresh_match'],
        external_only=True, results=results,
        status='Post-hoc ranking-metric diagnostic, not a model-selection or tuning run',
        limitation='Same fresh match already scored for frame metrics; no new independent validation. No challenge ground-truth access or model changes.')
    DESTINATION.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({s:{m:{k:v[k] for k in ('spell_f1', 'n_reference_spells', 'n_predicted_spells', 'bootstrap_5_95')}
                        for m,v in rows.items()} for s,rows in results.items()}, indent=2))


if __name__ == '__main__':
    main()
