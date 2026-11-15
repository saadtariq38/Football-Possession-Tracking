"""Run the frozen external team/release candidate on automatic challenge inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from games import GAME, GAMES
from possession import restart, restart_release, stoppage
from possession.pipeline import load_tracking, tag_evidence
from possession.idsse_joint_reference import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).with_name('model_external_restart_v2.json')
RELEASE = Path(__file__).with_name('model_external_release_v2a.json')


def load_models():
    bundle, refinement = json.loads(BASE.read_text()), json.loads(RELEASE.read_text())
    if (bundle['inference_code_sha256'] != sha(Path(restart.__file__)) or
        refinement['code_sha256'] != sha(Path(restart_release.__file__)) or
        refinement['base_model_sha256'] != sha(BASE)):
        raise ValueError('Frozen inference or model hashes differ')
    for path, expected in bundle['frozen_v1_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Frozen dependency changed: {path}')
    if bundle['v1_stoppage_model']['extraction_code_sha256'] != sha(Path(stoppage.__file__)):
        raise ValueError('V1 motion feature code changed')
    for name, expected in (('restart_protocol_v2.json', bundle['protocol_sha256']),
                           ('restart_release_protocol_v2a.json', refinement['protocol_sha256'])):
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen protocol changed')
    return bundle, refinement


def run(output, data_dir=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing candidate: {output}')
    bundle, refinement = load_models()
    print('Loading automatic tactical tracking...', flush=True)
    data, slots, teams, paths = load_tracking(data_dir)
    fps, pitch = GAMES[GAME]['fps'], GAMES[GAME]['pitch']
    tag_margin, restart_context = tag_evidence(paths['smart_tagging'], teams, len(data), fps,
                                              bundle['control_config'])
    print('Computing frozen team evidence and bounded restart release...', flush=True)
    prediction = restart_release.predict(data, slots, pitch, fps, bundle, refinement, tag_margin)
    if not np.isin(prediction['state'], [0, 1, 2]).all():
        raise ValueError('Invalid candidate states')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        handle.write('frame,label\n')
        for frame, state in enumerate(prediction['state']):
            handle.write(f'{frame},{("H", "A", "D")[state]}\n')
    data_root = Path(data_dir).resolve() if data_dir else Path(paths['atd_raw']).resolve().parent.parent
    destination = data_root/'processed'/output.stem
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'evidence.npz', **prediction, tag_margin=tag_margin,
                        restart_context=restart_context)
    validation_path = ROOT/'data/processed/external/idsse/restart_v2a_validation/report.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if validation and (validation['model_sha256'] != sha(BASE) or validation['release_model_sha256'] != sha(RELEASE)):
        raise ValueError('External validation refers to a different model')
    passed = bool(validation and validation['release_gate']['passed'])
    report = dict(status=('Passed fresh external release gate; challenge accuracy unverified' if passed else
                         'Experimental candidate; fresh external release gate not passed'),
        frames=len(data), fps=fps, pitch=list(pitch),
        label_counts={label: int((prediction['state'] == code).sum()) for code, label in enumerate(('H', 'A', 'D'))},
        release_changed_frames=int(prediction['release_changed'].sum()),
        source_sha256={key: sha(Path(paths[key])) for key in ('atd_meta', 'atd_raw', 'smart_tagging')},
        base_model_sha256=sha(BASE), release_model_sha256=sha(RELEASE),
        protocol_sha256=bundle['protocol_sha256'], release_protocol_sha256=refinement['protocol_sha256'],
        runner_sha256=sha(Path(__file__)), submission_sha256=sha(output),
        validation_sha256=sha(validation_path) if validation else None,
        fresh_external_gate_passed=passed,
        tag_contribution='Frozen 1.2 home log-odds per Smart Tagging team margin; not calibrated by this experiment',
        scores='Diagnostic logits and conditional release probabilities, not calibrated correctness probabilities',
        reference='possession/RESTART_IMPROVEMENT.md',
        attribution=refinement['attribution'])
    (destination/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT/'evaluation/results/tactical_external_restart_v2a.csv'))
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()
    run(args.out, args.data_dir)
