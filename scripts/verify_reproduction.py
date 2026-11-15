"""Reproduce v1 in an isolated source export, exposing only three automatic assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import asset_paths


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable,
                        help='Python in a separate environment with requirements-submission.txt installed')
    parser.add_argument('--data-dir')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT/'evaluation/results/tactical_external_stoppage_v1.csv')
    parser.add_argument('--report', type=Path, default=ROOT/'data/processed/reproduction_v1.json')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f'Reproduction report already exists: {args.report}')
    expected = sha(args.candidate)
    files = ['games.py', 'possession/__init__.py', 'possession/pipeline.py',
             'possession/stoppage.py', 'possession/config.json',
             'possession/config_external_control_v1.json', 'possession/model_external_stoppage_v1.json',
             'evaluation/possession_metrics.py', 'evaluation/validate_submission.py']
    sources = asset_paths(data_dir=args.data_dir)
    code_hashes = {name: sha(ROOT/name) for name in files}
    asset_hashes = {name: sha(sources[name]) for name in ('atd_meta', 'atd_raw', 'smart_tagging')}
    python = str(Path(args.python).absolute())
    version = subprocess.check_output([python, '-c',
        'import sys,numpy,pandas; print(sys.version); print("numpy",numpy.__version__); '
        'print("pandas",pandas.__version__)'], text=True)
    with tempfile.TemporaryDirectory(prefix='mtah-reproduction-') as temporary:
        export = Path(temporary)
        for name in files:
            target = export/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        assets = export/'data/tactical'
        assets.mkdir(parents=True)
        for name in asset_hashes:
            original = Path(sources[name]).resolve()
            (assets/original.name).symlink_to(original)
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment['MTAH_DATA_DIR'] = str(export/'data')
        command = [python, '-m', 'possession.pipeline', '--config',
                   'possession/config_external_control_v1.json', '--stoppage-model',
                   'possession/model_external_stoppage_v1.json', '--data-dir', str(export/'data'),
                   '--out', str(export/'reproduced.csv')]
        subprocess.run(command, cwd=export, env=environment, check=True)
        subprocess.run([python, 'evaluation/validate_submission.py', 'reproduced.csv'],
                       cwd=export, env=environment, check=True)
        actual = sha(export/'reproduced.csv')
        if actual != expected:
            raise ValueError(f'CSV differs: expected {expected}, reproduced {actual}')
    if any(sha(ROOT/name) != digest for name, digest in code_hashes.items()):
        raise ValueError('Source changed during reproduction')
    if sha(args.candidate) != expected:
        raise ValueError('Original candidate changed during reproduction')
    report = dict(passed=True, check='Byte-identical CSV from isolated source export',
                  runtime=version.strip(), candidate=str(args.candidate.resolve()),
                  original_sha256=expected, reproduced_sha256=actual, code_sha256=code_hashes,
                  automatic_asset_sha256=asset_hashes,
                  exposed_assets=list(asset_hashes),
                  excluded_assets='No video, challenge references, external training data or cached predictions in export')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: byte-identical reproduction; report {args.report}', flush=True)


if __name__ == '__main__':
    main()
