"""Download one pinned external match, retaining licence, attribution and hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/external/skillcorner'
MATCH = '2017461'


def get(url):
    with urlopen(Request(url, headers={'User-Agent': 'Football-Possession-data-audit'}), timeout=120) as r:
        return r.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default=MATCH)
    match = parser.parse_args().match
    if not match.isdigit():
        raise ValueError('Match ID must be numeric')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT/'provenance.json'
    if manifest.exists():
        revision = json.loads(manifest.read_text())['revision']
    else:
        revision = json.loads(get('https://api.github.com/repos/SkillCorner/opendata/commits/master'))['sha']
    records = json.loads(manifest.read_text()).get('files', []) if manifest.exists() else []
    files = ['LICENSE', 'README.md', 'data/matches.json']
    files += [f'data/matches/{match}/{match}_{suffix}' for suffix in
              ('match.json', 'dynamic_events.csv', 'phases_of_play.csv', 'tracking_extrapolated.jsonl')]
    for name in files:
        dest = OUT/(Path(name).name if not name.startswith('data/matches/') else f'{match}/{Path(name).name}')
        dest.parent.mkdir(exist_ok=True)
        url = f'https://raw.githubusercontent.com/SkillCorner/opendata/{revision}/{name}'
        body = get(url)
        expected = None
        if body.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointer = body.decode().splitlines()
            expected = pointer[1].split('sha256:')[1]
            size = int(pointer[2].split()[1])
            url = f'https://media.githubusercontent.com/media/SkillCorner/opendata/{revision}/{name}'
            if dest.exists() and dest.stat().st_size == size and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:
                body = dest.read_bytes()
            else:
                print(f'Downloading {size/1e6:.1f} MB: {name}', flush=True)
                body = get(url)
            if len(body)!=size or hashlib.sha256(body).hexdigest()!=expected:
                raise ValueError('Tracking download failed LFS checksum/size validation')
        dest.write_bytes(body)
        records = [r for r in records if r['path'] != str(dest.relative_to(OUT))]
        records.append(dict(path=str(dest.relative_to(OUT)), url=url, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
        print('Saved', dest.name, flush=True)
    manifest.write_text(json.dumps(dict(repository='https://github.com/SkillCorner/opendata',
        revision=revision, attribution='SkillCorner / PySport open data', match_ids=sorted({match} | {r['path'].split('/')[0] for r in records if '/' in r['path']}),
        scope='External adapter and calibration preparation; no calibration performed', files=records),indent=2)+'\n')


if __name__=='__main__':
    main()
