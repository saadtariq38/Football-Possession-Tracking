"""Download one official IDSSE match with Figshare checksums and attribution."""
import hashlib
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/idsse'
URL='https://api.figshare.com/v2/articles/28196177/versions/1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='J03WPY')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with urlopen(Request(URL,headers={'User-Agent':'Football-Possession-audit'}),timeout=120) as r:
        article=json.load(r)
    (OUT/'article_v1.json').write_text(json.dumps(article,indent=2)+'\n')
    wanted=[f for f in article['files'] if args.match in f['name']]
    if len(wanted)!=3:raise ValueError('Expected exactly metadata, events and tracking for match')
    prior=json.loads((OUT/'provenance.json').read_text()) if (OUT/'provenance.json').exists() else {}
    records={f['path']:f for f in prior.get('files',[])}
    for f in wanted:
        dest=OUT/f['name']
        if not dest.exists():
            temporary=dest.with_suffix('.part')
            print('Downloading',f['name'],f['size'],'bytes',flush=True)
            with urlopen(f['download_url'],timeout=120) as r,temporary.open('wb') as h:
                while chunk:=r.read(1024*1024):h.write(chunk)
            temporary.replace(dest)
        digest=hashlib.md5()
        sha=hashlib.sha256()
        with dest.open('rb') as h:
            for chunk in iter(lambda:h.read(1024*1024),b''):
                digest.update(chunk);sha.update(chunk)
        expected=f.get('supplied_md5') or f.get('computed_md5')
        if dest.stat().st_size!=f['size'] or (expected and digest.hexdigest()!=expected):
            raise ValueError(f'Official size/checksum mismatch: {dest}')
        records[dest.name]=dict(path=dest.name,url=f['download_url'],bytes=dest.stat().st_size,
                            sha256=sha.hexdigest(),md5=digest.hexdigest())
        print('Verified',dest.name,flush=True)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        article_id=28196177,version=1,match_ids=sorted(set(prior.get('match_ids',[prior.get('match_id','J03WPY')])+[args.match])),license=article['license'],
        attribution='Bassek, Rein, Weber & Memmert (2025); DFL / Sportec Solutions',
        paper='https://doi.org/10.1038/s41597-025-04505-y',files=list(records.values()),
        purpose='External live/dead adaptation and calibration; see possession/stoppage_protocol.json'),indent=2)+'\n')


if __name__=='__main__':main()
