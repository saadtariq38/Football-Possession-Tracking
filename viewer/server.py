"""Local raw tactical tracking viewer. Run: python viewer/server.py."""
import argparse
from array import array
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import DATA_DIR, GAME, GAMES, asset_paths, source_frames

STATIC = Path(__file__).resolve().parent
OUTPUT = Path(DATA_DIR).resolve() / 'processed' / 'viewer'
CHUNK = 500


def prepare():
    paths = {k: Path(v) for k, v in asset_paths().items()}
    for key in ('atd_raw', 'atd_meta'):
        if not paths[key].is_file():
            raise FileNotFoundError(f'Missing {paths[key]}')
    signature = {key: [str(paths[key].resolve()), paths[key].stat().st_size,
                       paths[key].stat().st_mtime_ns] for key in ('atd_raw', 'atd_meta')}
    manifest_path = OUTPUT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        count = math.ceil(manifest['frames'] / CHUNK)
        if manifest.get('signature') == signature and all(
                (OUTPUT / f'{i:04d}.bin').is_file() for i in range(count)):
            return
    meta = ET.parse(paths['atd_meta']).getroot()
    teams = list(meta.iter('Team'))
    players = list(meta.iter('Player'))
    team_ids = [t.get('id') for t in teams]
    slots = [team_ids.index(p.get('teamId')) for p in players]
    frame_count = source_frames()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Invalidate the manifest before replacing chunks so interrupted builds cannot be reused.
    manifest_path.unlink(missing_ok=True)
    buffer = array('f')
    expected = 1
    chunk_id = 0
    with paths['atd_raw'].open() as handle:
        for line in handle:
            frame, groups, ball = line.strip().split(':')
            if int(frame) != expected:
                raise ValueError(f'Expected raw frame {expected}, got {frame}')
            values = [float(v) for group in groups.split(';') + [ball]
                      for v in group.split(',')]
            if len(values) != (len(slots) + 1) * 3:
                raise ValueError(f'Unexpected slot count at raw frame {frame}')
            buffer.extend(values)
            if expected % CHUNK == 0:
                write_chunk(chunk_id, buffer)
                chunk_id += 1
                buffer = array('f')
            expected += 1
    if expected - 1 != frame_count:
        raise ValueError(f'Expected {frame_count} frames, found {expected - 1}')
    if buffer:
        write_chunk(chunk_id, buffer)
    manifest = dict(version=1, signature=signature, frames=frame_count,
                    chunkSize=CHUNK, stride=(len(slots)+1)*3, slots=slots,
                    teams=[t.findtext('Name') for t in teams], fps=GAMES[GAME]['fps'],
                    pitch=GAMES[GAME]['pitch'])
    manifest_path.write_text(json.dumps(manifest))
    print(f'Prepared {frame_count:,} frames in {chunk_id+1} chunks.', flush=True)


def write_chunk(index, values):
    if values.itemsize != 4:
        raise RuntimeError('This platform does not use 32-bit floats')
    if sys.byteorder != 'little':
        values.byteswap()
    (OUTPUT / f'{index:04d}.bin').write_bytes(values.tobytes())


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        from urllib.parse import urlsplit
        name = urlsplit(path).path
        if name in ('/', '/index.html'):
            return str(STATIC / 'index.html')
        if name in ('/app.js', '/style.css'):
            return str(STATIC / name.lstrip('/'))
        if name == '/data/manifest.json':
            return str(OUTPUT / 'manifest.json')
        if name.startswith('/data/'):
            filename = name.removeprefix('/data/')
            if len(filename) == 8 and filename[:4].isdigit() and filename.endswith('.bin'):
                return str(OUTPUT / filename)
        return str(STATIC / '__not_found__')

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if not args.prepare_only:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        print(f'Viewer: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
