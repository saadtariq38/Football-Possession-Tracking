"""Record expected match assets and local checksums; no external dependencies."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import DATA_DIR, SOURCES, asset_paths


def main():
    root = Path(DATA_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("processed", "cache"):
        (root / folder).mkdir(exist_ok=True)
    entries = []
    for source in SOURCES:
        for asset, filename in asset_paths(source=source).items():
            path = Path(filename).resolve()
            exists = path.is_file()
            entry = dict(source=source, asset=asset,
                         path=str(path.relative_to(root)), present=exists)
            if exists:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                entry.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
            entries.append(entry)
            print(f"{'OK     ' if exists else 'MISSING'} {entry['path']}")
    destination = root / "inventory.json"
    destination.write_text(json.dumps({"assets": entries}, indent=2) + "\n")
    print(f"\nInventory: {destination}")


if __name__ == "__main__":
    main()
