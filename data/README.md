# Match data layout

Keep original assets in `tactical/` and `broadcast/`. These paths are used by
`games.py` and the starter notebooks. Keep their filenames unchanged.

```text
data/
├── README.md        # this guide; tracked in Git
├── inventory.json   # generated local asset status and SHA-256 hashes
├── tactical/        # primary input; submission timeline
├── broadcast/       # optional second feed; separate timeline
├── processed/       # generated features, cleaned tracking and aligned events
└── cache/           # disposable intermediate files
```

## Original assets

Both source folders contain these three files, prefixed with `BAR-ATH_tactical`
or `BAR-ATH_broadcast`:

| Suffix | Contents | Use |
| --- | --- | --- |
| `_atd.xml` | Team and tracking-slot metadata | Interpret tracking |
| `_atd.txt` | Player and ball positions and speeds | Main possession input |
| `_homography.csv` | Camera-to-pitch transforms | Project image/pitch coordinates |

`tactical/` also contains `BAR-ATH_tactical_smart_tagging.xml`: automatically
inferred tactical phases and set pieces, used as supporting evidence.

All seven non-video assets are present locally. The optional videos have not
been downloaded. If added later, put `BAR-ATH_tactical.mp4` in `tactical/` and
`BAR-ATH_broadcast.mp4` in `broadcast/`.

Submissions use tactical video frames **0 through 199318**, at **25 fps**.
Broadcast frames are not synchronised to tactical frames. See [DATA.md](../DATA.md)
for exact formats and alignment conventions.

## Working with the data

- Keep original files unchanged; write reusable generated data to `processed/`.
- Use `cache/` for intermediates that can be rebuilt.
- Write prediction CSVs to `../evaluation/results/`.
- Scoring reports go to `../evaluation/reports/`.
- Public labels live separately in `../evaluation/gt/`.

Rebuild the asset inventory from the repository root:

```bash
.venv/bin/python scripts/data_inventory.py
```

This records availability, byte sizes and SHA-256 hashes in `inventory.json`.
Hashes identify local contents; they are not compared with organiser checksums.
Missing files are reported, not downloaded. The command also creates empty
`processed/` and `cache/` folders on a fresh checkout.

Only this README under `data/` is tracked in Git. Assets, generated data,
inventory and caches remain ignored. The original ZIPs were removed from Downloads
after extraction and verification; the extracted assets remain in the source folders.

To keep data elsewhere, set `MTAH_DATA_DIR` to the folder containing `tactical/`
and `broadcast/`. The inventory command honours this setting too.
See [the main README](../README.md#getting-the-data) for access instructions.
