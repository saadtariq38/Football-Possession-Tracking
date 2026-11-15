# Raw tactical tracking viewer

From the repository root, run:

```bash
.venv/bin/python viewer/server.py
```

Open http://127.0.0.1:8765. The server binds only to localhost. Stop with Ctrl+C.
The original tactical ATD text and XML files are required. No video or third-party
Python packages are needed. `MTAH_DATA_DIR` is supported through `games.py`.

## Controls

- Play/pause, speed selector (0.25 to 4 times), and timeline scrubbing.
- Arrow buttons or keyboard left/right: move one frame. Space: play/pause.
- Jump to any zero-based video frame.
- Click a dot: inspect its tracking slot, team, pitch coordinates and speed.
- Toggle track IDs or one second of recent observation trails.

Starts at video frame 11185, the documented first-half kickoff anchor.
Time is elapsed recording time, not the match clock. Coordinates retain their
original pitch orientation: x increases rightward, y increases downward.
Home is coral, away is blue, and the ball is pale yellow.

## Data conventions

The viewer shows automatic observations only, without possession predictions,
ground-truth labels, interpolation or manual corrections. Missing positions are
not drawn. Missing speed is displayed as unavailable. Track IDs are slots, not
stable player identities. Trails connect successive observed positions of a slot;
they can include tracker jumps and do not prove one person's continuous movement.
No line is drawn across an unobserved frame. Off-pitch coordinates are retained.

The ball has no height field in this representation. Its dot is the supplied x/y
position, not evidence of ground contact or player control.

## Local processing

The server validates sequential raw frame IDs and slot/frame counts, then packs
all observations into little-endian float32 chunks of 500 frames. NaNs remain NaNs;
float32 introduces only small numerical rounding relative to the source text.
Chunks and the manifest live in ignored `data/processed/viewer/` and can be rebuilt
using `--prepare-only`. The browser keeps up to eight loaded chunks in memory.
The manifest records source path, size and modification time to invalidate its
cache when inputs change. Delete the generated viewer folder to force a rebuild.

The server serves only the viewer files and generated tracking chunks. Keep it
local; these match-derived files must not be published with the public repository.
