# Explicit live/dead source audit and saved candidate

Update: the external adapter, locked calibration and held-out evaluation are now
complete. See [STOPPAGE_CALIBRATION.md](STOPPAGE_CALIBRATION.md) for the model,
results and separate submission candidate. The source audit below records the
initial inspection preceding that work.

## Saved challenge candidate

`evaluation/results/tactical_external_control_v1.csv` uses the frozen externally
selected control configuration. It contains 199,319 frames and passes the official
format validator. Its separate evidence/provenance folder is
`data/processed/tactical_external_control_v1/`. The original prototype CSV and
`data/processed/possession_v1/` evidence are preserved; their recorded output hash
was checked after generating this candidate. No challenge scoring was performed.

Candidate SHA-256: `ade9f9f80007a34a6351a32ad7b5271270afec62689c057e2f37870ac20bbf21`.
Configuration SHA-256: `28d702b8f687a8b4296d334d7ff72219110a21113f2712058e3cbbcab2bea429`.

The run report identifies partial external control calibration correctly; it does
not claim that dead-ball or Smart Tagging parameters were calibrated. Source hashes,
finite emissions and normalised internal marginals were checked.

## External source

IDSSE / DFL: Bassek, Rein, Weber & Memmert (2025),
*An integrated dataset of spatiotemporal and event data in elite soccer*.
Paper: https://doi.org/10.1038/s41597-025-04505-y
Dataset: https://doi.org/10.6084/m9.figshare.28196177.v1
Licence recorded by official Figshare metadata: CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/ . Retain attribution when using it.

Sample match J03WPY is Fortuna Düsseldorf vs 1. FC Nürnberg, 2022/23.
One match's metadata, events and positions were downloaded. Official Figshare
version-1 metadata is stored with them. File sizes and official MD5 checksums were
verified, and SHA-256 hashes recorded in provenance.json. Downloaded and generated
files remain under ignored data directories, not the public code repository.

## Raw field verification

The raw ball FrameSet (`TeamId="BALL"`) contains an explicit `BallStatus` attribute.
The installed Kloppy Sportec deserialiser maps 1 to alive and 0 to dead. Our audit
reads the raw attribute directly, counting missing/other values as unknown rather
than applying the loader's default of treating a missing status as dead.

| Period | Frames | Alive | Dead | Missing frame IDs |
| --- | --- | --- | --- | --- |
| First half | 69,131 | 46,180 | 22,951 | 0 |
| Second half | 77,080 | 35,280 | 41,800 | 0 |

All 146,211 union-of-object frames have an explicit ball state. Consecutive ball
frame timestamps within each half differ by 0.04 seconds: verified 25 fps. There
are 97 dead-state runs across the two halves. Ball x/y exists on all reference
frames, including dead frames: coordinate availability is not a live/dead label.
Frame numbers jump between halves; never infer continuous motion across that gap.

## Suitability and limits

This is suitable for external live/dead calibration experiments because status is
an explicit reference field, not inferred from absent possession or phase tags.
It is one audited match, not a completed calibration or independent validation.
The source uses a multi-camera tracking system and is cleaner than challenge ATD;
external validation should include artificial observation gaps/jumps while keeping
original reference labels. Such corruption experiments are simulated robustness
checks, not proof of transfer to the challenge match.

Keep BallStatus, BallPossession, provider-derived event state and altitude **out of
model features** if they supply labels or are unavailable in the challenge. Confirm
coordinate conventions and speed units from source documentation before adapting
tracking; do not assume the raw S field is in metres per second. The subsequently
implemented adapter derives motion from adjacent x/y and excludes S.

The planned next stage was to fix an external development/validation split using different IDSSE
matches before scoring, adapt observed x/y separately from explicit state labels,
then calibrate live/dead detection and transition timing; this is now complete.
No challenge video,
manual challenge annotations or public-score tuning may enter that process.

## Reproduce this audit

```bash
.venv/bin/python scripts/fetch_idsse_sample.py
.venv/bin/python scripts/audit_idsse_sample.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_external_control_v1.csv
```
