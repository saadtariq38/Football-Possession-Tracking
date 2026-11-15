# External data audit: SkillCorner sample

## Acquisition and terms

Source: https://github.com/SkillCorner/opendata
Revision: `4340d274572876239c154c90bc507a9b3250a656`
Match: 2017461, Melbourne Victory vs Auckland FC, 2025-05-17.

Downloaded one match only: metadata, extrapolated tracking, dynamic events and
phases of play. README, repository MIT licence and match index are retained under
`data/external/skillcorner/`. Credit SkillCorner and PySport. The upstream README
explicitly presents the dataset for community research and requests attribution.
The licence text and source material are retained for review; no independent
legal determination about additional data rights is implied.

`provenance.json` records pinned URLs, sizes and SHA-256 hashes. Git LFS tracking
was downloaded from the media endpoint and its declared size and SHA-256 verified.
All downloaded data and generated audit reports remain ignored by Git.

## Measured coverage

- 71,451 consecutive tracking frames; documented sampling rate 10 fps.
- 58,452 frames assigned to periods 1 or 2; 12,999 outside those periods.
- Within period frames, possession group is home on 17,796, away on 11,092,
  and unknown on 29,564. Unknown is not a dead-ball label.
- Player in possession is specified on 7,556 period frames.
- Ball x/y is populated on 40,404 period frames, but explicitly detected on
  only 31,875. The difference matters: populated does not mean observed.
- 433 phase intervals; coverage audit counted 28,716 period frames as covered.
  This preliminary calculation uses inclusive endpoints; exact endpoint semantics
  must be established before building calibration labels.
- 1,020 phase-covered period frames have unknown possession, and 1,192 frames
  with known possession fall outside phase coverage under that convention.
  Therefore these two sources must not be treated as interchangeable labels.
- Player data includes `is_detected`; retain only observed positions for a
  like-for-like observation experiment and preserve missing/extrapolated flags.
- Coordinates use metres about pitch centre; the sample pitch is 105 x 68 m.
  The ball also has z, unlike the challenge input. Exclude z from features when
  testing the current algorithm so it cannot exploit unavailable information.

## Decision

Usable for initial experiments on control/continuity against **provider-generated
reference labels**, with explicit masks for unknown team/player possession.
These are derived labels, not independent manually verified ground truth, and
provider definitions can differ from the challenge definition.

Not yet suitable as a definitive dead-ball calibration set. Tracking contains no
explicit live/dead flag. `game_state` in dynamic events means winning/drawing/losing,
not whether the ball is in play. Phases provide a live-play proxy, but their absence
must not silently become D without checking coverage and label semantics.

No labels have been constructed and no parameters have been calibrated from this
sample. The challenge configuration and prediction CSV were left unchanged.

## Next implementation steps

1. Confirm coordinate orientation, 10 fps timing and phase endpoint conventions.
2. Create an adapter that preserves observation masks and unknown reference labels;
   exclude possession, phases and derived events from input features whenever they
   supply reference labels, to prevent circular calibration.
3. Acquire additional external matches and reserve complete matches for validation.
4. Calibrate control and continuity on known-reference frames, reporting coverage.
   Evaluate full spells only on sufficiently complete contiguous reference spans.
5. Obtain explicit dead/live reference data or annotate a rights-cleared external
   match; the rules permit manual labels on external games. Do not annotate the
   provided challenge game. Record external annotator, instructions and uncertainty.
6. Freeze configuration and document external results before regenerating the
   challenge submission. Do not tune from challenge labels, video or public scores.

## Reproduce

```bash
.venv/bin/python scripts/fetch_skillcorner_sample.py
.venv/bin/python scripts/audit_skillcorner_sample.py
```

The downloader reuses the recorded revision on later runs. The audit only reads
this external match; neither script reads challenge tracking, labels or video.

## Adapter and fixed match split

Preparation now includes three matches at the pinned upstream revision:

| Role | Match IDs |
| --- | --- |
| Development / parameter selection | 2017461, 2016236 |
| Held-out validation | 2015213 |

The split is versioned in `possession/external_split.json` and was fixed before
algorithm scoring. Only three matches means validation evidence will be limited.
Do not change the split or tune parameters in response to validation results.

```bash
.venv/bin/python scripts/fetch_skillcorner_sample.py --match 2016236
.venv/bin/python scripts/fetch_skillcorner_sample.py --match 2015213
.venv/bin/python -m possession.skillcorner
```

For each match, `data/processed/external/skillcorner/<id>/` contains:

- `inputs.npz`: observed-only normalised positions, team slots, player IDs, frame
  IDs, period IDs, timestamps, detection flags, source position availability,
  pitch dimensions, verified fps, and half-open period segments.
- `reference.npz`: provider team/player possession references and known masks.
  Team codes are 0=H, 1=A, -1=unknown. There are no D references.
- `summary.json`: coverage, source hashes and adapter conventions.

Source detection flags are three-valued: 1=detected, 0=extrapolated, -1=unspecified.
Only detected x/y positions within playing periods enter the input position array.
The source-position-present mask records where coordinates existed even if excluded.
The third input channel is unavailable speed (NaN), never ball height. Velocity
must be calculated from successive observed positions. Missing observations stay
missing. Centred metre coordinates are converted to normalised pitch coordinates;
source orientation is preserved rather than inventing an attacking-direction flip.

The adapter verified 0.1-second timestamp increments within both periods of all
three matches. Match timestamps restart at 45:00 in period two; do not treat them
as elapsed recording time. Run features and decoding separately on each provided
period segment so smoothing and control continuity cannot bridge halftime.

Phase endpoint semantics remain unconfirmed and phases are not used by the adapter.
Possession and derived events are excluded from inputs to prevent circular scoring
against provider-derived possession references. External control calibration and
one held-out evaluation are now complete; see [CALIBRATION.md](CALIBRATION.md).
The validation match was not used for parameter selection. A metric on known H/A
frames alone cannot establish dead-ball detection quality. The separate IDSSE
live/dead experiment is documented in [STOPPAGE_CALIBRATION.md](STOPPAGE_CALIBRATION.md).
