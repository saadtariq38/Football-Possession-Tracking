# Local setup

Run these commands from the repository root. Install uv first if unavailable.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
source .venv/bin/activate
python evaluation/run_evaluation.py
```

The lock file records the working macOS arm64 environment, including notebook dependencies.
The four supplied baseline CSVs can be evaluated without downloading match assets.
To open the starter notebooks:

```bash
python -m notebook starter/
```

## Match assets

Register and accept the terms at https://landing.mobileworldcapital.com/more-than-a-hack-2027.
The organisers provide the download link; there is no public asset URL in the repository.
Extract the supplied bundle so `data/tactical/` and `data/broadcast/` contain the files
listed in `data/README.md`. Alternatively, set `MTAH_DATA_DIR` to the bundle's parent folder.
The match assets must remain outside Git.

Once the assets are present:

```bash
python starter/load_smart_tagging.py
python starter/baseline.py
python evaluation/run_evaluation.py
```

Baseline generation requires tactical tracking metadata and raw tracking. Videos are not
needed for that step. Broadcast assets are optional and have a separate timeline.

## Raw tracking viewer

```bash
.venv/bin/python viewer/server.py
```

Open http://127.0.0.1:8765. On first run, the server prepares binary chunks under
`data/processed/viewer/`; subsequent runs reuse them while the source tracking and
metadata are unchanged. No extra packages are needed. Use `--port 8766` if the
default port is occupied, or `--prepare-only` to generate chunks without serving.
See `viewer/README.md` for controls and data conventions.

## Externally calibrated candidate

**Current conservative submission fallback: v1 below.** Later v2a/v3 experiments
did not pass all replacement checks. V3 improves fresh external spellF1 but narrowly
fails the missing-ball restart guard; see `possession/SPELL_CALIBRATION.md`.
`TECHNICAL_SUBMISSION.md` is the technical document draft. Minimal inference and
format-validation dependencies are pinned in `requirements-submission.txt`.

To check exact reproduction in a separate environment:

```bash
uv venv --python 3.11 /tmp/mtah-submission-check
uv pip install --python /tmp/mtah-submission-check/bin/python -r requirements-submission.txt
.venv/bin/python scripts/verify_reproduction.py --python /tmp/mtah-submission-check/bin/python
```

This exports only the required source and three automatic input files into a
temporary directory. It compares the regenerated CSV byte for byte with v1 and
saves `data/processed/reproduction_v1.json`. No challenge reference is read.

The fitted external stoppage model is included under `possession/`; regenerating
this candidate does not require downloading the external training matches:

```bash
.venv/bin/python -m possession.pipeline \
  --config possession/config_external_control_v1.json \
  --stoppage-model possession/model_external_stoppage_v1.json \
  --out evaluation/results/tactical_external_stoppage_v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_external_stoppage_v1.csv
```

This command uses only the provided tactical tracking and generated Smart Tagging
as challenge inputs. See `possession/STOPPAGE_CALIBRATION.md` for external-only
fitting, the fixed match split, held-out results and transfer limits. Follow the
organisers' later clarification: do not tune on challenge possession references
or compare algorithm output against the provided video. Format validation alone
does not measure accuracy.

The frozen external combined H/A/D and restart audit is documented in
`possession/JOINT_VALIDATION.md`. On prepared external assets, run
`.venv/bin/python -m possession.idsse_joint_reference --match J03WQQ` and then
`.venv/bin/python -m possession.validate_joint`. These create separate reference
and diagnostic files, leave the model and challenge candidate unchanged, and
refuse to overwrite completed results.

## Restart candidate v2a

The newer candidate has improved whole-match and overall restart metrics on a
fresh external match, with a known regression in missing-ball team selection.
See `possession/RESTART_IMPROVEMENT.md` for the comparison and limits.

```bash
.venv/bin/python -m possession.run_restart
.venv/bin/python evaluation/validate_submission.py evaluation/results/tactical_external_restart_v2a.csv
```

The runner uses the saved external weights and automatic tactical/tagging inputs.
It keeps earlier candidates and refuses to overwrite an existing v2a CSV. No
external training downloads are required to generate labels from the saved model.
