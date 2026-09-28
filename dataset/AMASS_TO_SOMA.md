# Fixed-identity AMASS to SOMA77 BVH

The converter discards AMASS identity during retargeting. It poses a neutral
SMPL v1 body with ten zero betas, fits every frame to one neutral, zero-beta
SOMA identity, and exports the repository's fixed 77-joint hierarchy at 30 Hz.
The input beta and gender are recorded in the manifest but never affect motion.

## Required data

- A licensed neutral SMPL v1.x model with 6,890 vertices, 24 joints, and at
  least ten beta components. Pass its file path with `--smpl-model-path`; do
  not commit it to this repository.
- SOMA-X 0.3.0 assets, either downloadable on first use or already present in
  the package asset cache. An explicit cache can be supplied with `--data-root`.
- BABEL JSON is not used by conversion. For later annotation alignment, keep
  BABEL v1.0 `train.json`, `val.json`, and `test.json` with the original AMASS
  root. The manifest preserves source-relative paths, FPS, and sampling stride.

## Environment

The current `kimodo` environment uses `py-soma-x==0.3.0` and its `usd-core`
dependency. Build the vendored Apache-2.0 MotionCorrection extension once:

```bash
conda run -n kimodo python -m pip install -e third_party/motion_correction
```

The converter locally aliases the stale `soma.fitting.body` import present in
the 0.3.0 wheel to `soma.body`; it does not modify the installed package.

## Conversion

Both input and output directories are mandatory. Only `*_stageii.npz` files
are considered; `stagei` identity files are ignored. The default path applies
MotionCorrection. Disable it with `--no-motion-correction`.

```bash
conda run -n kimodo python dataset/convert_amass_to_soma.py \
  --input-dir dataset/amass_sample \
  --output-dir dataset/amass_soma_bvh \
  --smpl-model-path /path/to/SMPL_NEUTRAL.pkl
```

An input is stride-sampled only when its half-up-rounded FPS is divisible by
30; all other rates are recorded as discards. Outputs mirror the input tree.
Successful conversions are appended to `conversion_manifest.jsonl`, while
discard and error details are appended to `errors.jsonl`.

## MotionJEPA preprocessing and visualization

The AMASS preprocessor validates the conversion manifest against every BVH,
keeps only complete 90-frame windows with 50% overlap, and places all windows
in the training split until BABEL annotations are joined:

```bash
conda run -n kimodo python dataset/preprocess_amass.py \
  --input-dir dataset/amass_soma_bvh \
  --output dataset/amass-soma-processed
```

The resulting NPY dataset can be opened by the existing viser viewer:

```bash
conda run -n kimodo python visualize_dataset.py \
  dataset/amass-soma-processed \
  --split train \
  --mesh
```

AMASS conversion already writes T-pose-relative rotations, so this preprocessor
does not apply the BONES-SEED rest-pose conversion a second time.

## BABEL-60/120 action preprocessing

After converting the full AMASS root, join the official BABEL train/val labels
into 150-frame action clips with either label vocabulary:

```bash
conda run -n kimodo python dataset/preprocess_babel.py \
  --subset 60 \
  --input-dir /data/seokhyeon/MotionJEPA/amass-soma-bvh \
  --output /data/seokhyeon/MotionJEPA/babel-60-processed
```

Use `--subset 120` for BABEL-120. The script reads annotations from
`dataset/babel-annotation` and label PKLs from `dataset/babel-60-and-120` by
default. Missing or upstream-discarded AMASS motions are logged and skipped.
The final partial chunk stores its valid length; `MotionDataset` zero-pads it
to 150 frames when loaded. Test remains empty because the released test PKLs
contain only `-1` labels.

The checked-in `action_label_2_idx.json` is copied from the BABEL reference
implementation at
`action_recognition/data/action_label_2_idx.json`; preprocessing performs no
runtime downloads.
