# RE10K indices (canonical)

This folder contains the **canonical JSON indices** used across **V1/V2/V3**.

## Files

- `dataset_index_train.json`
  - Mapping `{scene_id -> chunk_filename.torch}` for the **train split**.
  - This is the same data MVSplat expects at `dataset/re10k/train/index.json`.

- `dataset_index_eval.json`
  - Mapping `{scene_id -> chunk_filename.torch}` for the **evaluation split** (stored under `dataset/re10k/test/`).
  - This is the same data MVSplat expects at `dataset/re10k/test/index.json`.

- `evaluation_index_re10k.json`
  - Mapping `{scene_id -> {context: [i,j], target: [k,l,m]}}` used by the **evaluation view sampler**
    (2 context views + 3 target views per scene).
  - Used for *fair*, fixed-index evaluation across V1/V2/V3.

- `SHA256SUMS`
  - Checksums for the canonical index files.

## Re-generating

- Dataset split indices (train/eval, where eval = `dataset/re10k/test/`):

```bash
python scripts/indices/build_re10k_dataset_index.py --stage train --write-dataset-index
python scripts/indices/build_re10k_dataset_index.py --stage eval  --write-dataset-index
```

- Evaluation view-sampler index (optional):

```bash
python scripts/indices/generate_re10k_evaluation_index.py
```

Notes:
- The evaluation-index generator uses MVSplat's upstream implementation and depends on your environment + dataset.
- For the paper/benchmark story, prefer using the committed `evaluation_index_re10k.json` and treating it as fixed.
