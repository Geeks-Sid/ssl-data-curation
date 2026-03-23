# Patchselect

`patchselect` is a staged patch re-curation pipeline for large Arrow-backed IHC datasets.

It is designed for the regime described in this workspace:

- roughly `10M` source images
- each image tiled into `256x256` patches
- roughly `400M` non-empty patches after tissue filtering
- a hard training budget where patch re-curation must happen before expensive SSL

The method is framed as **budgeted semantic coverage**, not generic patch diversity.

It aims to maximize compute-constrained coverage of semantically meaningful IHC states and spatial interfaces while suppressing nuisance variation and redundancy.

## Objective

`patchselect` uses a practical surrogate for:

```text
max_{S: |S| <= B}
  semantic_coverage(S)
  + lambda * interface_coverage(S)
  - alpha * redundancy(S)
  - beta * nuisance(S)
```

subject to per-image caps in the local stage and metadata-aware bin quotas in the global stage.

In the current implementation:

- patch-level scoring is **target-label-free**
- metadata such as `tissue`, `is_cancer`, `gene`, and diagnosis text are used only for balancing or analysis
- image-level `rle_mask` is used only as a foreground/background pre-filter when present
- the problem is explicitly **patch-level re-curation after tiling**, not image-level dataset cleaning
- the objective weights are exposed as CLI and config parameters for reproducible sweeps

## Workflow

1. `local-select`
   - loads Arrow shards one image at a time
   - tiles each image into `256x256` patches
   - optionally decodes image-level `rle_mask` metadata and skips patches with less than `75%` foreground overlap
   - computes a cheap target-label-free descriptor for every non-empty patch
   - adds neighborhood/interface features from adjacent patches
   - scores each patch with semantic coverage, interface gain, redundancy penalty, nuisance score, and final objective
   - keeps a role-based local coreset per image:
     - `prototype`
     - `positive_tail`
     - `interface`
     - `rare_state`

2. `global-select`
   - reads locally selected candidate parquet files
   - rebalances the candidate pool over metadata groups and semantic/interface bins
   - writes final parquet shards for downstream training

3. `export-images`
   - helper for full-image inspection from Arrow shards

## Descriptor

Each retained patch gets a `48D` descriptor. The descriptor is implementation detail, not the method claim.

- `42D` base descriptor
  - stain statistics
    - DAB and hematoxylin optical-density quantiles
    - soft histograms over normalized hematoxylin and DAB channels
  - morphology / texture
    - nuclei-density proxies
    - localization proxies for nuclear / peri-nuclear / extra-nuclear staining
    - edge and texture measures
  - nuisance indicators
    - holes
    - unexpected color residuals
    - fold-like dark smooth regions
    - border occupancy
- `6D` neighborhood descriptor
  - semantic difference to adjacent patches
  - positivity change to adjacent patches
  - coarse stain-state transition fraction

Artifacts are treated as penalties or filters, not semantic coverage axes.

If your Arrow metadata contains an image-level `rle_mask`, `patchselect` intersects it with the stain-derived tissue mask and, by default, keeps only patches with at least `75%` foreground overlap before descriptor extraction. This is a compute optimization, not a semantic target.

The exact feature names are defined in [constants.py](/D:/FMIHCS/ssl-data-curation/patchselect/constants.py).

## Example

Local candidate generation:

```bash
 python -m patchselect local-select ^
   --data_dir Data ^
   --output_dir patchselect/out/local_selection ^
   --split train ^
   --rle_min_fraction 0.75 ^
   --local_keep_ratio 0.10 ^
   --local_keep_max 4 ^
   --semantic_weight 0.50 ^
   --interface_weight 0.30 ^
   --redundancy_weight 0.20 ^
   --nuisance_weight 0.35
```

Global balancing:

```bash
python -m patchselect global-select ^
  --candidate_dir patchselect/out/local_selection/candidates ^
  --output_dir patchselect/out/global_selection ^
  --target_size 10000000 ^
  --bin_columns tissue,is_cancer,state_bin,interface_bin ^
  --bin_alpha 0.5
```

Full-image export for inspection:

```bash
python -m patchselect export-images ^
  --data_dir Data ^
  --output_dir patchselect/out/exported_images ^
  --split eval ^
  --limit 500
```

## Output

`local-select` writes parquet files under:

- `patchselect/out/local_selection/candidates/`

Each row contains:

- patch coordinates
- sample and shard identifiers
- canonical metadata fields
- optional RLE foreground coverage fields
- local role assignments
- objective terms:
  - `objective_score`
  - `semantic_coverage_score`
  - `interface_score`
  - `redundancy_penalty`
  - `nuisance_score`
- all `48` descriptor features

`global-select` writes partitioned parquet files under:

- `patchselect/out/global_selection/final_selection/`

These final parquet shards can be used as a training manifest for an on-demand crop loader, or converted into exported patch files if needed.

## Recommended Evaluation Framing

For a paper, evaluate on a **compute frontier** rather than a single number:

- random patch sampling
- per-image random retention
- generic embedding-balanced curation
- state-only selection
- state + interface selection

at fixed pretraining GPU-hours and fixed downstream recipes.
