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
- descriptor extraction can run on the default CPU path or on an optional `cuCIM/CuPy` backend

## Workflow

1. `local-select`
   - loads Arrow shards one image at a time
   - tiles each image into `256x256` patches
   - optionally decodes image-level `rle_mask` metadata and skips patches with less than `75%` foreground overlap
   - computes a target-label-free descriptor for every non-empty patch at full patch resolution by default
   - computes slide-level stain normalization statistics on the full source image by default
   - can batch the descriptor stage on GPU with `--descriptor_backend cucim`
   - can process images with multiple worker processes via `--num_workers`
   - can reduce `cucim` worker count after GPU OOM with `--auto_reduce_gpu_workers`
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
   - can optionally pack the final selected crops into image-only tar archives grouped by source Arrow shard

3. `export-images`
   - helper for full-image inspection from Arrow shards

4. `pack-tars`
   - exports the final selected manifest into `.tar`, `.tar.gz`, loose image files, or both
   - groups outputs by source Arrow shard
   - stores patch images only, with no JSON metadata sidecars

5. `benchmark`
   - measures the full local-selection path on identical images for `cpu` vs `cucim`
   - supports either Arrow-backed samples or synthetic fallback images

## Stage Inputs And Outputs

### 1. `local-select`

Input:

- a directory of Arrow shards, typically `Data/*.arrow`
- each Arrow row is expected to include image bytes under `jpg.bytes`
- optional metadata in `json` / `custom_metadata`, including fields like `md5`, `gene`, `tissue`, `cell_type`, `diagnosis`, `is_cancer`
- optional image-level `rle_mask` metadata for foreground pre-filtering

Output:

- parquet parts under `patchselect/out/local_selection/candidates/`
- optional debug crops under `patchselect/out/local_selection/selected_patches/` when `--save_selected_patches` is enabled
- a run summary JSON at `patchselect/out/local_selection/run_summary.json`

What one output row represents:

- one retained patch from one source image
- the patch coordinates and source identifiers needed to recrop later
- normalized metadata fields
- local role assignment such as `prototype` or `interface`
- local objective terms and all descriptor features

### 2. `global-select`

Input:

- parquet candidate files from `local-select`
- balancing configuration such as `--target_size`, `--bin_columns`, `--bin_alpha`, and `--utility_column`

Output:

- final parquet manifest shards under `patchselect/out/global_selection/final_selection/`
- a run summary JSON at `patchselect/out/global_selection/run_summary.json`
- optional tar archives and/or loose patch image files under `patchselect/out/global_selection/final_selection_tars/` when `--export_tars` is enabled

What one output row represents:

- one globally retained patch chosen from the local candidate pool
- the same patch-level manifest fields from `local-select`, now filtered to the globally balanced final set

### 3. `pack-tars`

Input:

- final-selection parquet manifest files from `global-select`
- access to the source Arrow shards, either via stored `source_shard` paths or `--data_dir`

Output:

- one `.tar` or `.tar.gz` file per source Arrow shard under `patchselect/out/global_selection/final_selection_tars/` when tar output is enabled
- optional loose image files under `patchselect/out/global_selection/final_selection_tars/debug_images/` when file output is enabled
- a tar export summary JSON at `patchselect/out/global_selection/final_selection_tars/tar_export_summary.json`

What each tar contains:

- image patch files only
- filenames encoding sample slug, source index, patch index, and crop coordinates
- no JSON sidecars or metadata payloads inside the tar

### 4. `export-images`

Input:

- a directory of Arrow shards

Output:

- full exported images for inspection under `out/exported_images/` or your chosen `--output_dir`

What each output file represents:

- one original image row copied out of the Arrow dataset for manual inspection

### 5. `benchmark`

Input:

- either Arrow-backed images or synthetic images
- one or both descriptor backends: `cpu`, `cucim`

Output:

- timing and throughput metrics printed to stdout
- optional benchmark JSON when `--output_json` is provided
- optional worker-scaling plot when `--output_plot` is provided

What the output summarizes:

- end-to-end local-selection runtime for the chosen backend configuration
- comparable measurements across CPU and GPU backends on the same image set
- worker-scaling curves so you can compare CPU vs GPU behavior and identify the best worker count

## Backends

CPU is the default backend and requires only the base dependencies already used by `patchselect`.

The optional GPU backend is selected with:

```bash
python -m patchselect local-select --descriptor_backend cucim ...
```

The GPU path is designed for the descriptor stage and uses a `cuCIM/CuPy` stack. The current environment in this workspace does not have those packages installed, so the backend will report as unavailable until you install them in your CUDA-matched environment.

For multiprocessing:

```bash
python -m patchselect local-select --num_workers 8 ...
python -m patchselect local-select --descriptor_backend cucim --num_workers 8 --gpu_ids 0,1,2,3 --auto_reduce_gpu_workers ...
```

The GPU retry logic is chunk-based: if a `cucim` chunk OOMs, `patchselect` retries that chunk with one fewer worker and keeps the lower worker count for subsequent chunks.

Multiprocessing requires a normal local Python environment. Restricted sandboxes may block worker process creation, in which case `patchselect` will tell you to rerun with `--num_workers 1`.

## Descriptor

Each retained patch gets a `48D` descriptor. The descriptor is implementation detail, not the method claim.

By default, the patch descriptor runs on the original patch resolution, and slide-level stain normalization runs on the full source image. `--downsample_size` and `--slide_stats_size` are now optional and should be used only if you explicitly want faster lower-resolution ablations.

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
    --descriptor_backend cucim ^
    --num_workers 8 ^
    --gpu_ids 0,1,2,3 ^
    --auto_reduce_gpu_workers ^
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
  --bin_alpha 0.5 ^
  --export_tars ^
  --data_dir Data
```

Tar packing from an existing final-selection manifest:

```bash
python -m patchselect pack-tars ^
  --final_selection_dir patchselect/out/global_selection/final_selection ^
  --output_dir patchselect/out/global_selection/final_selection_tars ^
  --data_dir Data ^
  --image_format jpg ^
  --compression none
```

Loose debug image export:

```bash
python -m patchselect pack-tars ^
  --final_selection_dir patchselect/out/global_selection/final_selection ^
  --output_dir patchselect/out/global_selection/final_selection_tars ^
  --data_dir Data ^
  --output_mode files ^
  --image_output_dir patchselect/out/global_selection/final_selection_debug_images ^
  --image_format jpg
```

Full-image export for inspection:

```bash
python -m patchselect export-images ^
  --data_dir Data ^
  --output_dir patchselect/out/exported_images ^
  --split eval ^
  --limit 500
```

Backend benchmark:

```bash
python -m patchselect benchmark ^
  --data_dir Data ^
  --split train ^
  --limit_images 8 ^
  --warmup_images 1 ^
  --worker_counts 1,2,4,8 ^
  --backend both ^
  --output_json patchselect/out/benchmark_backend.json ^
  --output_plot patchselect/out/benchmark_backend_scaling.png
```

## Output

`local-select` writes parquet files under:

- `patchselect/out/local_selection/candidates/`

Each row contains:

- patch coordinates
- sample and shard identifiers
- canonical metadata fields
- optional RLE foreground coverage fields
- descriptor backend provenance
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

If `--export_tars` is enabled during `global-select`, or if you run `pack-tars` afterward, `patchselect` also writes:

- `patchselect/out/global_selection/final_selection_tars/`

By default, this directory contains one plain `.tar` archive per source Arrow shard. Each tar stores only patch image members, which fits common WebDataset-style SSL training setups. No JSON sidecars are written inside the tar archives.

For debugging, you can switch to loose image export with `--output_mode files` or write both tar archives and loose files with `--output_mode both`. Loose files are grouped by source shard under the configured image output directory.

## Recommended Evaluation Framing

For a paper, evaluate on a **compute frontier** rather than a single number:

- random patch sampling
- per-image random retention
- generic embedding-balanced curation
- state-only selection
- state + interface selection

at fixed pretraining GPU-hours and fixed downstream recipes.
