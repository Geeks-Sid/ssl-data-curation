# Patchselect

`patchselect` is a staged patch curation pipeline for large Arrow-backed IHC datasets.

It is designed for the regime described in this workspace:

- roughly `10M` source images
- each image tiled into `256x256` patches
- roughly `400M` non-empty patches after tissue filtering
- a hard training budget where patch curation must happen before expensive SSL

## Workflow

1. `local-select`
   - loads Arrow shards one image at a time
   - tiles each image into `256x256` patches
   - computes a cheap stain-aware descriptor for every non-empty patch
   - adds neighborhood/interface features from adjacent patches
   - keeps a small local coreset per image with rarity, interface, and quality scoring

2. `global-select`
   - reads locally selected candidate parquet files
   - rebalances the candidate pool over metadata and stain-state bins
   - writes final parquet shards for downstream training

3. `export-images`
   - helper for full-image inspection from Arrow shards

## Descriptor

Each retained patch gets a `48D` descriptor:

- `42D` base descriptor
  - tissue fraction
  - DAB and hematoxylin optical-density statistics
  - positive area fractions at multiple thresholds
  - nuclei-density and localization proxies
  - blur / fold / artifact proxies
  - texture and border occupancy
- `6D` neighborhood descriptor
  - semantic difference to adjacent patches
  - positivity change to adjacent patches
  - coarse stain-state transition fraction

The exact feature names are defined in [constants.py](/D:/FMIHCS/ssl-data-curation/patchselect/constants.py).

## Example

Local candidate generation:

```bash
python -m patchselect local-select ^
  --data_dir Data ^
  --output_dir patchselect/out/local_selection ^
  --split train ^
  --local_keep_ratio 0.10 ^
  --local_keep_max 4
```

Global balancing:

```bash
python -m patchselect global-select ^
  --candidate_dir patchselect/out/local_selection/candidates ^
  --output_dir patchselect/out/global_selection ^
  --target_size 10000000 ^
  --bin_columns tissue,cell_type,state_bin ^
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
- local utility scores
- all `48` descriptor features

`global-select` writes partitioned parquet files under:

- `patchselect/out/global_selection/final_selection/`

These final parquet shards can be used as a training manifest for an on-demand crop loader, or converted into exported patch files if needed.
