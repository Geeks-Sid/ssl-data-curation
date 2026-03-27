@REM # Stage 1
python main.py local-select --data_dir D:/IHC-found/Data --output_dir ./output_full --split train  --descriptor_backend cpu  --num_workers 60  --gpu_ids 0  --auto_reduce_gpu_workers  --rle_min_fraction 0.75 --local_keep_ratio 0.10 --local_keep_max 16 --semantic_weight 0.50 --interface_weight 0.30 --redundancy_weight 0.20 --nuisance_weight 0.35 >> logs_cpu_112.txt
@REM # Stage 2
@REM # python main.py global-select --candidate_dir ./output/candidates --output_dir ./out/global_selection --target_size 1000 --bin_columns tissue,is_cancer,state_bin,interface_bin --bin_alpha 0.5 --export_tars --data_dir ./Dummy
@REM # Stage 3
@REM # python main.py pack-tars --final_selection_dir ./output/global_selection/final_selection --output_dir ./output/global_selection/final_selection_tars --data_dir Data --output_mode files --image_output_dir ./output/global_selection/final_selection_debug_images --image_format jpg
@REM # Stage 4
@REM # export-images reads original Arrow shards, not global-selection parquet manifests
@REM # python main.py export-images --data_dir ./Dummy --output_dir patchselect/out/exported_images --split train --limit 500
@REM # Stage 5
@REM # python main.py benchmark --data_dir ./Dummy --split train --limit_images 8 --warmup_images 1 --worker_counts 1,2,4,8,16,32,64,96,128 --backend cpu --output_json patchselect/out/benchmark_backend.json --output_plot patchselect/out/benchmark_backend_scaling.png
